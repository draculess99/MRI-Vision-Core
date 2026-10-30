"""Kaggle kernel, Phase 18 leaderboard attempt (code-competition notebook): weak-label C0 trained on ALL trainable report-derived targets.

One kernel does everything so that it can be submitted as-is and re-run by the competition on the hidden test set:
  1 manifest (rules 0.1.0, 11 active targets: every target except Synovitis, which has almost no negatives); gold and test studies are excluded by construction
  2 decode (128x9) of every eligible study, then train C0 for two seeds (701, 702): 12 epochs, lr 1e-4, clip 1.0, FP16 AMP, 80/20 hash split for monitoring
  3 resume check (re-running the training command must resume at epoch 12 and change nothing), integrity checks (no gold / test study in training or validation)
  4 score the 58 gold studies (never trained on) with each seed and with their mean, for the local pre-registered gate (the gate itself runs off-Kaggle)
  5 hidden-test inference: mean of the two seeds for the 11 targets, the preserved baseline (2-checkpoint mean) for Synovitis; writes /kaggle/working/submission.csv
No pretrained weights, no external data, no network. A failure at any step leaves no submission.csv.
"""

import glob
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

INPUT = pathlib.Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORKING = pathlib.Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
CACHE = pathlib.Path(os.environ.get("P18_CACHE", "/tmp/decode_cache"))
DRY = bool(os.environ.get("P18_DRY"))
EPOCHS = int(os.environ.get("P18_EPOCHS", "12"))
GOLD_LIMIT = int(os.environ.get("P18_GOLD_LIMIT", "0"))  # rehearsal only
SEEDS = [701, 702]
OUT = WORKING / "phase18"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"experiment": "phase18_w11", "stage": "start", "stopped": None, "seeds": SEEDS}
T0 = time.perf_counter()


def save():
    SUMMARY["wall_seconds"] = round(time.perf_counter() - T0, 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    sys.exit(1)  # a failed attempt must not look like a finished one (and must leave no submission.csv)


def run(cmd, name, timeout=7000):
    print(f"\n$ {' '.join(str(c) for c in cmd)[:200]}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-300:])
    if proc.returncode:
        print("[stderr]", proc.stderr[-1500:])
    SUMMARY.setdefault("steps", {})[name] = {"exit_code": proc.returncode, "seconds": round(time.perf_counter() - started, 1)}
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs" / f"{name}.log").write_text((proc.stdout or "") + "\n[stderr]\n" + (proc.stderr or ""))
    return proc


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_json(text):
    """The last top-level JSON object printed by a subprocess (None when there is none); handles output that starts with the object."""
    nl = chr(10)
    k = text.rfind(nl + "{" + nl)
    k = k + 1 if k >= 0 else (0 if text.startswith("{" + nl) else -1)
    if k < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[k:])[0]
    except ValueError:
        return None


def find(name, base, depth=3, prune=("train_series", "test_series")):
    found, frontier = [], [base]
    for d in range(depth + 1):
        nxt = []
        for directory in frontier:
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for e in entries:
                if e.is_file() and e.name == name:
                    found.append(pathlib.Path(e.path))
                elif e.is_dir() and e.name not in prune and d < depth:
                    nxt.append(e.path)
        frontier = nxt
    return sorted(found)


# ---- bundle and baseline checkpoints
bundles = find("BUNDLE_FILES.json", pathlib.Path(os.environ.get("BUNDLE_DIR", INPUT)))
if not bundles:
    stop("code dataset with BUNDLE_FILES.json is not attached")
expected = json.loads(bundles[0].read_text())
code = WORKING / "code"
import zipfile  # noqa: E402
_root = bundles[0].parent
_zips = glob.glob(str(_root / "**" / "rsna_weak_label_code.zip"), recursive=True)  # Kaggle may present the dataset zip already extracted
if _zips:
    zipfile.ZipFile(_zips[0]).extractall(code)
else:
    _tree = [p for p in _root.rglob("kaggle_weak_label_experiment.py")]
    if not _tree:
        stop("neither the code zip nor an extracted code tree was found in the code dataset")
    for _rel in expected["files"]:  # copy exactly the listed files (the dataset folder also holds BUNDLE_FILES.json and the like)
        (code / _rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_tree[0].parent.parent / _rel, code / _rel)
actual = {str(p.relative_to(code)).replace("\\", "/"): sha(p) for p in code.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
if actual != expected["files"]:
    stop("code differs from the verified bundle")
SUMMARY["bundle"] = {"git_commit": expected["git_commit"], "files": len(actual)}
sys.path[:0] = [str(code), str(code / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402
from mri_core.rsna_knee_dataset import PLANE_NAMES, TARGET_COLUMNS, load_rsna_metadata  # noqa: E402
from mri_core.rsna_knee_train import RSNAKneeDicomDataset  # noqa: E402
from mri_core.weak_label_submit import SubmissionError, combine, load_checkpoint_model, predict_test, validate_submission  # noqa: E402
from mri_core.weak_label_train import predict_probabilities  # noqa: E402

TARGETS11 = [t for t in TARGET_COLUMNS if t != "Synovitis"]
_bdir = pathlib.Path(os.environ.get("P18_BASELINE_DIR", INPUT))
baseline_ckpts = [find(n, _bdir)[0] for n in ("fold0_checkpoint_best.pt", "fold1_checkpoint_best.pt") if find(n, _bdir)]
if len(baseline_ckpts) != 2:
    stop("the two preserved baseline checkpoints (fold0/fold1_checkpoint_best.pt) are not attached")
SUMMARY["baseline_checkpoints"] = {p.name: sha(p) for p in baseline_ckpts}
meta_dir, dicom_root = discover_data(INPUT)
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1)}
if not SUMMARY["environment"]["cuda_available"] and not DRY:
    stop("no CUDA device available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metadata = load_rsna_metadata(meta_dir, dicom_root)
gold = metadata.complete_label_train.reset_index(drop=True)
gold_ids = set(gold["StudyInstanceUID"].astype(str))
test_ids = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
SUMMARY["data"] = {"gold_studies": len(gold_ids), "test_studies": len(test_ids)}

# ---- 1. manifest
manifest = OUT / "manifest" / "weak_manifest.jsonl"
if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv", "--test-csv", meta_dir / "test.csv",
        "--sample-submission", meta_dir / "sample_submission.csv", "--active-targets", ",".join(TARGETS11), "--output", manifest], "manifest").returncode != 0:
    stop("manifest build failed")
msum = json.loads(manifest.with_name(manifest.name + ".summary.json").read_text())
SUMMARY["manifest"] = {"sha256": msum["manifest_sha256"], "included": msum["counts"]["included"], "leakage_checks": msum["leakage_checks"],
                       "state_counts": {t: msum["state_counts"][t] for t in TARGETS11}}
if any(msum["leakage_checks"][k] for k in ("overlap_with_gold", "overlap_with_test", "overlap_with_holdout")) or msum["counts"]["excluded_has_structured_labels"] != len(gold_ids):
    stop("manifest leakage check failed (gold or test studies would be trainable)")
for t in TARGETS11:
    c = msum["state_counts"][t]
    if c["positive"] < 20 or c["negative"] < 20:
        stop(f"target {t} has too few decided classes ({c['positive']}/{c['negative']})")
save()

# ---- 2. decode + train two seeds
base = json.loads((code / "configs" / "weak_label_full_tier1.json").read_text())
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]
COMMON = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest]


def cfg_for(name, seed):
    cfg = {**base, "arm": "C", "amp": True, "val_fraction": 0.2, "seed": seed, "batch_size": 8, "epochs": EPOCHS, "early_stopping_patience": 0, "learning_rate": 1e-4,
           "grad_clip": 1.0, "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE), "image_size": 128, "slices_per_series": 9}
    path = OUT / "configs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


SUMMARY["stage"] = "cache"
if run(EXP + ["cache-warm-multi", *COMMON, "--config", cfg_for("warm", 0), "--variants", "128x9", "--work-dir", OUT / "warm"], "cache_warm").returncode != 0:
    stop("cache build failed")
SELECTED, EXCLUDED = OUT / "warm" / "selected_studies.json", OUT / "warm" / "excluded_unreadable.json"
SUMMARY["trainable_studies"] = len(json.loads(SELECTED.read_text()))
save()
SUMMARY["stage"] = "train"
runs, models = {}, {}
for seed in SEEDS:
    name = f"s{seed}"
    work = OUT / "runs" / name
    cmd = EXP + ["train", *COMMON, "--config", cfg_for(name, seed), "--include-studies", SELECTED, "--exclude-studies", EXCLUDED, "--arm", "C", "--work-dir", work]
    if run(cmd, f"train_{name}").returncode != 0:
        stop(f"training {name} failed")
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    hist = state["metrics"]["history"]
    seen = set(state["train_study_ids"]) | set(state["valid_study_ids"])
    runs[name] = {"epochs": state["epoch"], "global_step": state["global_step"], "last_pt_sha256": sha(work / "last.pt"), "last_best_sha256": sha(work / "last_best.pt"),
                  "train_studies": len(state["train_study_ids"]), "validation_studies": len(state["valid_study_ids"]), "dataset_fingerprint": state["dataset_fingerprint"],
                  "no_gold_in_training_or_validation": not (seen & gold_ids), "no_test_in_training_or_validation": not (seen & test_ids),
                  "keys_present": {k: state.get(k) is not None for k in ("model", "optimizer", "scheduler", "scaler", "torch_rng", "python_rng", "numpy_rng", "cuda_rng", "loader_rng", "config", "metrics")},
                  "weak_validation_final_macro_auc": next((h["val_macro_auc_weak"] for h in reversed(hist) if h.get("val_macro_auc_weak") is not None), None),
                  "weak_validation_final_loss": hist[-1]["val_loss"], "finite": all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"]) for h in hist)}
    before = runs[name]["last_pt_sha256"]
    resume = run(cmd, f"resume_check_{name}")  # a finished run must resume at its last epoch and leave the checkpoint untouched
    runs[name]["resume_check"] = {"exit_code": resume.returncode, "resumed_from_epoch": (last_json(resume.stdout) or {}).get("resumed_from_epoch") if resume.returncode == 0 else None,
                                  "checkpoint_unchanged": sha(work / "last.pt") == before}
    runs[name]["resume_check"]["ok"] = bool(resume.returncode == 0 and runs[name]["resume_check"]["resumed_from_epoch"] == EPOCHS and runs[name]["resume_check"]["checkpoint_unchanged"])
    models[name] = work / "last.pt"
    SUMMARY.setdefault("runs", {})[name] = runs[name]
    save()
if not all(r["no_gold_in_training_or_validation"] and r["no_test_in_training_or_validation"] and r["finite"] and r["resume_check"]["ok"] for r in runs.values()):
    stop("an integrity or resume check failed")

# ---- 4. gold scoring (studies never trained on): per seed and ensemble
SUMMARY["stage"] = "gold"
cfg_ds = {"image_size": 128, "slices_per_series": 9, "planes": list(PLANE_NAMES)}
gold_eval = gold.head(GOLD_LIMIT) if GOLD_LIMIT else gold
gold_preds = {}
for name, path in models.items():
    model, _ = load_checkpoint_model(path, device)
    ds = RSNAKneeDicomDataset(metadata, gold_eval, cfg_ds, split="train")
    uids, probs = predict_probabilities(model, ds, device, 8, num_workers=0)
    assert list(map(str, uids)) == list(gold_eval["StudyInstanceUID"].astype(str))
    gold_preds[name] = probs
    pd.DataFrame(probs, columns=list(TARGET_COLUMNS)).assign(StudyInstanceUID=list(uids))[["StudyInstanceUID", *TARGET_COLUMNS]].to_csv(OUT / f"gold_predictions_{name}.csv", index=False)
mean_gold = np.mean(list(gold_preds.values()), axis=0)
pd.DataFrame(mean_gold, columns=list(TARGET_COLUMNS)).assign(StudyInstanceUID=list(gold_eval["StudyInstanceUID"].astype(str)))[["StudyInstanceUID", *TARGET_COLUMNS]].to_csv(OUT / "gold_predictions_ensemble.csv", index=False)
SUMMARY["gold_scored_studies"] = len(gold_eval)
save()

# ---- 5. hidden-test inference and submission
SUMMARY["stage"] = "submission"
template = metadata.sample_submission
order = list(template["StudyInstanceUID"].astype(str))
failed_all = set()
base_runs = []
for path in baseline_ckpts:
    preds, failed = predict_test(path, metadata, device)
    base_runs.append(preds)
    failed_all |= failed
new_runs = []
for path in models.values():
    preds, failed = predict_test(path, metadata, device)
    new_runs.append(preds)
    failed_all |= failed
if any(u not in r for r in base_runs + new_runs for u in order):
    stop("a sample-submission study has no test metadata")
baseline = np.stack([np.mean([r[u] for r in base_runs], axis=0) for u in order])
new = np.stack([np.mean([r[u] for r in new_runs], axis=0) for u in order])
merged = combine(baseline, new, TARGETS11, 1.0)
if failed_all:
    if len(failed_all) / len(order) > float(os.environ.get("P18_MAX_FAILED", "0.05")):
        stop(f"{len(failed_all)}/{len(order)} test studies unreadable")
    good = np.array([u not in failed_all for u in order])
    merged[~good] = merged[good].mean(axis=0) if good.any() else 0.5
frame = pd.DataFrame({"StudyInstanceUID": template["StudyInstanceUID"].astype("string")})
for j, t in enumerate(TARGET_COLUMNS):
    frame[t] = merged[:, j].astype(float)
try:
    validate_submission(frame, template)
except SubmissionError as error:
    stop(f"submission validation failed: {error}")
frame.to_csv(WORKING / "submission.csv", index=False)
SUMMARY["submission"] = {"rows": len(frame), "sha256": sha(WORKING / "submission.csv"), "targets_from_new_model": TARGETS11, "targets_from_baseline": ["Synovitis"],
                         "unreadable_test_studies": len(failed_all)}
SUMMARY["stage"] = "done"
save()
print(json.dumps({k: SUMMARY[k] for k in ("stage", "wall_seconds", "submission")}, indent=1, default=str))
sys.exit(0)
