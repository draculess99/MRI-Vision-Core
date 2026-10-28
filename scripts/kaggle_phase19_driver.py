"""Kaggle driver, Phase 19A (PREPARED, NOT LAUNCHED): plain ConvNeXt-T (arm D) on the registered Phase 17 split, ACL + Medial Meniscus only.

Same manifest (v0.1.0, original B0 labels), same 1,200-study pool and seed-201 validation partition as Phase 17A, so the output can be compared paired with the preserved
Phase 17A B0/C1 predictions and the metadata baselines locally (scripts/analyze_phase19.py). The ROI arm (E) is NOT run here: it needs independent ROI boxes that do not exist
(see docs/phase19_convnext_reproduction.md); the driver only records that. Reference labels are the weak labels: descriptive only, no claim. No pretrained weights unless a LOCAL
weights file is attached with PROVENANCE kind phase19_weights (none exists; random initialisation is the default and the stated limitation).
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
CACHE = pathlib.Path(os.environ.get("P19_CACHE", "/tmp/decode_cache"))
DRY = bool(os.environ.get("P19_DRY"))
SEED = int(os.environ.get("P19_SEED", "201"))
EPOCHS = int(os.environ.get("P19_EPOCHS", "12"))
IMAGE_SIZE = int(os.environ.get("P19_IMAGE_SIZE", "256"))
BATCH = int(os.environ.get("P19_BATCH", "2"))
TINY = bool(os.environ.get("P19_TINY"))  # CPU rehearsal with a small ConvNeXt
OUT = WORKING / "phase19"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"experiment": "phase19a_convnext_plain", "weak_label_descriptive_only": True, "claim_allowed": False, "stage": "start", "stopped": None, "seed": SEED,
           "roi_arm": {"run": False, "reason": "no independent ROI boxes or detector available (see docs/phase19_convnext_reproduction.md)"}}
T0 = time.perf_counter()


def save():
    SUMMARY["wall_seconds"] = round(time.perf_counter() - T0, 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    sys.exit(0)


def run(cmd, name, timeout=6900):
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


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _find_bundle(base, max_depth=3, prune=("train_series", "test_series")):
    found, frontier = [], [base]
    for depth in range(max_depth + 1):
        nxt = []
        for directory in frontier:
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                if entry.is_file() and entry.name == "BUNDLE_FILES.json":
                    found.append(entry.path)
                elif entry.is_dir() and entry.name not in prune and depth < max_depth:
                    nxt.append(entry.path)
        frontier = nxt
    return sorted(found)


side = _find_bundle(pathlib.Path(os.environ.get("BUNDLE_DIR", INPUT)))
if not side:
    stop("code dataset with BUNDLE_FILES.json is not attached")
expected = json.loads(pathlib.Path(side[0]).read_text())
root = pathlib.Path(side[0]).parent
code = WORKING / "code"
zips = glob.glob(str(root / "**" / "rsna_weak_label_code.zip"), recursive=True)
if zips:
    import zipfile
    zipfile.ZipFile(zips[0]).extractall(code)
else:
    for rel in expected["files"]:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(next(root.rglob("kaggle_weak_label_experiment.py")).parent.parent / rel, code / rel)
actual = {str(p.relative_to(code)).replace("\\", "/"): sha(p) for p in code.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
if actual != expected["files"]:
    stop("code differs from the verified bundle")
SUMMARY["bundle"] = {"git_commit": expected["git_commit"], "files": len(actual)}
sys.path[:0] = [str(code), str(code / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402
from mri_core import weak_label_confound as wc  # noqa: E402
from mri_core.kaggle_inputs import InputsError, locate_inputs_package  # noqa: E402
from mri_core.rsna_knee_dataset import TARGET_COLUMNS, load_rsna_metadata  # noqa: E402
from mri_core.weak_label_data import assert_no_leakage, hash_split, load_weak_manifest  # noqa: E402

REG = json.loads((code / "configs" / "phase17_registry.json").read_text())
if str(SEED) not in REG["partitions"]:
    stop(f"seed {SEED} is not a registered Phase 17 seed")
meta_dir, dicom_root = discover_data(INPUT)
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1)}
if not SUMMARY["environment"]["cuda_available"] and not DRY:
    stop("no CUDA device available")
metadata = load_rsna_metadata(meta_dir, dicom_root)
try:
    IN, PROV = locate_inputs_package(INPUT, "phase17a_inputs", override=os.environ.get("P19_INPUTS"))
except InputsError as error:
    SUMMARY["inputs_diagnosis"] = {"code": error.code, **error.diagnostics}
    stop(f"inputs package problem ({error.code}): {error}")
selection = json.loads((IN / "selected_studies.json").read_text())
if sha_ids(selection) != REG["training_pool"]["sha256"] and not DRY:
    stop("selection differs from the registered training pool")
features = pd.read_csv(IN / "header_features.csv", index_col=0)

manifest_path = OUT / "manifest" / "weak_manifest_tier1.jsonl"
if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv", "--test-csv", meta_dir / "test.csv",
        "--sample-submission", meta_dir / "sample_submission.csv", "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus", "--output", manifest_path], "manifest").returncode != 0:
    stop("manifest regeneration failed")
manifest = load_weak_manifest(manifest_path)
if manifest.sha256 != REG["manifest_sha256"] and not DRY:
    stop("manifest differs from the registered manifest")
order = [u for u in manifest.ids if u in set(selection)]
try:
    tr, va = hash_split(order, 0.2, SEED)
except ValueError:
    stop("empty split (rehearsal data too small)")
train_ids, val_ids = [order[i] for i in tr], [order[i] for i in va]
if sha_ids(val_ids) != REG["partitions"][str(SEED)]["validation_sha256"] and not DRY:
    stop("validation partition differs from the registry")
test_ids = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
assert_no_leakage(train_ids, val_ids, gold_ids=set(metadata.complete_label_train["StudyInstanceUID"].astype(str)), test_ids=test_ids)
SUMMARY["partition"] = {"train": len(train_ids), "validation": len(val_ids), "validation_sha256": sha_ids(val_ids)}
(OUT / "selected_studies.json").write_text(json.dumps(selection))

base = json.loads((code / "configs" / "weak_label_full_tier1.json").read_text())
cfg = {**base, "arm": "D", "amp": True, "val_fraction": 0.2, "seed": SEED, "batch_size": BATCH, "epochs": EPOCHS, "early_stopping_patience": 0, "learning_rate": 1e-4,
       "weight_decay": 0.05, "grad_clip": 1.0, "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE), "image_size": IMAGE_SIZE, "slices_per_series": 9,
       "reliability_weights": {"ACL": 1.0, "Medial Meniscus": 1.0, "Lateral Meniscus": 0.0}, "drop_path": 0.1, "pretrained_weights": None}
if TINY:
    cfg.update({"convnext_depths": [1, 1, 1, 1], "convnext_dims": [8, 16, 32, 64]})
(OUT / "configs").mkdir(exist_ok=True)
cfg_path = OUT / "configs" / "arm_D.json"
cfg_path.write_text(json.dumps(cfg, indent=2))
SUMMARY["config"] = cfg
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]
COMMON = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest_path]

SUMMARY["stage"] = "cache"
if run(EXP + ["cache-warm-multi", *COMMON, "--config", cfg_path, "--variants", f"{IMAGE_SIZE}x9", "--include-studies", OUT / "selected_studies.json", "--work-dir", OUT / "warm"],
       "cache_warm").returncode != 0:
    stop("cache build failed")
EXCLUDED = OUT / "warm" / "excluded_unreadable.json"
SUMMARY["stage"] = "train"
work = OUT / "runs" / f"s{SEED}_D"
if run(EXP + ["train", *COMMON, "--config", cfg_path, "--include-studies", OUT / "selected_studies.json", "--exclude-studies", EXCLUDED, "--work-dir", work], "train_D").returncode != 0:
    stop("training failed")
state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
rec = {"epochs": state["epoch"], "global_step": state["global_step"], "dataset_fingerprint": state["dataset_fingerprint"], "validation_sha256": sha_ids(state["valid_study_ids"]),
       "last_pt_sha256": sha(work / "last.pt"), "keys_present": {k: state.get(k) is not None for k in ("model", "optimizer", "scheduler", "scaler", "torch_rng", "python_rng", "numpy_rng", "cuda_rng", "loader_rng", "config", "metrics")},
       "peak_cuda_gb": max((json.loads(l).get("cuda_peak_gb") or 0) for l in (work / "runtime_log.jsonl").read_text().splitlines() if '"event": "epoch"' in l),
       "epoch_seconds": [round(json.loads(l)["seconds"], 1) for l in (work / "runtime_log.jsonl").read_text().splitlines() if '"event": "epoch"' in l]}
if rec["validation_sha256"] != sha_ids(val_ids) and not DRY:  # (a rehearsal on a few local studies trains on a subset)
    stop("the trained run's validation set differs from the registered partition")
SUMMARY["run_D"] = rec
(work / "last_best.pt").unlink(missing_ok=True)
if run(EXP + ["score-validation", *COMMON, "--config", cfg_path, "--checkpoint", work / "last.pt", "--output", work / "validation_predictions.csv"], "score_D").returncode != 0:
    stop("validation scoring failed")

# metadata-only comparison fitted on the same training studies (ACL and Medial Meniscus; same header features as Phase 17A)
SUMMARY["stage"] = "metadata"
idx = {u: i for i, u in enumerate(manifest.ids)}
tr_ids = [u for u in train_ids if u in features.index]
ev_ids = [u for u in val_ids if u in features.index]
tv, tm = manifest.values[[idx[u] for u in tr_ids]], manifest.masks[[idx[u] for u in tr_ids]]
meta = wc.fit_predict_baselines(features, tr_ids, tv, tm, {"val": ev_ids})
pd.DataFrame({"study": ev_ids, **{f"{m}_{t.split()[0]}": meta[m]["val"][:, i] for m in meta for i, t in enumerate(wc.ACTIVE)}}).to_csv(OUT / "metadata_predictions.csv", index=False)
SUMMARY["metadata_baseline"] = "logistic regression and boosted tree on header features, fitted on the registered training partition only (predictions saved)"
SUMMARY["stage"] = "done"
save()
print(json.dumps({k: SUMMARY[k] for k in ("stage", "wall_seconds", "run_D")}, indent=1, default=str)[:1500])
sys.exit(0)
