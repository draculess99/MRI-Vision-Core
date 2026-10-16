"""Kaggle driver, Phase 17A (PREPARED, NOT LAUNCHED): controlled comparison on the frozen candidate labels.

Arms per registered seed (201-205): B0 (C0, original weak-label mask), C1 (C0, candidate mask), S (C1 mask with labels shuffled across training
studies), M0 (metadata-only boosted tree / logistic regression on header features, same training studies), N (untrained C0, seeds 9001-9010).
Primary target Medial Meniscus; ACL and Lateral Meniscus secondary. Everything is scored against the manifest's WEAK labels on the registered
validation partitions, which is DESCRIPTIVE ONLY: no claim is made without an independent adjudicated reference (see docs/phase17a_experiment_protocol.md).
The Phase 16 hold-out pool, gold and test studies are not used. Private kernel, internet off. Resumable: attach a previous output (P17_PREV).
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
CACHE = pathlib.Path(os.environ.get("P17_CACHE", "/tmp/decode_cache"))
DRY = bool(os.environ.get("P17_DRY"))
HARD_LIMIT = float(os.environ.get("P17_HARD", "6900"))
ROUNDS = int(os.environ.get("P17_ROUNDS", "1000"))
OUT = WORKING / "phase17a"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"experiment": "phase17a", "weak_label_descriptive_only": True, "claim_allowed": False, "stage": "start", "stopped": None, "runs": {}}
T0 = time.perf_counter()


def elapsed():
    return time.perf_counter() - T0


def save():
    SUMMARY["wall_seconds"] = round(elapsed(), 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    sys.exit(0)  # a planned stop is not a kernel error; the reason is in final_summary.json


def run(cmd, name, timeout=3600):
    print(f"\n$ {' '.join(str(c) for c in cmd)[:220]}", flush=True)
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
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


# ---- bundle
side = glob.glob(str(pathlib.Path(os.environ.get("BUNDLE_DIR", INPUT)) / "**" / "BUNDLE_FILES.json"), recursive=True)
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
    shutil.copytree(next(p.parent.parent for p in root.rglob("kaggle_weak_label_experiment.py")), code, dirs_exist_ok=True)
actual = {str(p.relative_to(code)).replace("\\", "/"): sha(p) for p in code.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
if actual != expected["files"]:
    stop("code differs from the verified bundle")
SUMMARY["bundle"] = {"git_commit": expected["git_commit"], "files": len(actual)}
sys.path[:0] = [str(code), str(code / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402
from mri_core import weak_label_compare as wc_  # noqa: E402
from mri_core import weak_label_confound as wc  # noqa: E402
from mri_core import weak_label_mask_override as mo  # noqa: E402
from mri_core.rsna_knee_dataset import TARGET_COLUMNS, load_rsna_metadata  # noqa: E402
from mri_core.rsna_knee_train import seed_everything  # noqa: E402
from mri_core.weak_label_data import WeakLabelDicomDataset, assert_no_leakage, hash_split, load_weak_manifest  # noqa: E402
from mri_core.weak_label_models import build_model  # noqa: E402
from mri_core.weak_label_train import predict_probabilities  # noqa: E402

REG = json.loads((code / "configs" / "phase17_registry.json").read_text())
CAND = json.loads((code / "configs" / "phase17_candidate_v1.json").read_text())
SUMMARY["registry_sha256"] = sha(code / "configs" / "phase17_registry.json")
SEEDS = [int(x) for x in os.environ["P17_SEEDS"].split(",")] if os.environ.get("P17_SEEDS") else list(REG["seeds"])
if len(SEEDS) > 1 and not DRY and os.environ.get("P17_ALLOW_FULL") != "1":  # the five-seed experiment must be launched deliberately
    stop(f"{len(SEEDS)} seeds requested without P17_ALLOW_FULL=1; the full experiment needs explicit approval (one seed: P17_SEEDS=201)")
EPOCHS = int(os.environ.get("P17_EPOCHS", REG["model"]["epochs"]))
meta_dir, dicom_root = discover_data(INPUT)
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1)}
if not SUMMARY["environment"]["cuda_available"] and not DRY:
    stop("no CUDA device available")
metadata = load_rsna_metadata(meta_dir, dicom_root)

# ---- inputs package (private dataset) and frozen definitions
def _find_provenance():
    if os.environ.get("P17_INPUTS"):
        return glob.glob(str(pathlib.Path(os.environ["P17_INPUTS"]) / "PROVENANCE.json"))
    found = []
    for depth in ("*", "*/*", "*/*/*"):  # Kaggle mounts datasets at /kaggle/input/<slug>/ or a deeper owner/slug path
        found += glob.glob(str(INPUT / depth / "PROVENANCE.json"))
    return sorted(set(found))


prov_files = _find_provenance()
prov_files = [f for f in prov_files if json.loads(pathlib.Path(f).read_text()).get("kind") == "phase17a_inputs"]
if len(prov_files) != 1:
    listing = {str(p.relative_to(INPUT)): sorted(x.name for x in p.iterdir())[:8] for d in ("*", "*/*") for p in list(INPUT.glob(d))[:60] if p.is_dir()}
    SUMMARY["input_listing_for_diagnosis"] = dict(list(listing.items())[:40])  # directory names only
    stop(f"the private phase17a inputs dataset (PROVENANCE.json, kind phase17a_inputs) was found {len(prov_files)} times, expected exactly once")
IN = pathlib.Path(prov_files[0]).parent
PROV = json.loads((IN / "PROVENANCE.json").read_text())
for name, info in PROV["files"].items():
    if sha(IN / name) != info["sha256"]:
        stop(f"{name} does not match PROVENANCE.json")
if PROV["files"]["candidate_mask_v1.json"]["sha256"] != CAND["override_file_sha256"]:
    stop("candidate mask file differs from the frozen candidate definition")
override = mo.load_override(IN / "candidate_mask_v1.json")
if override["fingerprint"] != REG["override_fingerprint_sha256"]:
    stop("candidate mask fingerprint differs from the registry")
selection = json.loads((IN / "selected_studies.json").read_text())
SUMMARY["inputs"] = {"provenance_sha256": sha(IN / "PROVENANCE.json"), "override_fingerprint": override["fingerprint"],
                     "selection_sha256": sha_ids(selection), "candidate_counts": CAND["counts_per_target"]}
if sha_ids(selection) != REG["training_pool"]["sha256"] and not DRY:
    stop("selection differs from the registered training pool")

manifest_path = OUT / "manifest" / "weak_manifest_tier1.jsonl"
if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv", "--test-csv", meta_dir / "test.csv",
        "--sample-submission", meta_dir / "sample_submission.csv", "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus", "--output", manifest_path], "manifest").returncode != 0:
    stop("manifest regeneration failed")
manifest = load_weak_manifest(manifest_path)
if manifest.sha256 != REG["manifest_sha256"] and not DRY:
    stop("manifest differs from the registered manifest")
if override["manifest_sha256"] != manifest.sha256 and not DRY:
    stop("candidate mask was built for a different manifest")
SUMMARY["manifest_sha256"] = manifest.sha256
index = {u: i for i, u in enumerate(manifest.ids)}
order = [u for u in manifest.ids if u in set(selection)]
test_ids = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
gold_ids = set(metadata.complete_label_train["StudyInstanceUID"].astype(str))
pool = set(manifest.ids) - set(selection)  # includes the inspected Phase 16 hold-out pool: never trained on, validated on or used as confirmation
partitions = {}
for seed in SEEDS:
    try:
        tr, va = hash_split(order, 0.2, seed)
    except ValueError:
        if DRY:
            continue
        raise
    train_ids, val_ids = [order[i] for i in tr], [order[i] for i in va]
    if sha_ids(val_ids) != REG["partitions"][str(seed)]["validation_sha256"] and not DRY:
        stop(f"seed {seed}: validation partition differs from the registry")
    assert_no_leakage(train_ids, val_ids, gold_ids=gold_ids, test_ids=test_ids)
    assert not (set(train_ids) | set(val_ids)) & pool, "training/validation overlaps the protected pool"
    partitions[seed] = {"train": train_ids, "val": val_ids}
SEEDS = [s for s in SEEDS if s in partitions]
SUMMARY["leakage"] = {"partitions": len(partitions), "gold_overlap": 0, "test_overlap": 0, "overlap_with_phase16_holdout_pool": 0}
save()

base = json.loads((code / "configs" / "weak_label_full_tier1.json").read_text())
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]
COMMON = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest_path]
(OUT / "selected_studies.json").write_text(json.dumps(selection))


def cfg_for(name, seed):
    m = REG["model"]
    cfg = {**base, "arm": "C", "amp": True, "val_fraction": m["val_fraction"], "seed": seed, "batch_size": m["batch_size"], "epochs": EPOCHS, "early_stopping_patience": 0,
           "learning_rate": m["learning_rate"], "grad_clip": m["grad_clip"], "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE),
           "image_size": 128, "slices_per_series": 9}
    path = OUT / "configs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


SUMMARY["stage"] = "cache"
if run(EXP + ["cache-warm-multi", *COMMON, "--config", cfg_for("warm", 0), "--variants", "128x9", "--include-studies", OUT / "selected_studies.json",
              "--work-dir", OUT / "warm"], "cache_warm", timeout=int(max(600, HARD_LIMIT - elapsed()))).returncode != 0:
    stop("cache build failed")
EXCLUDED = OUT / "warm" / "excluded_unreadable.json"
unreadable = set(json.loads(EXCLUDED.read_text()))
SUMMARY["unreadable_studies"] = len(unreadable)

# ---- resume support
_prev = os.environ.get("P17_PREV")
for prev in glob.glob(str(pathlib.Path(_prev) / "phase17a" / "runs" / "*")) if _prev else glob.glob(str(INPUT / "*" / "phase17a" / "runs" / "*")):
    prev = pathlib.Path(prev)
    target = OUT / "runs" / prev.name
    if target.exists() or not prev.is_dir():
        continue
    if (prev / "result.json").is_file() or (prev / "last.pt").is_file():
        shutil.copytree(prev, target)
        SUMMARY.setdefault("restored_from_previous_session", []).append(prev.name)


def one_run(seed, arm):
    name = f"s{seed}_{arm}"
    work = OUT / "runs" / name
    if (work / "result.json").is_file():
        return json.loads((work / "result.json").read_text())
    extra = []
    if arm in ("C1", "S"):
        extra += ["--label-mask", "C1", "--label-mask-file", IN / "candidate_mask_v1.json", "--expected-mask-fingerprint", REG["override_fingerprint_sha256"]]
    if arm == "S":
        extra += ["--shuffle-train-labels", REG["shuffle_seed_offset"] + seed]
    proc = run(EXP + ["train", *COMMON, "--config", cfg_for(name, seed), "--include-studies", OUT / "selected_studies.json", "--exclude-studies", EXCLUDED,
                      "--arm", "C", "--work-dir", work, *extra], f"train_{name}", timeout=int(max(300, HARD_LIMIT - elapsed())))
    if proc.returncode != 0:
        return {"failure": (proc.stderr or proc.stdout)[-800:]}
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    rec = {"name": name, "seed": seed, "arm": arm, "epochs": state["epoch"], "dataset_fingerprint": state["dataset_fingerprint"],
           "validation_sha256": sha_ids(state["valid_study_ids"]), "train_studies": len(state["train_study_ids"]), "validation_studies": len(state["valid_study_ids"]),
           "leakage": json.loads((work / "leakage.json").read_text()), "last_pt_sha256": sha(work / "last.pt")}
    if not DRY and rec["validation_sha256"] != REG["partitions"][str(seed)]["validation_sha256"]:
        return {**rec, "failure": "validation partition differs from the registry", "fatal": True}
    if not rec["leakage"]["clean"]:
        return {**rec, "failure": "leakage report not clean", "fatal": True}
    if run(EXP + ["score-validation", *COMMON, "--config", cfg_for(name, seed), "--checkpoint", work / "last.pt", "--output", work / "validation_predictions.csv"],
           f"score_{name}").returncode != 0:
        return {**rec, "failure": "validation scoring failed"}
    (work / "result.json").write_text(json.dumps(rec, indent=2, default=str))
    (work / "last_best.pt").unlink(missing_ok=True)
    if arm == "S":  # shuffled control: hash only
        (work / "last.pt").unlink(missing_ok=True)
    return rec


SUMMARY["stage"] = "runs"
failures, block_seconds = 0, None
for seed in SEEDS:
    if block_seconds is not None and elapsed() + 1.1 * block_seconds > HARD_LIMIT:
        SUMMARY["seeds_not_started"] = [s for s in SEEDS if s >= seed]
        break
    started = elapsed()
    for arm in ("B0", "C1", "S"):
        rec = one_run(seed, arm)
        SUMMARY["runs"][f"s{seed}_{arm}"] = rec
        if "failure" in rec:
            failures += 1
            save()
            if rec.get("fatal") or failures >= 2:
                stop(f"run s{seed}_{arm}: {rec['failure']}")
        else:
            failures = 0
        save()
    block_seconds = elapsed() - started

# ---- metadata-only baselines and untrained networks, then the descriptive comparison
SUMMARY["stage"] = "analysis"
features = pd.read_csv(IN / "header_features.csv", index_col=0)
cfg = {**json.loads(cfg_for("untrained", 0).read_text()), "amp": False}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
usable_set = set(json.loads((OUT / "warm" / "selected_studies.json").read_text()))  # preflighted and decoded by the cache step
all_ids = [u for u in order if u in usable_set]
vals_all = manifest.values[[index[u] for u in all_ids]]
masks_all = manifest.masks[[index[u] for u in all_ids]]
dataset = WeakLabelDicomDataset(metadata, all_ids, vals_all, masks_all, cfg, on_error="raise", cache_dir=CACHE)
untrained = {}
for useed in REG["untrained_seeds"]:
    seed_everything(useed)
    model = build_model("C", cfg).to(device)
    _, probs = predict_probabilities(model, dataset, device, batch_size=8, num_workers=0 if os.name == "nt" else cfg["num_workers"])
    untrained[useed] = probs[:, wc.ACTIVE_IDX]
pos = {u: i for i, u in enumerate(all_ids)}
cand_mask_full = np.zeros_like(masks_all)
for t, d in override["targets"].items():
    j = TARGET_COLUMNS.index(t)
    for u in d["studies"]:
        if u in pos:
            cand_mask_full[pos[u], j] = 1.0
protected = {"gold": gold_ids, "test": test_ids, "phase16 hold-out pool": pool}
per_seed = {}
for seed in SEEDS:
    p = partitions[seed]
    csvs = {arm: OUT / "runs" / f"s{seed}_{arm}" / "validation_predictions.csv" for arm in ("B0", "C1", "S")}
    if not all(c.is_file() for c in csvs.values()):
        continue
    frames = {arm: pd.read_csv(c).set_index("study") for arm, c in csvs.items()}
    ids = [u for u in frames["B0"].index if u in pos]
    rows = [pos[u] for u in ids]
    scores = {arm: frames[arm].loc[ids, list(wc.ACTIVE)].values for arm in frames}
    tr_ids = [u for u in p["train"] if u in features.index and u in pos]
    ev_ids = [u for u in ids if u in features.index]
    keep = [i for i, u in enumerate(ids) if u in features.index]
    for variant, mask in (("B0", masks_all), ("C1", masks_all * cand_mask_full)):
        tv, tm = vals_all[[pos[u] for u in tr_ids]], mask[[pos[u] for u in tr_ids]]
        meta = wc.fit_predict_baselines(features, tr_ids, tv, tm, {"val": ev_ids})
        for model_name, key in (("logreg", "M0l"), ("boosted", "M0b")):
            full = np.full((len(ids), 3), np.nan)
            full[keep] = meta[model_name]["val"]
            scores[f"{key}_{variant}"] = full
    for useed, arr in untrained.items():
        scores[f"N_{useed}"] = arr[rows]
    tables = {}
    for ref_name, mask in (("orig", masks_all), ("cand", masks_all * cand_mask_full)):
        reference = wc_.ReferenceLabels("weak", tuple(ids), vals_all[rows], mask[rows], source=f"manifest labels, {ref_name} mask, seed {seed} validation")
        main_scores = {k: v for k, v in scores.items() if not k.startswith("N_")}
        tables[ref_name] = wc_.score_table(reference, main_scores, rounds=ROUNDS, seed=0, protected=protected,
                                           differences=[("C1", "B0"), ("C1", "M0b_C1"), ("B0", "M0b_B0"), ("C1", "S")])
        everyone = np.arange(len(ids))
        tables[ref_name]["untrained_macro_auc"] = {k: wc_._macro(reference, v, everyone) for k, v in scores.items() if k.startswith("N_")}
    per_seed[seed] = tables
    pd.DataFrame({"study": ids, **{f"{a}_{t.split()[0]}": v[:, i] for a, v in scores.items() if not a.startswith("N_") for i, t in enumerate(wc.ACTIVE)}}).to_csv(
        OUT / f"predictions_seed{seed}.csv", index=False)
SUMMARY["per_seed"] = per_seed
if per_seed:
    for ref_name in ("orig", "cand"):
        SUMMARY[f"across_seeds_{ref_name}_medial"] = wc_.across_seeds([t[ref_name] for t in per_seed.values()],
                                                                      [("C1", "B0"), ("C1", "M0b_C1"), ("B0", "M0b_B0"), ("C1", "S")])
        SUMMARY[f"across_seeds_{ref_name}_acl"] = wc_.across_seeds([t[ref_name] for t in per_seed.values()], [("C1", "B0"), ("C1", "M0b_C1")], target="ACL")
        SUMMARY[f"across_seeds_{ref_name}_lateral"] = wc_.across_seeds([t[ref_name] for t in per_seed.values()], [("C1", "B0"), ("C1", "M0b_C1")], target="Lateral Meniscus")
allowed, reasons = wc_.claim_allowed(wc_.ReferenceLabels("weak", (), np.zeros((0, 12)), np.zeros((0, 12))), protected)
SUMMARY["claim_allowed"], SUMMARY["claim_blockers"] = allowed, reasons
SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = all("failure" not in r for r in SUMMARY["runs"].values()) and not SUMMARY.get("seeds_not_started")
save()
print(json.dumps({"claim_allowed": allowed, "blockers": reasons, "wall": SUMMARY["wall_seconds"], "all_passed": SUMMARY["all_passed"]}, indent=1))
sys.exit(0)
