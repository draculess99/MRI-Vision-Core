"""Kaggle driver, Phase 16X: POST-HOC EXPLORATORY acquisition-confounding diagnostic (not a confirmatory test, not part of the Phase 16
pre-registration, which is unchanged and stays stopped).

Question: does the trained C0 rank studies by weak label better than (a) DICOM-header acquisition metadata and (b) untrained networks?
  1  verify bundle, manifest, selection/hold-out lists and validation partitions against configs/phase16_registry.json
  2  header-only features (no pixel data, no patient fields) for every manifest study
  3  metadata baselines (logistic regression, small boosted tree; fixed, untuned) fitted on each REGISTERED training partition only
  4  untrained C0 forward passes for ten fixed seeds (+ the initialisation of the Phase 16 seed-101 run); no optimisation, no labels
  5  the recovered Phase 16 seed-101 real-label checkpoint (SHA-256 verified) on the same studies
  6  AUCs with study-level bootstrap CIs, paired differences, and within-stratum AUCs (series-count strata; baseline-score strata)
The validation and hold-out studies are only SCORED here; no model, feature, hyper-parameter or stratum is chosen with them.
No gold studies, no test images, no leaderboard, no training. Private kernel, internet off.
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
CACHE = pathlib.Path(os.environ.get("P16X_CACHE", "/tmp/decode_cache"))
DRY = bool(os.environ.get("P16X_DRY"))  # local rehearsal: registry equality checks relaxed, CPU allowed
HARD_LIMIT = float(os.environ.get("P16X_HARD", "2100"))
ROUNDS = int(os.environ.get("P16X_ROUNDS", "1000"))
UNTRAINED_SEEDS = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010]  # fixed in advance
OUT = WORKING / "phase16x"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"exploratory": True, "post_hoc": True, "confirmatory": False,
           "note": "Post-hoc diagnostic prompted by the Phase 16 control alarm; the pre-registered Phase 16 decision rules are not applied or changed here.",
           "stage": "start", "stopped": None}
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
    sys.exit(0)  # a planned stop is not a kernel error; the summary says why


def run(cmd, name, timeout=3000):
    print(f"\n$ {' '.join(str(c) for c in cmd)[:220]}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-400:])
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


# ---- 1. bundle, registry, lists
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
from mri_core import weak_label_confound as wc  # noqa: E402
from mri_core.rsna_knee_dataset import load_rsna_metadata  # noqa: E402
from mri_core.weak_label_data import WeakLabelDicomDataset, assert_no_leakage, hash_split, load_weak_manifest  # noqa: E402
from mri_core.weak_label_models import build_model  # noqa: E402
from mri_core.weak_label_prereg import load_registry  # noqa: E402
from mri_core.weak_label_train import predict_probabilities  # noqa: E402
from mri_core.rsna_knee_train import seed_everything  # noqa: E402

REG = load_registry(code / "configs" / "phase16_registry.json")
SUMMARY["registry_sha256"] = sha(code / "configs" / "phase16_registry.json")
meta_dir, dicom_root = discover_data(INPUT)
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1), "cpu_count": os.cpu_count()}
metadata = load_rsna_metadata(meta_dir, dicom_root)

manifest_path = OUT / "manifest" / "weak_manifest_tier1.jsonl"
if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv", "--test-csv", meta_dir / "test.csv",
        "--sample-submission", meta_dir / "sample_submission.csv", "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus",
        "--output", manifest_path], "manifest").returncode != 0:
    stop("manifest regeneration failed")
manifest = load_weak_manifest(manifest_path)
if not DRY and manifest.sha256 != REG["manifest_sha256"]:
    stop("manifest differs from the registered manifest")
SUMMARY["manifest_sha256"] = manifest.sha256

# private dataset with the recovered Phase 16 seed-101 checkpoint and the registered selection / hold-out lists (copied unchanged from the
# Phase 16 kernel output, which Kaggle will not attach as a kernel source)
prov_files = glob.glob(str(pathlib.Path(os.environ["P16X_INPUTS"]) / "PROVENANCE.json")) if os.environ.get("P16X_INPUTS") else     glob.glob(str(INPUT / "*" / "PROVENANCE.json"))
prov_files = [f for f in prov_files if json.loads(pathlib.Path(f).read_text()).get("kind") == "phase16x_inputs"]
if len(prov_files) != 1:
    stop("the private dataset draculess/rsna-weak-label-phase16x-inputs (PROVENANCE.json, kind phase16x_inputs) is not attached exactly once")
IN = pathlib.Path(prov_files[0]).parent
PROV = json.loads((IN / "PROVENANCE.json").read_text())
for fname, info in PROV["files"].items():
    if sha(IN / fname) != info["sha256"]:
        stop(f"{fname} does not match the hash in PROVENANCE.json")
if PROV["files"]["last.pt"]["sha256"] != PROV["files"]["last.pt"]["original_recorded_sha256_in_result_json"]:
    stop("the checkpoint hash differs from the hash recorded when it was trained")
selected_ids = json.loads((IN / "selected_studies.json").read_text())
holdout_ids = json.loads((IN / "holdout_studies.json").read_text())
SUMMARY["inputs_dataset"] = {"provenance_sha256": sha(IN / "PROVENANCE.json"), "files": {k: v["sha256"] for k, v in PROV["files"].items()}}
pool = set(manifest.ids) - set(selected_ids)
SUMMARY["lists"] = {"selected": len(selected_ids), "selected_sha256": sha_ids(selected_ids), "holdout": len(holdout_ids),
                    "holdout_subset_of_registered_pool": set(holdout_ids) <= pool, "pool_sha256": sha_ids(pool)}
if not DRY and (SUMMARY["lists"]["selected_sha256"] != REG["training_pool"]["sha256"] or SUMMARY["lists"]["pool_sha256"] != REG["holdout_pool"]["sha256"]):
    stop("selection or hold-out pool differs from the registry")
index = {u: i for i, u in enumerate(manifest.ids)}
order = [u for u in manifest.ids if u in set(selected_ids)]
partitions = {}
for seed in REG["seeds"]:
    try:
        tr, va = hash_split(order, 0.2, seed)
    except ValueError:
        if DRY:  # tiny rehearsal sets can give an empty side
            continue
        raise
    train_ids, val_ids = [order[i] for i in tr], [order[i] for i in va]
    ok = sha_ids(val_ids) == REG["partitions"][str(seed)]["validation_sha256"]
    if not ok and not DRY:
        stop(f"seed {seed}: validation partition differs from the registry")
    partitions[seed] = {"train": train_ids, "val": val_ids}
test_ids = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
gold_ids = set(metadata.complete_label_train["StudyInstanceUID"].astype(str))
for seed, p in partitions.items():
    assert_no_leakage(p["train"], p["val"], gold_ids=gold_ids, test_ids=test_ids)
    assert not set(holdout_ids) & (set(p["train"]) | set(p["val"])), "hold-out overlaps a registered partition"
assert_no_leakage(holdout_ids, [], gold_ids=gold_ids, test_ids=test_ids)
SUMMARY["leakage"] = {"partitions_checked": len(partitions), "gold_overlap": 0, "test_overlap": 0, "holdout_disjoint_from_all_partitions": True}
save()


def labels_for(ids):
    rows = [index[u] for u in ids]
    return manifest.values[rows], manifest.masks[rows]


# ---- 2. header-only features for every manifest study
SUMMARY["stage"] = "features"
t = time.perf_counter()
features, failures = wc.build_feature_table(metadata, list(manifest.ids), workers=8)
features.to_csv(OUT / "header_features.csv")
SUMMARY["features"] = {"studies": int(len(features)), "failed": len(failures), "failure_kinds": sorted(set(failures.values())),
                       "columns": [c for c in features.columns], "seconds": round(time.perf_counter() - t, 1)}
(OUT / "feature_failures.json").write_text(json.dumps(failures))
if len(failures) > 0.02 * len(manifest.ids) and not DRY:
    stop(f"{len(failures)} studies had unreadable headers")
save()
usable = lambda ids: [u for u in ids if u in features.index]

# ---- 3. metadata baselines on every registered training partition
SUMMARY["stage"] = "baselines"
SUMMARY["baselines"] = {}
baseline_scores = {}
for seed, p in partitions.items():
    tr_ids = usable(p["train"])
    tv, tm = labels_for(tr_ids)
    evals = {"val": usable(p["val"]), "holdout": usable(holdout_ids)}
    preds = wc.fit_predict_baselines(features, tr_ids, tv, tm, evals)
    res = {}
    for model in ("logreg", "boosted"):
        for name, ids in evals.items():
            lv, lm = labels_for(ids)
            res[f"{model}_{name}"] = wc.macro_target_auc(lv, lm, preds[model][name])
    SUMMARY["baselines"][str(seed)] = {**res, "train_studies": len(tr_ids)}
    baseline_scores[seed] = (preds, evals)
save()

# ---- 4/5. untrained and trained forward passes on seed-101 validation + hold-out
SUMMARY["stage"] = "forward"
SEED = 101 if 101 in partitions else list(partitions)[0]  # (the fallback only matters in tiny local rehearsals)
val_ids, ho_ids = usable(partitions[SEED]["val"]), usable(holdout_ids)
union = val_ids + ho_ids
(OUT / "union_studies.json").write_text(json.dumps(union))
cfg = {**json.loads((code / "configs" / "weak_label_full_tier1.json").read_text()), "arm": "C", "image_size": 128, "slices_per_series": 9,
       "amp": False, "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE), "dropout": 0.2}
(OUT / "cfg.json").write_text(json.dumps(cfg))
if run([sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py", "cache-warm-multi", "--data-dir", meta_dir, "--dicom-root", dicom_root,
        "--manifest", manifest_path, "--config", OUT / "cfg.json", "--variants", "128x9", "--include-studies", OUT / "union_studies.json",
        "--work-dir", OUT / "warm"], "cache_warm", timeout=int(max(600, HARD_LIMIT - elapsed()))).returncode != 0:
    stop("cache build failed")
unreadable = set(json.loads((OUT / "warm" / "excluded_unreadable.json").read_text()))
val_ids, ho_ids = [u for u in val_ids if u not in unreadable], [u for u in ho_ids if u not in unreadable]
union = val_ids + ho_ids
lv, lm = labels_for(union)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = WeakLabelDicomDataset(metadata, union, lv, lm, cfg, on_error="raise", cache_dir=CACHE)


def forward(model):
    model.to(device)
    _, probs = predict_probabilities(model, dataset, device, batch_size=8, num_workers=0 if os.name == "nt" else cfg["num_workers"])  # Windows spawn would re-run this script in every worker
    return probs[:, wc.ACTIVE_IDX]


scores = {}
for seed in UNTRAINED_SEEDS:
    seed_everything(seed)
    scores[f"untrained_{seed}"] = forward(build_model("C", cfg))
seed_everything(SEED)
scores[f"init_of_phase16_seed{SEED}"] = forward(build_model("C", cfg))
SUMMARY["untrained_forward_seconds"] = round(elapsed(), 1)

ckpts = [str(IN / "last.pt")] if PROV["checkpoint"]["seed"] == SEED else []  # the provenance names the seed of this checkpoint
if ckpts:
    recorded = PROV["files"]["last.pt"]["original_recorded_sha256_in_result_json"]
    digest = sha(ckpts[0])
    SUMMARY["trained_checkpoint"] = {"sha256": digest, "recorded_sha256": recorded, "verified": recorded == digest}
    if recorded is not None and recorded != digest and not DRY:
        stop("checkpoint hash differs from the recorded hash")
    state = torch.load(ckpts[0], map_location="cpu", weights_only=True)
    if (set(state["train_study_ids"]) | set(state["valid_study_ids"])) & set(ho_ids) and not DRY:
        stop("hold-out studies were in the checkpoint's training/validation sets")
    model = build_model("C", state["config"])
    model.load_state_dict(state["model"])
    scores["trained_real"] = forward(model)
    SUMMARY["trained_checkpoint"]["sha256_after_use"] = sha(ckpts[0])
else:
    SUMMARY["trained_checkpoint"] = {"available": False, "note": "the checkpoint in the inputs dataset belongs to a different seed than the analysed partition"}
for name, arr in scores.items():
    assert np.isfinite(arr).all(), name
save()

# ---- 6. analysis: AUC + bootstrap, paired differences, strata
SUMMARY["stage"] = "analysis"
n_val = len(val_ids)
parts = {"val": slice(0, n_val), "holdout": slice(n_val, len(union))}
preds_b, evals_b = baseline_scores[SEED]
tr_ids = usable(partitions[SEED]["train"])
tv, tm = labels_for(tr_ids)
# strata cut-points come from TRAINING studies only
series_cuts = np.quantile(features.loc[tr_ids, "n_series"].values, [1 / 3, 2 / 3])
train_pred = wc.fit_predict_baselines(features, tr_ids, tv, tm, {"train": tr_ids})["logreg"]["train"]
score_cuts = [np.quantile(train_pred[:, k][~np.isnan(train_pred[:, k])], [0.2, 0.4, 0.6, 0.8]) if (~np.isnan(train_pred[:, k])).any() else [] for k in range(3)]
analysis = {}
for pname, sl in parts.items():
    ids = union[sl]
    labels, masks = lv[sl], lm[sl]
    base_ids = evals_b["val" if pname == "val" else "holdout"]
    align = [base_ids.index(u) for u in ids if u in base_ids]
    keep = [i for i, u in enumerate(ids) if u in base_ids]
    sc = {name: arr[sl][keep] for name, arr in scores.items()}
    sc["metadata_logreg"] = preds_b["logreg"]["val" if pname == "val" else "holdout"][align]
    sc["metadata_boosted"] = preds_b["boosted"]["val" if pname == "val" else "holdout"][align]
    labels, masks = labels[keep], masks[keep]
    kept_ids = [ids[i] for i in keep]
    s1 = wc.strata_from_cuts(features.loc[kept_ids, "n_series"].values, series_cuts)
    strata1 = np.tile(s1[:, None], (1, 3))
    strata2 = np.stack([wc.strata_from_cuts(sc["metadata_logreg"][:, k], score_cuts[k]) if len(score_cuts[k]) else np.zeros(len(kept_ids), int) for k in range(3)], axis=1)
    focus = [k for k in ("trained_real", "metadata_logreg", "metadata_boosted") if k in sc]
    metrics = {}
    for k in sc:
        metrics[f"auc:{k}"] = (lambda arr: lambda idx: wc.macro_target_auc(labels, masks, arr, idx))(sc[k])
    for k in focus:
        metrics[f"within_series_count:{k}"] = (lambda arr: lambda idx: wc.macro_stratified_auc(labels, masks, arr, strata1, idx))(sc[k])
        metrics[f"within_baseline_score:{k}"] = (lambda arr: lambda idx: wc.macro_stratified_auc(labels, masks, arr, strata2, idx))(sc[k])
    diffs = []
    if "trained_real" in sc:
        diffs += [("auc:trained_real", "auc:metadata_logreg"), ("auc:trained_real", "auc:metadata_boosted"),
                  ("within_series_count:trained_real", "within_series_count:metadata_logreg"),
                  ("within_baseline_score:trained_real", "auc:trained_real")]
    u_names = [k for k in sc if k.startswith("untrained_")]
    boot = wc.bootstrap({k: v for k, v in metrics.items() if not k.startswith("auc:untrained_") and not k.startswith("auc:init_")},
                        len(labels), rounds=ROUNDS, seed=0, differences=[d for d in diffs if d[0] in metrics and d[1] in metrics])
    untrained = {k: wc.macro_target_auc(labels, masks, sc[k]) for k in sc if k.startswith(("untrained_", "init_of"))}
    vals = [v for k, v in untrained.items() if k.startswith("untrained_") and v is not None]
    # coverage: share of positive-negative pairs that lie inside a stratum
    cover = {}
    for sname, st in (("series_count", strata1), ("baseline_score", strata2)):
        c = []
        for k, j in enumerate(wc.ACTIVE_IDX):
            on = masks[:, j] == 1
            y = np.nan_to_num(labels[on, j]).astype(int)
            sk, stk = sc["metadata_logreg"][on, k], st[on, k]
            good = ~np.isnan(sk)  # undefined baseline scores (single-class training target) are excluded, never filled in
            c.append(wc.stratified_auc(y[good], sk[good], stk[good])[1] if good.any() else None)
        cover[sname] = [None if x is None else round(float(x), 3) for x in c]
    analysis[pname] = {"studies": len(labels), "class_counts": {t: {"positive": int(((labels[:, j] == 1) & (masks[:, j] == 1)).sum()),
                       "negative": int(((labels[:, j] == 0) & (masks[:, j] == 1)).sum())} for t, j in zip(wc.ACTIVE, wc.ACTIVE_IDX)},
                       "bootstrap": boot, "untrained_macro_auc": untrained,
                       "untrained_summary": {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
                                             "n_ge_0.60": int(sum(v >= 0.60 for v in vals)), "n": len(vals)} if vals else None,
                       "per_target_auc": {k: [wc.target_auc(labels, masks, sc[k], i) for i in range(3)] for k in focus},
                       "spearman_with_metadata_logreg": {k: wc.spearman_by_target(sc[k], sc["metadata_logreg"]) for k in sc if k != "metadata_logreg"},
                       "stratum_pair_coverage": cover, "series_count_cuts": [float(x) for x in series_cuts]}
    pd.DataFrame({"study": kept_ids, **{f"{k}_{t.split()[0]}": v[:, i] for k, v in sc.items() if k in focus for i, t in enumerate(wc.ACTIVE)}}).to_csv(
        OUT / f"predictions_{pname}.csv", index=False)
SUMMARY["analysis"] = analysis
SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = SUMMARY["trained_checkpoint"].get("verified", False) or DRY
save()
print(json.dumps({p: {"bootstrap": a["bootstrap"]["point"], "untrained": a["untrained_summary"]} for p, a in analysis.items()}, indent=2, default=str)[:3000])
sys.exit(0)
