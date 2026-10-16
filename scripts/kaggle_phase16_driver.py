"""Kaggle driver, Phase 16: pre-registered paired comparison of C0 (128x9, lr 1e-4) trained on real vs shuffled weak labels,
gradient clipping 1.0 vs 5.0, five paired seeds. Design, seeds, partition hashes and decision rules are fixed in
configs/phase16_registry.json and mri_core/weak_label_prereg.py BEFORE the first run.

  stage 0  verify bundle, regenerate the v0.1.0 manifest (hash must match), rebuild the Phase 14/15 selection of 1,200 studies
           (hash must match the registry), decode them, then decode the hold-out pool (manifest studies outside the selection)
  stage 1  seed blocks (one block = one seed x {clip 1.0, 5.0} x {real, shuffled}); each run: train 12 epochs, check the validation
           partition against the registry, score the final checkpoint on the hold-out pool, write result.json, drop control weights
  stage 2  registered verdict (prereg.evaluate), hash re-verification of kept checkpoints

No gold studies, no test images, no leaderboard feedback, no pretrained weights, no submission. Private kernel, internet off.
Resumable across kernel sessions: attach the previous output as a kernel source (P16_PREV) and completed runs are skipped while
an interrupted run continues from its last epoch-boundary checkpoint.
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
CACHE = pathlib.Path(os.environ.get("P16_CACHE", "/tmp/decode_cache"))
HARD_LIMIT = float(os.environ.get("P16_HARD", "6900"))  # kernel limit is 7200 s
DRY = bool(os.environ.get("P16_DRY"))
OUT = WORKING / "phase16"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"stage": "start", "stopped": None, "runs": {}, "results": {}}
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
    print(json.dumps(SUMMARY, indent=2, default=str)[:4000])
    sys.exit(1)


def run(cmd, name, timeout=3600):
    print(f"\n$ {' '.join(str(c) for c in cmd)[:240]}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-500:])
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


# ---- bundle, registry, data
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

import torch  # noqa: E402
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402
from mri_core.weak_label_data import load_weak_manifest  # noqa: E402
from mri_core.weak_label_prereg import evaluate, load_registry, run_name  # noqa: E402

REG = load_registry(code / "configs" / "phase16_registry.json")
SUMMARY["registry_sha256"] = sha(code / "configs" / "phase16_registry.json")
SEEDS = [int(x) for x in os.environ["P16_SEEDS"].split(",")] if os.environ.get("P16_SEEDS") else list(REG["seeds"])
if DRY:
    REG = {**REG, "seeds": SEEDS}
EPOCHS = int(os.environ.get("P16_EPOCHS", REG["epochs"]))
N_STUDIES = int(os.environ.get("P16_STUDIES", REG["selection"]["studies"]))
MIN_PER_CLASS = int(os.environ.get("P16_MIN_PER_CLASS", REG["selection"]["min_per_class"]))

meta_dir, dicom_root = discover_data(INPUT)
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1), "cpu_count": os.cpu_count()}
if not SUMMARY["environment"]["cuda_available"] and not DRY:
    stop("no CUDA device available")

ref = expected["manifest_reference"]
manifest = OUT / "manifest" / "weak_manifest_tier1.jsonl"
if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv", "--test-csv", meta_dir / "test.csv",
        "--sample-submission", meta_dir / "sample_submission.csv", "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus",
        "--output", manifest], "manifest").returncode != 0:
    stop("manifest regeneration failed")
msum = json.loads(manifest.with_name(manifest.name + ".summary.json").read_text())
SUMMARY["manifest"] = {"sha256": msum["manifest_sha256"], "included": msum["counts"]["included"],
                       "matches_local_reference": msum["manifest_sha256"] == ref["manifest_sha256"], "leakage_checks": msum["leakage_checks"]}
if not DRY and (msum["manifest_sha256"] != REG["manifest_sha256"] or not SUMMARY["manifest"]["matches_local_reference"]):
    stop("manifest differs from the registered manifest")
if any(msum["leakage_checks"][k] for k in ("overlap_with_gold", "overlap_with_test", "overlap_with_holdout")):
    stop("manifest failed a leakage check")

base = json.loads((code / "configs" / "weak_label_full_tier1.json").read_text())
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]
COMMON = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest]


def cfg_for(name, seed=0, clip=1.0, **kw):
    cfg = {**base, "arm": "C", "amp": True, "val_fraction": REG["model"]["val_fraction"], "seed": seed, "batch_size": REG["model"]["batch_size"],
           "epochs": EPOCHS, "early_stopping_patience": 0, "learning_rate": REG["learning_rate"], "grad_clip": clip,
           "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE), "image_size": 128, "slices_per_series": 9, **kw}
    path = OUT / "configs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


# ---- stage 0: decode the 1,200-study selection (seed 0 selection, as in Phases 14-15) and the hold-out pool
SUMMARY["stage"] = "cache"
if run(EXP + ["cache-warm-multi", *COMMON, "--config", cfg_for("warm"), "--smoke-subset", N_STUDIES, "--min-per-class", MIN_PER_CLASS,
              "--variants", "128x9", "--work-dir", OUT / "warm"], "cache_warm").returncode != 0:
    stop("cache build failed")
SELECTED, EXCLUDED = OUT / "warm" / "selected_studies.json", OUT / "warm" / "excluded_unreadable.json"
selected_ids = json.loads(SELECTED.read_text())
SUMMARY["selected_studies"] = {"count": len(selected_ids), "sha256": sha_ids(selected_ids)}
if not DRY and SUMMARY["selected_studies"]["sha256"] != REG["training_pool"]["sha256"]:
    stop("the 1,200-study selection differs from the registered one")
pool = {u for u in load_weak_manifest(manifest).ids} - set(selected_ids)
if run(EXP + ["cache-warm-multi", *COMMON, "--config", cfg_for("warm_holdout"), "--variants", "128x9",
              "--exclude-studies", SELECTED, "--work-dir", OUT / "warm_holdout"], "cache_warm_holdout").returncode != 0:
    stop("hold-out cache build failed")
HOLDOUT = OUT / "warm_holdout" / "selected_studies.json"
holdout_ids = set(json.loads(HOLDOUT.read_text()))
SUMMARY["holdout"] = {"count": len(holdout_ids), "pool_before_filters": len(pool), "subset_of_pool": holdout_ids <= pool,
                      "disjoint_from_selection": not (holdout_ids & set(selected_ids)),
                      "pool_sha256": sha_ids(pool), "registered_pool_sha256": REG["holdout_pool"]["sha256"]}
if not SUMMARY["holdout"]["subset_of_pool"] or not SUMMARY["holdout"]["disjoint_from_selection"] or not holdout_ids:
    stop("hold-out pool is not disjoint from the training selection")
if not DRY and SUMMARY["holdout"]["pool_sha256"] != REG["holdout_pool"]["sha256"]:
    stop("hold-out pool differs from the registered pool")
save()

# ---- resume support: completed runs (and an interrupted run's checkpoint) from a previous session's output
_prev_root = os.environ.get("P16_PREV")  # an earlier session's output attached as a kernel source: <root>/phase16/runs/<run>
for prev in glob.glob(str(pathlib.Path(_prev_root) / "phase16" / "runs" / "*")) if _prev_root else         glob.glob(str(INPUT / "*" / "phase16" / "runs" / "*")):
    prev = pathlib.Path(prev)
    target = OUT / "runs" / prev.name
    if target.exists() or not prev.is_dir():
        continue
    if (prev / "result.json").is_file():
        target.mkdir(parents=True)
        for f in prev.iterdir():
            if f.suffix != ".pt":
                shutil.copy2(f, target / f.name)
        for f in prev.glob("*.pt"):  # keep real-run weights so the final hash check covers them
            shutil.copy2(f, target / f.name)
    elif (prev / "last.pt").is_file():
        shutil.copytree(prev, target)
    SUMMARY.setdefault("restored_from_previous_session", []).append(prev.name)


def final_epoch_record(state):
    hist = state["metrics"]["history"]
    h = hist[-1]
    auc = [v for v in h["val_auc_weak"].values() if v is not None]
    best = min(hist, key=lambda x: x["val_loss"])
    best_auc = [v for v in best["val_auc_weak"].values() if v is not None]
    return {"epochs": state["epoch"], "val_final_macro_auc": sum(auc) / len(auc), "val_final_loss": h["val_loss"],
            "val_best_epoch": best["epoch"], "val_best_loss": best["val_loss"],
            "val_best_macro_auc": sum(best_auc) / len(best_auc), "history": [
                {k: x.get(k) for k in ("epoch", "train_loss", "val_loss", "val_macro_auc_weak", "grad_norm_mean", "grad_clipped_fraction",
                                       "nonfinite_grad_steps")} | {"val_prob_std": x.get("val_prob_std")} for x in hist]}


def one_run(seed, clip, label):
    name = run_name(seed, clip, label)
    work = OUT / "runs" / name
    if (work / "result.json").is_file():
        return json.loads((work / "result.json").read_text())
    extra = ["--shuffle-train-labels", REG["shuffle_seed_offset"] + seed] if label == "shuf" else []
    proc = run(EXP + ["train", *COMMON, "--config", cfg_for(name, seed, clip), "--include-studies", SELECTED, "--exclude-studies", EXCLUDED,
                      "--arm", "C", "--work-dir", work, *extra], f"train_{name}", timeout=int(max(300, HARD_LIMIT - elapsed())))
    if proc.returncode != 0:
        return {"failure": (proc.stderr or proc.stdout)[-800:]}
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    rec = final_epoch_record(state)
    valid_sha = sha_ids(state["valid_study_ids"])
    registered = REG["partitions"].get(str(seed), {}).get("validation_sha256")
    rec.update({"name": name, "seed": seed, "clip": clip, "label": label, "validation_sha256": valid_sha,
                "partition_matches_registry": valid_sha == registered if not DRY else None,
                "train_studies": len(state["train_study_ids"]), "validation_studies": len(state["valid_study_ids"]),
                "dataset_fingerprint": state["dataset_fingerprint"], "global_step": state["global_step"],
                "leakage": json.loads((work / "leakage.json").read_text()),
                "val_floor_loss": json.loads((work / "baselines.json").read_text())["val_loss_p05"],
                "last_pt_sha256": sha(work / "last.pt"), "last_best_sha256": sha(work / "last_best.pt")})
    if not DRY and not rec["partition_matches_registry"]:
        return {**rec, "failure": "validation partition differs from the registered partition", "fatal": True}
    if not rec["leakage"]["clean"]:
        return {**rec, "failure": "leakage report not clean", "fatal": True}
    if run(EXP + ["holdout-eval", *COMMON, "--config", cfg_for(name, seed, clip), "--checkpoint", work / "last.pt", "--include-studies", HOLDOUT,
                  "--exclude-studies", SELECTED, "--output", work / "holdout.json"], f"holdout_{name}").returncode != 0:
        return {**rec, "failure": "hold-out evaluation failed"}
    hold = json.loads((work / "holdout.json").read_text())
    if hold["macro_auc"] is None:  # cannot happen with hundreds of hold-out studies; tolerated only in tiny local dry runs
        if not DRY:
            return {**rec, "failure": "hold-out macro AUC undefined"}
        hold["macro_auc"] = 0.5
    rec.update({"holdout_macro_auc": hold["macro_auc"], "holdout_per_target_auc": hold["per_target_auc"], "holdout_studies": hold["studies"],
                "holdout_prob_std": hold["prob_std"], "holdout_class_counts": hold["class_counts"]})
    rec["kept_checkpoints"] = label == "real"
    (work / "result.json").write_text(json.dumps(rec, indent=2, default=str))
    if label != "real":  # control weights are not needed off Kaggle; their hashes are recorded above
        for n in ("last.pt", "last_best.pt"):
            (work / n).unlink(missing_ok=True)
    return rec


# ---- stage 1: seed blocks, paired real/shuffled twins, both clip arms
SUMMARY["stage"] = "runs"
block_seconds, failures, completed_blocks = None, 0, []
for seed in SEEDS:
    if block_seconds is not None and elapsed() + 1.1 * block_seconds > HARD_LIMIT:
        SUMMARY["blocks_not_started"] = [s for s in SEEDS if s not in completed_blocks and s != seed] + [seed]
        break
    started = elapsed()
    for clip in REG["grad_clip_arms"]:
        for label in ("real", "shuf"):
            rec = one_run(seed, clip, label)
            name = run_name(seed, clip, label)
            SUMMARY["runs"][name] = rec
            if "failure" in rec:
                failures += 1
                save()
                if rec.get("fatal") or failures >= 2:
                    stop(f"run {name}: {rec['failure']}")
                continue
            failures = 0
            SUMMARY["results"][name] = {k: rec[k] for k in ("val_final_macro_auc", "val_final_loss", "val_floor_loss", "holdout_macro_auc")}
            save()
            if not DRY and label == "shuf" and rec["val_final_macro_auc"] >= 0.60 and rec["holdout_macro_auc"] >= 0.60:
                stop(f"control alarm: shuffled run {name} scored like a real model (validation and hold-out AUC >= 0.60); investigate leakage")
    completed_blocks.append(seed)
    block_seconds = elapsed() - started
    SUMMARY.setdefault("block_seconds", {})[str(seed)] = round(block_seconds, 1)
    save()

# ---- stage 2: registered verdict and hash re-verification of every kept checkpoint
SUMMARY["stage"] = "verdict"
SUMMARY["verdict"] = evaluate(REG, SUMMARY["results"])
(OUT / "prereg_verdict.json").write_text(json.dumps(SUMMARY["verdict"], indent=2))
mismatch = []
for name, rec in SUMMARY["runs"].items():
    if rec.get("kept_checkpoints"):
        for fname, key in (("last.pt", "last_pt_sha256"), ("last_best.pt", "last_best_sha256")):
            if sha(OUT / "runs" / name / fname) != rec[key]:
                mismatch.append(f"{name}/{fname}")
SUMMARY["checkpoint_hash_mismatches"] = mismatch
SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = not mismatch and not SUMMARY.get("blocks_not_started") and all("failure" not in r for r in SUMMARY["runs"].values())
save()
print(json.dumps({"verdict": {k: v["verdict"] for k, v in SUMMARY["verdict"]["arms"].items()}, "clip_comparison": SUMMARY["verdict"]["clip_comparison"],
                  "control_alarm": SUMMARY["verdict"]["control_alarm"], "mismatch": mismatch, "wall": SUMMARY["wall_seconds"]}, indent=2, default=str))
sys.exit(0 if SUMMARY["all_passed"] else 1)
