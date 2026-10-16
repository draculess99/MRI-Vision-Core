"""Kaggle driver, Phase 14: resolution x slice-count ablation of C0 (ResNet-18 from scratch, 2.5D, Attention-MIL), one seed.

Variants: 128x9 (production default), 192x9, 128x15, 192x15 (image size x slices per plane). Everything else is held constant: labels
(rules 0.1.0 manifest), C0, optimizer, batch size, seed, study selection and hash split (so the weak-validation loss and AUCs of the four
runs are directly comparable). Stages:
  0  verify bundle + manifest, select 1,200 studies, decode each ONCE into four cache variants (time and size recorded)
  A  feasibility on a 64-study subset per variant: forward/backward, masked loss, AMP, export, stop-after-epoch-1 + resume
  B  one-seed comparison on the full selected set, 6 epochs per variant
Private kernel, internet off. No gold, no test images, no pretrained weights, no submission.
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

N_STUDIES = int(os.environ.get("P14_STUDIES", "1200"))
MIN_PER_CLASS = int(os.environ.get("P14_MIN_PER_CLASS", "60"))
EPOCHS = int(os.environ.get("P14_EPOCHS", "6"))
BUDGET = float(os.environ.get("P14_BUDGET", str(2.5 * 3600)))
FEAS_STUDIES = int(os.environ.get("P14_FEAS", "64"))
VARIANTS = [(128, 9), (192, 9), (128, 15), (192, 15)]
INPUT = pathlib.Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORKING = pathlib.Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
OUT = WORKING / "phase14"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = WORKING / "decode_cache"
SUMMARY = {"stage": "start", "stopped": None}
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
    print(json.dumps(SUMMARY, indent=2, default=str)[:6000])
    sys.exit(1)


def run(cmd, name, timeout=7000):
    print(f"\n$ {' '.join(str(c) for c in cmd)[:240]}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-900:])
    if proc.returncode:
        print("[stderr]", proc.stderr[-1800:])
    SUMMARY.setdefault("steps", {})[name] = {"exit_code": proc.returncode, "seconds": round(time.perf_counter() - started, 1)}
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs" / f"{name}.log").write_text((proc.stdout or "") + "\n[stderr]\n" + (proc.stderr or ""))
    return proc


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---- bundle, data, manifest
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

meta_dir, dicom_root = discover_data(INPUT)
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1), "cpu_count": os.cpu_count()}
if not SUMMARY["environment"]["cuda_available"] and not os.environ.get("SMOKE_DEVICE_NOTE"):
    stop("no CUDA device available")
need_gb = N_STUDIES * 7.6e-3 + 3
if SUMMARY["environment"]["free_disk_gb"] < need_gb:
    stop(f"need about {need_gb:.0f} GB free for the cache variants, have {SUMMARY['environment']['free_disk_gb']}")

ref = expected["manifest_reference"]
manifest = OUT / "manifest" / "weak_manifest_tier1.jsonl"
if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv", "--test-csv", meta_dir / "test.csv",
        "--sample-submission", meta_dir / "sample_submission.csv", "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus",
        "--output", manifest], "manifest").returncode != 0:
    stop("manifest regeneration failed")
msum = json.loads(manifest.with_name(manifest.name + ".summary.json").read_text())
SUMMARY["manifest"] = {"profile": "0.1.0", "sha256": msum["manifest_sha256"], "included": msum["counts"]["included"],
                       "matches_local_reference": msum["manifest_sha256"] == ref["manifest_sha256"], "leakage_checks": msum["leakage_checks"]}
if not SUMMARY["manifest"]["matches_local_reference"] or any(msum["leakage_checks"][k] for k in ("overlap_with_gold", "overlap_with_test", "overlap_with_holdout")):
    stop("manifest differs from the local reference or failed a leakage check")

base = json.loads((code / "configs" / "weak_label_full_tier1.json").read_text())
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]


def cfg_for(name, size, slices, **kw):
    cfg = {**base, "arm": "C", "amp": True, "val_fraction": 0.2, "seed": 0, "batch_size": 8, "epochs": EPOCHS, "early_stopping_patience": 0,
           "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE), "image_size": size, "slices_per_series": slices, **kw}
    path = OUT / "configs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


COMMON = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest]

# ---- stage 0: one decode per study, four cache variants
SUMMARY["stage"] = "cache"
warm_cfg = cfg_for("warm", 128, 9)
proc = run(EXP + ["cache-warm-multi", *COMMON, "--config", warm_cfg, "--smoke-subset", N_STUDIES, "--min-per-class", MIN_PER_CLASS,
                  "--variants", ",".join(f"{s}x{n}" for s, n in VARIANTS), "--work-dir", OUT / "warm"], "cache_warm_multi")
if proc.returncode != 0:
    stop("multi-variant cache build failed")
warm = json.loads((OUT / "warm" / "cache_warm_multi.json").read_text())
SUMMARY["cache"] = warm
if warm["unreadable"] > 0.02 * warm["studies"]:
    stop(f"{warm['unreadable']} unreadable studies")
SELECTED, EXCLUDED = OUT / "warm" / "selected_studies.json", OUT / "warm" / "excluded_unreadable.json"
SUMMARY["selected_studies"] = len(json.loads(SELECTED.read_text()))
SUMMARY["disk_free_gb_after_cache"] = round(shutil.disk_usage(WORKING).free / 1e9, 1)
save()


def train(name, size, slices, epochs, work, *extra):
    cfg = cfg_for(name, size, slices, epochs=epochs)
    return run(EXP + ["train", *COMMON, "--config", cfg, "--include-studies", SELECTED, "--exclude-studies", EXCLUDED, "--arm", "C",
                      "--work-dir", work, *extra], f"train_{name}", timeout=int(max(600, BUDGET - elapsed())))


def report(work):
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    log = [json.loads(l) for l in (work / "runtime_log.jsonl").read_text().splitlines()]
    epochs = [r for r in log if r["event"] == "epoch"]
    hist = state["metrics"]["history"]
    best = min(hist, key=lambda h: h["val_loss"])
    split = json.loads((work / "split_summary.json").read_text())
    aucs = {t: a for t, a in best["val_auc_weak"].items() if a is not None}
    return {
        "epoch": state["epoch"], "global_step": state["global_step"], "dataset_fingerprint": state["dataset_fingerprint"],
        "train_studies": split["train_studies"], "validation_studies": split["validation_studies"],
        "class_counts": {"train": split["train_targets"], "validation": split["validation_targets"]},
        "history": [{"epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": h["val_loss"],
                     "val_auc_weak": {t: a for t, a in h["val_auc_weak"].items() if a is not None}, "invalid": h["invalid_samples"],
                     "train_images": h["train_images"]} for h in hist],
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "best_weak_auc": aucs,
        "best_weak_macro_auc": sum(aucs.values()) / len(aucs) if len(aucs) == 3 else None,
        "epoch_seconds": [round(e["seconds"], 1) for e in epochs], "data_wait_seconds": [e["data_wait_seconds"] for e in epochs],
        "images_per_second": [round(e["train_images"] / max(e["seconds"], 1e-9), 1) for e in epochs],
        "peak_cuda_gb": max((e["cuda_peak_gb"] or 0) for e in epochs), "sessions_in_log": sum(r["event"] == "session_start" for r in log),
        "scaler_scale": (state["scaler"] or {}).get("scale"), "invalid_total": sum(h["invalid_samples"] for h in hist),
        "finite": all(h["train_loss"] == h["train_loss"] and h["val_loss"] == h["val_loss"] and abs(h["val_loss"]) != float("inf") for h in hist),
        "last_pt_sha256": sha(work / "last.pt"), "last_best_sha256": sha(work / "last_best.pt"),
        "last_pt_mb": round((work / "last.pt").stat().st_size / 1e6, 1), "last_best_mb": round((work / "last_best.pt").stat().st_size / 1e6, 1)}


def export_ok(work, tag):
    dest = OUT / "exports" / tag
    if run(EXP + ["export", "--work-dir", work, "--export-dir", dest], f"export_{tag}").returncode != 0:
        return False
    m = json.loads((dest / "MANIFEST.json").read_text())
    return all(sha(dest / n) == i["sha256"] for n, i in m.items() if n != "runtime")


# ---- stage A: feasibility on a small deterministic subset
SUMMARY["stage"] = "feasibility"
SUMMARY["feasibility"] = {}
for size, slices in VARIANTS:
    tag = f"{size}x{slices}"
    work = OUT / "feasibility" / tag
    first = train(f"feas_{tag}_e1", size, slices, 2, work, "--smoke-subset", FEAS_STUDIES, "--min-per-class", 8, "--stop-after-epochs", 1)
    if first.returncode != 0:
        SUMMARY["feasibility"][tag] = {"failure": (first.stderr or first.stdout)[-700:]}
        stop(f"feasibility {tag}: epoch 1 failed")
    mid = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    second = train(f"feas_{tag}_resume", size, slices, 2, work, "--smoke-subset", FEAS_STUDIES, "--min-per-class", 8)
    if second.returncode != 0:
        SUMMARY["feasibility"][tag] = {"failure": (second.stderr or second.stdout)[-700:]}
        stop(f"feasibility {tag}: resume failed")
    r = report(work)
    r["restore"] = {"epoch_before": mid["epoch"], "step_before": mid["global_step"], "resumed_to": r["epoch"], "step_continued": r["global_step"] > mid["global_step"]}
    r["export_ok"] = export_ok(work, f"feas_{tag}")
    r["passed"] = bool(r["finite"] and r["invalid_total"] == 0 and r["export_ok"] and r["restore"]["step_continued"] and r["epoch"] == 2)
    SUMMARY["feasibility"][tag] = r
    save()
    if not r["passed"]:
        stop(f"feasibility {tag} failed a check")
    if elapsed() > BUDGET:
        stop("time budget reached during feasibility")

# ---- stage B: one-seed comparison
SUMMARY["stage"] = "comparison"
SUMMARY["comparison"] = {}
for size, slices in VARIANTS:
    tag = f"{size}x{slices}"
    if elapsed() > BUDGET:
        SUMMARY["time_budget_reached"] = True
        break
    work = OUT / "comparison" / tag
    proc = train(f"cmp_{tag}", size, slices, EPOCHS, work)
    if proc.returncode != 0:
        SUMMARY["comparison"][tag] = {"failure": (proc.stderr or proc.stdout)[-700:]}
        stop(f"comparison {tag} failed")
    r = report(work)
    r["export_ok"] = export_ok(work, f"cmp_{tag}")
    r["run_seconds"] = SUMMARY["steps"][f"train_cmp_{tag}"]["seconds"]
    r["passed"] = bool(r["finite"] and r["invalid_total"] == 0 and r["export_ok"] and r["epoch"] == EPOCHS)
    SUMMARY["comparison"][tag] = r
    save()
    if not r["passed"]:
        stop(f"comparison {tag} failed a check")

base_r = SUMMARY["comparison"].get("128x9")
SUMMARY["differences_vs_128x9"] = {
    tag: {"best_val_loss": r["best_val_loss"] - base_r["best_val_loss"],
          "best_weak_macro_auc": None if None in (r["best_weak_macro_auc"], base_r["best_weak_macro_auc"]) else r["best_weak_macro_auc"] - base_r["best_weak_macro_auc"],
          "per_target_auc": {t: r["best_weak_auc"][t] - base_r["best_weak_auc"][t] for t in r["best_weak_auc"] if t in base_r["best_weak_auc"]}}
    for tag, r in SUMMARY["comparison"].items() if base_r and tag != "128x9"}
SUMMARY["same_fingerprint_all_runs"] = len({r["dataset_fingerprint"] for r in SUMMARY["comparison"].values()}) == 1
SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = len(SUMMARY["comparison"]) == len(VARIANTS) and all(r["passed"] for r in SUMMARY["comparison"].values())
save()
print(json.dumps({k: SUMMARY[k] for k in ("differences_vs_128x9", "same_fingerprint_all_runs", "all_passed", "wall_seconds")}, indent=2, default=str))
sys.exit(0 if SUMMARY["all_passed"] else 1)
