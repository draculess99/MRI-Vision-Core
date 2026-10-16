"""Kaggle driver, Phase 15: training-health diagnostics for Model C0 (ResNet-18 from scratch, 2.5D, Attention-MIL), 128x9 only.

Question: can C0 learn from the weak labels at all? Experiments, all on the Phase 14 selection (1,200 training-only studies, labels
v0.1.0, hash split) so numbers are directly comparable with Phase 14:
  A  overfit: tiny subsets, validation == the training studies (fit check, never a performance claim), incl. a resume-equivalence check
  B  constant-prediction baselines (computed by the train command for every run from the same split and loss)
  C  shuffled-label negative control (labels permuted across TRAINING studies; validation labels stay true)
  D  per-epoch gradient-norm / clipping / prediction-spread diagnostics, plus learning-rate sensitivity and a second seed
No gold, no test images, no pretrained weights, no submission. Private kernel, internet off, one T4.
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

N_STUDIES = int(os.environ.get("P15_STUDIES", "1200"))
MIN_PER_CLASS = int(os.environ.get("P15_MIN_PER_CLASS", "60"))
BUDGET = float(os.environ.get("P15_BUDGET", str(75 * 60)))  # soft limit: optional runs are skipped once exceeded
HARD_LIMIT = float(os.environ.get("P15_HARD", str(85 * 60)))
E_LONG = int(os.environ.get("P15_EPOCHS", "12"))
E_SHORT = int(os.environ.get("P15_EPOCHS_SHORT", "6"))
E_OVERFIT = int(os.environ.get("P15_OVERFIT_EPOCHS", "60"))
INPUT = pathlib.Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORKING = pathlib.Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
CACHE = pathlib.Path(os.environ.get("P15_CACHE", "/tmp/decode_cache"))  # outside the downloadable output
OUT = WORKING / "phase15"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"stage": "start", "stopped": None, "runs": {}}
T0 = time.perf_counter()
PHASE14_FINGERPRINT = "955416d9b42948d3a9c110863265d7feb832ad26eb8c4d640c3a8c80e3789c39"
PHASE14_REFERENCE = {"best_val_loss": 1.048, "best_weak_macro_auc": 0.603}  # 128x9, seed 0, 6 epochs (Phase 14 final_summary.json)


def elapsed():
    return time.perf_counter() - T0


def save():
    SUMMARY["wall_seconds"] = round(elapsed(), 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    print(json.dumps(SUMMARY, indent=2, default=str)[:5000])
    sys.exit(1)


def run(cmd, name, timeout=3600):
    print(f"\n$ {' '.join(str(c) for c in cmd)[:240]}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-700:])
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
COMMON = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest]


def cfg_for(name, **kw):
    cfg = {**base, "arm": "C", "amp": True, "val_fraction": 0.2, "seed": 0, "batch_size": 8, "epochs": E_LONG, "early_stopping_patience": 0,
           "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE), "image_size": 128, "slices_per_series": 9, **kw}
    path = OUT / "configs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


# ---- stage 0: decode 128x9 once for the Phase 14 selection
SUMMARY["stage"] = "cache"
proc = run(EXP + ["cache-warm-multi", *COMMON, "--config", cfg_for("warm"), "--smoke-subset", N_STUDIES, "--min-per-class", MIN_PER_CLASS,
                  "--variants", "128x9", "--work-dir", OUT / "warm"], "cache_warm")
if proc.returncode != 0:
    stop("cache build failed")
warm = json.loads((OUT / "warm" / "cache_warm_multi.json").read_text())
SUMMARY["cache"] = warm
if warm["unreadable"] > 0.02 * warm["studies"]:
    stop(f"{warm['unreadable']} unreadable studies")
SELECTED, EXCLUDED = OUT / "warm" / "selected_studies.json", OUT / "warm" / "excluded_unreadable.json"
SUMMARY["selected_studies"] = len(json.loads(SELECTED.read_text()))
save()


def train(name, epochs, *extra, lr=1e-3, seed=0, subset=None, keep_checkpoints=False, stop_after=None):
    work = OUT / "runs" / name
    cfg = cfg_for(name, epochs=epochs, learning_rate=lr, seed=seed)
    args = EXP + ["train", *COMMON, "--config", cfg, "--include-studies", SELECTED, "--exclude-studies", EXCLUDED, "--arm", "C",
                  "--work-dir", work, *extra]
    if subset:
        args += ["--smoke-subset", subset[0], "--min-per-class", subset[1]]
    if stop_after:
        args += ["--stop-after-epochs", stop_after]
    return work, run(args, f"train_{name}", timeout=int(max(300, HARD_LIMIT - elapsed())))


def report(name, work, keep):
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    log = [json.loads(l) for l in (work / "runtime_log.jsonl").read_text().splitlines()]
    epochs = [r for r in log if r["event"] == "epoch"]
    hist = state["metrics"]["history"]
    best = min(hist, key=lambda h: h["val_loss"])
    split = json.loads((work / "split_summary.json").read_text())

    def macro(h):
        a = [v for v in h["val_auc_weak"].values() if v is not None]
        return sum(a) / len(a) if a else None

    r = {"epochs": state["epoch"], "dataset_fingerprint": state["dataset_fingerprint"], "global_step": state["global_step"],
         "train_studies": split["train_studies"], "validation_studies": split["validation_studies"],
         "shuffle_train_labels": split["shuffle_train_labels"], "val_equals_train": split["val_equals_train"],
         "leakage": json.loads((work / "leakage.json").read_text()), "baselines": json.loads((work / "baselines.json").read_text()),
         "best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "best_val_macro_auc": macro(best),
         "best_val_auc": {t: a for t, a in best["val_auc_weak"].items() if a is not None},
         "final_val_loss": hist[-1]["val_loss"], "final_val_macro_auc": macro(hist[-1]), "first_train_loss": hist[0]["train_loss"],
         "final_train_loss": hist[-1]["train_loss"], "max_val_macro_auc": max((macro(h) or 0) for h in hist),
         "history": [{"epoch": h["epoch"], "lr": h["lr"], "train_loss": h["train_loss"], "val_loss": h["val_loss"], "val_macro_auc": macro(h),
                      "grad_norm_mean": h.get("grad_norm_mean"), "grad_norm_max": h.get("grad_norm_max"),
                      "grad_clipped_fraction": h.get("grad_clipped_fraction"), "nonfinite_grad_steps": h.get("nonfinite_grad_steps"),
                      "val_prob_std": h.get("val_prob_std"), "val_prob_mean": h.get("val_prob_mean"), "invalid": h["invalid_samples"]} for h in hist],
         "epoch_seconds": [round(e["seconds"], 1) for e in epochs], "peak_cuda_gb": max((e["cuda_peak_gb"] or 0) for e in epochs),
         "sessions": sum(r_["event"] == "session_start" for r_ in log),
         "finite": all(h["train_loss"] == h["train_loss"] and h["val_loss"] == h["val_loss"] for h in hist),
         "last_pt_sha256": sha(work / "last.pt"), "last_best_sha256": sha(work / "last_best.pt"),
         "last_pt_mb": round((work / "last.pt").stat().st_size / 1e6, 1)}
    if not keep:  # controls: hashes are recorded, the 137 MB weights are not needed off Kaggle
        for n in ("last.pt", "last_best.pt"):
            (work / n).unlink()
        r["checkpoints_kept"] = False
    else:
        r["checkpoints_kept"] = True
    return r


def go(name, epochs, *extra, keep=False, optional=False, **kw):
    if optional and elapsed() > BUDGET:
        SUMMARY["skipped_for_time"] = SUMMARY.get("skipped_for_time", []) + [name]
        return None
    SUMMARY["stage"] = name
    work, proc = train(name, epochs, *extra, **kw)
    if proc.returncode != 0:
        SUMMARY["runs"][name] = {"failure": (proc.stderr or proc.stdout)[-800:]}
        stop(f"run {name} failed")
    if kw.get("stop_after"):
        return work
    r = report(name, work, keep)
    SUMMARY["runs"][name] = r
    save()
    if not (r["finite"] and r["leakage"]["clean"]):
        stop(f"run {name} failed a check")
    return work


# ---- A: overfit (validation == training studies by design), with a resume-equivalence check
go("overfit64_ref", E_OVERFIT, "--val-equals-train", subset=(64, 8))
go("overfit64_resume", E_OVERFIT, "--val-equals-train", subset=(64, 8), stop_after=5)
go("overfit64_resume", E_OVERFIT, "--val-equals-train", subset=(64, 8), keep=True)  # second session resumes from last.pt
ref_h, res_h = SUMMARY["runs"]["overfit64_ref"]["history"], SUMMARY["runs"]["overfit64_resume"]["history"]
SUMMARY["resume_equivalence"] = {"sessions": SUMMARY["runs"]["overfit64_resume"]["sessions"], "epochs_equal": len(ref_h) == len(res_h),
                                 "max_abs_train_loss_diff": max(abs(a["train_loss"] - b["train_loss"]) for a, b in zip(ref_h, res_h)),
                                 "max_abs_val_loss_diff": max(abs(a["val_loss"] - b["val_loss"]) for a, b in zip(ref_h, res_h))}

# ---- the Phase 14 reference configuration, re-run with diagnostics (reproducibility), and its shuffled-label control
go("real_e6_s0", E_SHORT, keep=True)
go("shuf_e6_s0", E_SHORT, "--shuffle-train-labels", 100)
go("real_e12_s0", E_LONG, keep=True)
go("shuf_e12_s0", E_LONG, "--shuffle-train-labels", 100)
go("overfit256", 40, "--val-equals-train", subset=(256, 20), optional=True)
go("real_e12_s1", E_LONG, seed=1, optional=True)
go("shuf_e12_s1", E_LONG, "--shuffle-train-labels", 101, seed=1, optional=True)
go("real_e12_s0_lr3e-4", E_LONG, lr=3e-4, optional=True)
go("real_e12_s0_lr1e-4", E_LONG, lr=1e-4, optional=True)

# ---- exports (hash-verified) for the kept runs
SUMMARY["stage"] = "export"
SUMMARY["exports"] = {}
for name in ("overfit64_resume", "real_e6_s0", "real_e12_s0"):
    dest = OUT / "exports" / name
    ok = run(EXP + ["export", "--work-dir", OUT / "runs" / name, "--export-dir", dest], f"export_{name}").returncode == 0
    m = json.loads((dest / "MANIFEST.json").read_text()) if ok else {}
    SUMMARY["exports"][name] = {"ok": ok and all(sha(dest / n) == i["sha256"] for n, i in m.items() if n != "runtime"),
                                "last_pt_sha256": m.get("last.pt", {}).get("sha256")}
    for n in ("last.pt", "weak_label_run_export.zip"):  # duplicates of the run's own files (hashes are in MANIFEST.json)
        if (dest / n).is_file():
            (dest / n).unlink()

# ---- verdicts (descriptive; nothing here is a clinical claim)
runs = SUMMARY["runs"]


def get(name, key):
    return runs[name][key] if name in runs else None


SUMMARY["reproducibility_vs_phase14"] = {
    "fingerprint_matches_phase14": runs["real_e6_s0"]["dataset_fingerprint"] == PHASE14_FINGERPRINT,
    "best_val_loss": runs["real_e6_s0"]["best_val_loss"], "phase14_best_val_loss": PHASE14_REFERENCE["best_val_loss"],
    "best_val_macro_auc": runs["real_e6_s0"]["best_val_macro_auc"], "phase14_best_macro_auc": PHASE14_REFERENCE["best_weak_macro_auc"]}
o = runs["overfit64_ref"]
SUMMARY["verdict"] = {
    "overfit64": {"train_loss_first_to_final": [o["first_train_loss"], o["final_train_loss"]], "final_fit_macro_auc": o["final_val_macro_auc"],
                  "passed": bool(o["final_val_macro_auc"] and o["final_val_macro_auc"] >= 0.9 and o["final_train_loss"] < 0.5 * o["first_train_loss"])},
    "real_vs_shuffled_final_macro_auc": {s: [get(f"real_e12_{s}", "final_val_macro_auc"), get(f"shuf_e12_{s}", "final_val_macro_auc")] for s in ("s0", "s1")},
    "real_vs_shuffled_best_val_loss": {s: [get(f"real_e12_{s}", "best_val_loss"), get(f"shuf_e12_{s}", "best_val_loss")] for s in ("s0", "s1")},
    "val_loss_constant_baselines": {k: v for k, v in runs["real_e12_s0"]["baselines"].items() if k.startswith("val_loss")}}
SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = all(r.get("finite") for r in runs.values()) and all(e["ok"] for e in SUMMARY["exports"].values())
save()
print(json.dumps({k: SUMMARY[k] for k in ("resume_equivalence", "reproducibility_vs_phase14", "verdict", "all_passed", "wall_seconds")}, indent=2, default=str))
sys.exit(0 if SUMMARY["all_passed"] else 1)
