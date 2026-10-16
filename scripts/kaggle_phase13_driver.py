"""Kaggle driver, Phase 13: bounded C0-vs-C2 x label-profile comparison on training-only validation (seeds 0 and 1).

Conditions: architecture {C0 = ResNet-18 2.5D Attention-MIL, C2 = C0 + bidirectional GRU} x labels {rules 0.1.0, rules 0.2.2}.
Each run: 6 epochs, no early stopping (so runs are comparable), FP16 AMP, selection of the best epoch by the run's own weak-validation
loss. Because that loss cannot rank label versions, every run is ALSO scored on a common yardstick: validation studies shared by both
manifests, on labels both manifests agree on. Private kernel, internet off, no gold, no test images, no pretrained weights, no submission.
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

SEEDS = [int(x) for x in os.environ.get("P13_SEEDS", "0,1").split(",")]
EPOCHS = int(os.environ.get("P13_EPOCHS", "6"))
BUDGET = float(os.environ.get("P13_BUDGET", str(3 * 3600)))
INPUT = pathlib.Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORKING = pathlib.Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
PREV = pathlib.Path(os.environ.get("PREV_INPUT", INPUT))
OUT = WORKING / "phase13"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = WORKING / "decode_cache"
SUMMARY = {"stage": "start", "stopped": None, "seeds": SEEDS, "epochs": EPOCHS}
T0 = time.perf_counter()


def save():
    SUMMARY["wall_seconds"] = round(time.perf_counter() - T0, 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    print(json.dumps(SUMMARY, indent=2, default=str)[:6000])
    sys.exit(1)


def run(cmd, name, timeout=7000):
    print(f"\n$ {' '.join(str(c) for c in cmd)[:260]}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-1200:])
    if proc.returncode:
        print("[stderr]", proc.stderr[-1800:])
    SUMMARY.setdefault("steps", {})[name] = {"exit_code": proc.returncode, "seconds": round(time.perf_counter() - started, 1)}
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs" / f"{name}.log").write_text((proc.stdout or "") + "\n[stderr]\n" + (proc.stderr or ""))
    return proc


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


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

import torch  # noqa: E402
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402

meta_dir, dicom_root = discover_data(INPUT)
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1), "cpu_count": os.cpu_count()}
if not SUMMARY["environment"]["cuda_available"] and not os.environ.get("SMOKE_DEVICE_NOTE"):
    stop("no CUDA device available")
if SUMMARY["environment"]["free_disk_gb"] < 8:
    stop("less than 8 GB free in /kaggle/working")

# ---- both manifests, reproduced inside Kaggle and compared with the local references
MANIFESTS = {}
for name, profile, ref in (("v010", "0.1.0", expected["manifest_reference"]), ("v022", "0.2.2", expected["manifest_reference_v022"])):
    path = OUT / "manifests" / f"{name}.jsonl"
    if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv", "--test-csv",
            meta_dir / "test.csv", "--sample-submission", meta_dir / "sample_submission.csv", "--active-targets",
            "ACL,Medial Meniscus,Lateral Meniscus", "--extractor-profile", profile, "--output", path], f"manifest_{name}").returncode != 0:
        stop(f"manifest {name} regeneration failed")
    msum = json.loads(path.with_name(path.name + ".summary.json").read_text())
    ok = msum["manifest_sha256"] == ref["manifest_sha256"] and msum["counts"]["included"] == ref["included"]
    SUMMARY.setdefault("manifests", {})[name] = {"profile": profile, "sha256": msum["manifest_sha256"], "included": msum["counts"]["included"],
                                                 "trainable": {t: [v["positive"], v["negative"]] for t, v in msum["trainable"].items()},
                                                 "matches_local_reference": ok, "leakage_checks": msum["leakage_checks"]}
    if not ok or any(msum["leakage_checks"][k] for k in ("overlap_with_gold", "overlap_with_test", "overlap_with_holdout")):
        stop(f"manifest {name} differs from the local reference or failed a leakage check")
    MANIFESTS[name] = path

# ---- restore cache (before anything reads pixels) and warm it for both manifests
marker = glob.glob(str(PREV / "**" / "PHASE10_MARKER.json"), recursive=True)
if not marker:
    stop("Phase 10 kernel output (decode cache) is not attached")
shutil.copytree(pathlib.Path(marker[0]).parent / "decode_cache", CACHE)
SUMMARY["cache_restored_entries"] = len(list(CACHE.glob("*.pt")))
base = json.loads((code / "configs" / "weak_label_full_tier1.json").read_text())
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]


def write_cfg(name, **kw):
    cfg = {**base, "arm": "C", "amp": True, "val_fraction": 0.2, "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE),
           "epochs": EPOCHS, "early_stopping_patience": 0, **kw}
    path = OUT / "configs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    return path


EXCLUDE = {}
SUMMARY["stage"] = "cache_warm"
for name in MANIFESTS:
    if run(EXP + ["cache-warm", "--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", MANIFESTS[name],
                  "--config", write_cfg("warm"), "--work-dir", OUT / f"warm_{name}"], f"warm_{name}").returncode != 0:
        stop(f"cache warm failed for {name}")
    warm = json.loads((OUT / f"warm_{name}" / "cache_warm.json").read_text())
    SUMMARY["manifests"][name]["cache_warm"] = warm
    EXCLUDE[name] = OUT / f"warm_{name}" / "excluded_unreadable.json"
    if warm["unreadable"] > 0.02 * warm["studies"]:
        stop(f"{warm['unreadable']} unreadable studies for {name}")
union = sorted(set(json.loads(EXCLUDE["v010"].read_text())) | set(json.loads(EXCLUDE["v022"].read_text())))
(OUT / "excluded_union.json").write_text(json.dumps(union))
EXCL = OUT / "excluded_union.json"

# ---- the bounded comparison
CONDITIONS = [(labels, arch) for labels in ("v010", "v022") for arch in ("C0", "C2")]


def best_of(state):
    hist = state["metrics"]["history"]
    best = min(hist, key=lambda h: h["val_loss"])
    return best, hist


def run_condition(labels, arch, seed):
    tag = f"{labels}_{arch}_s{seed}"
    work = OUT / "runs" / tag
    cfg = write_cfg(tag, seed=seed, **({"sequence": "gru"} if arch == "C2" else {}))
    entry = {"labels": labels, "arch": arch, "seed": seed}
    started = time.perf_counter()
    proc = run(EXP + ["train", "--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", MANIFESTS[labels], "--config", cfg,
                      "--exclude-studies", EXCL, "--arm", "C", "--work-dir", work], f"train_{tag}")
    if proc.returncode != 0:
        entry["failure"] = (proc.stderr or proc.stdout)[-700:]
        return entry, False
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    best_state = torch.load(work / "last_best.pt", map_location="cpu", weights_only=True)
    best, hist = best_of(state)
    log = [json.loads(l) for l in (work / "runtime_log.jsonl").read_text().splitlines()]
    epochs = [r for r in log if r["event"] == "epoch"]
    split = json.loads((work / "split_summary.json").read_text())
    entry.update({
        "train_studies": split["train_studies"], "validation_studies": split["validation_studies"],
        "class_counts": {"train": split["train_targets"], "validation": split["validation_targets"]},
        "pos_weight": next(r["pos_weight"] for r in log if r["event"] == "session_start")[:4],
        "history": [{"epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": h["val_loss"],
                     "val_auc_weak": {t: a for t, a in h["val_auc_weak"].items() if a is not None},
                     "invalid": h["invalid_samples"]} for h in hist],
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "best_epoch_weak_auc": {t: a for t, a in best["val_auc_weak"].items() if a is not None},
        "epoch_seconds": [round(e["seconds"], 1) for e in epochs], "data_wait_seconds": [e["data_wait_seconds"] for e in epochs],
        "peak_cuda_gb": max((e["cuda_peak_gb"] or 0) for e in epochs), "run_seconds": round(time.perf_counter() - started, 1),
        "finite": all(h["train_loss"] == h["train_loss"] and h["val_loss"] == h["val_loss"] and abs(h["val_loss"]) != float("inf") for h in hist),
        "invalid_total": sum(h["invalid_samples"] for h in hist), "dataset_fingerprint": state["dataset_fingerprint"],
        "last_pt_sha256": sha(work / "last.pt"), "last_best_sha256": sha(work / "last_best.pt"),
        "best_checkpoint_epoch": int(best_state["epoch"])})
    ev = run(EXP + ["common-eval", "--data-dir", meta_dir, "--dicom-root", dicom_root, "--config", cfg, "--checkpoint", work / "last_best.pt",
                    "--manifests", f"v010={MANIFESTS['v010']}", f"v022={MANIFESTS['v022']}", "--exclude-studies", EXCL,
                    "--work-dir", work / "common_eval", "--rounds", "500"], f"common_{tag}")
    entry["common_yardstick"] = json.loads((work / "common_eval" / "common_eval.json").read_text()) if ev.returncode == 0 else {"failed": True}
    exp = run(EXP + ["export", "--work-dir", work, "--export-dir", OUT / "exports" / tag], f"export_{tag}")
    m = json.loads((OUT / "exports" / tag / "MANIFEST.json").read_text()) if exp.returncode == 0 else {}
    entry["export_ok"] = bool(m) and all(sha(OUT / "exports" / tag / n) == i["sha256"] for n, i in m.items() if n != "runtime")
    ok = entry["finite"] and entry["invalid_total"] == 0 and entry["export_ok"] and ev.returncode == 0
    entry["passed"] = ok
    return entry, ok


SUMMARY["stage"] = "runs"
SUMMARY["runs"] = []
for seed in SEEDS:
    for labels, arch in CONDITIONS:
        if time.perf_counter() - T0 > BUDGET:
            SUMMARY["time_budget_reached"] = True
            break
        entry, ok = run_condition(labels, arch, seed)
        SUMMARY["runs"].append(entry)
        save()
        if not ok:
            stop(f"run {labels}/{arch}/seed{seed} failed an integrity check")

# ---- aggregation: per condition mean over seeds, and paired differences
import numpy as np  # noqa: E402

TIER = ["ACL", "Medial Meniscus", "Lateral Meniscus"]


def common_macro(r):
    return r["common_yardstick"].get("macro_auc")


def common_t(r, t):
    return r["common_yardstick"]["per_target"][t]["auc"]


agg = {}
for labels, arch in CONDITIONS:
    rs = [r for r in SUMMARY["runs"] if r["labels"] == labels and r["arch"] == arch]
    if not rs:
        continue
    macros = [common_macro(r) for r in rs if common_macro(r) is not None]
    agg[f"{labels}_{arch}"] = {
        "seeds": [r["seed"] for r in rs], "best_val_loss_mean": float(np.mean([r["best_val_loss"] for r in rs])),
        "best_epochs": [r["best_epoch"] for r in rs],
        "own_weak_auc_mean": {t: float(np.mean([r["best_epoch_weak_auc"].get(t) for r in rs])) for t in TIER
                              if all(r["best_epoch_weak_auc"].get(t) is not None for r in rs)},
        "common_macro_auc": macros, "common_macro_auc_mean": float(np.mean(macros)) if macros else None,
        "common_auc_mean": {t: float(np.mean([common_t(r, t) for r in rs])) for t in TIER if all(common_t(r, t) is not None for r in rs)},
        "epoch_seconds_mean": float(np.mean([np.mean(r["epoch_seconds"]) for r in rs])),
        "peak_cuda_gb": max(r["peak_cuda_gb"] for r in rs)}
SUMMARY["aggregate"] = agg


def paired(a_tag, b_tag):
    out = []
    for seed in SEEDS:
        ra = next((r for r in SUMMARY["runs"] if f"{r['labels']}_{r['arch']}" == a_tag and r["seed"] == seed), None)
        rb = next((r for r in SUMMARY["runs"] if f"{r['labels']}_{r['arch']}" == b_tag and r["seed"] == seed), None)
        if ra and rb and common_macro(ra) is not None and common_macro(rb) is not None:
            out.append(common_macro(ra) - common_macro(rb))
    return {"per_seed_common_macro_auc_difference": out, "mean": float(np.mean(out)) if out else None}


SUMMARY["paired_common_yardstick"] = {
    "C2_minus_C0 (labels v010)": paired("v010_C2", "v010_C0"), "C2_minus_C0 (labels v022)": paired("v022_C2", "v022_C0"),
    "v022_minus_v010 (C0)": paired("v022_C0", "v010_C0"), "v022_minus_v010 (C2)": paired("v022_C2", "v010_C2")}
SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = all(r.get("passed") for r in SUMMARY["runs"]) and len(SUMMARY["runs"]) == len(SEEDS) * len(CONDITIONS)
save()
print(json.dumps({k: SUMMARY[k] for k in ("aggregate", "paired_common_yardstick", "all_passed", "wall_seconds")}, indent=2, default=str))
sys.exit(0 if SUMMARY["all_passed"] else 1)
