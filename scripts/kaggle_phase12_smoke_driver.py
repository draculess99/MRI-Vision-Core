"""Kaggle driver: bounded GPU smoke test of Model C (C0 random-init ResNet-18 and C2 = C0 + GRU) on real TRAINING studies.

Private kernel, internet off, competition data + code bundle + Phase 10 kernel output (decode cache) attached.
No gold evaluation, no test images, no submission, no pretrained weights. Per candidate:
  1. 64-study stratified subset, 2 planned epochs, stop after epoch 1, then RESUME to epoch 2 (checkpoint save/restore, AMP scaler)
  2. one epoch on the 256-study Phase 9 subset to time a larger epoch and measure peak VRAM / throughput
Also measures cold DICOM decoding on 16 studies with an empty cache and checks that the restored cache is really used.
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
PREV = pathlib.Path(os.environ.get("PREV_INPUT", INPUT))
OUT = WORKING / "phase12_smoke"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = WORKING / "decode_cache"
SUMMARY = {"stage": "start", "stopped": None}
T0 = time.perf_counter()


def save():
    SUMMARY["wall_seconds"] = round(time.perf_counter() - T0, 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    print(json.dumps(SUMMARY, indent=2, default=str))
    sys.exit(1)


def run(cmd, name, timeout=3000):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-2500:])
    if proc.stderr.strip():
        print("[stderr]", proc.stderr[-1500:])
    SUMMARY.setdefault("steps", {})[name] = {"exit_code": proc.returncode, "seconds": round(time.perf_counter() - started, 1)}
    (OUT / f"{name}.log").write_text((proc.stdout or "") + "\n[stderr]\n" + (proc.stderr or ""))
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

# ---- manifest (original production labels, profile 0.1.0) must match the local reference
ref = expected["manifest_reference"]
manifest = OUT / "manifest" / "weak_manifest_tier1.jsonl"
if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv", "--test-csv",
        meta_dir / "test.csv", "--sample-submission", meta_dir / "sample_submission.csv", "--active-targets",
        "ACL,Medial Meniscus,Lateral Meniscus", "--output", manifest], "build_manifest").returncode != 0:
    stop("manifest regeneration failed")
msum = json.loads(manifest.with_name(manifest.name + ".summary.json").read_text())
SUMMARY["manifest"] = {"sha256": msum["manifest_sha256"], "matches_reference": msum["manifest_sha256"] == ref["manifest_sha256"],
                       "leakage_checks": msum["leakage_checks"]}
if not SUMMARY["manifest"]["matches_reference"]:
    stop("manifest differs from the local reference")

# ---- restore the decode cache (before any preflight) from the Phase 10 kernel output
markers = glob.glob(str(PREV / "**" / "PHASE10_MARKER.json"), recursive=True)
if not markers:
    stop("Phase 10 kernel output (PHASE10_MARKER.json) is not attached")
src = pathlib.Path(markers[0]).parent / json.loads(pathlib.Path(markers[0]).read_text())["decode_cache_dir"]
shutil.copytree(src, CACHE)
SUMMARY["cache_restored_entries"] = len(list(CACHE.glob("*.pt")))

base_cfg = json.loads((code / "configs" / "weak_label_full_tier1.json").read_text())
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]


def write_cfg(name, **kw):
    cfg = {**base_cfg, "arm": "C", "amp": True, "val_fraction": 0.2, "num_workers": min(4, os.cpu_count() or 2),
           "cache_dir": str(CACHE), "early_stopping_patience": 0, **kw}
    path = OUT / f"cfg_{name}.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def common(cfg, subset, min_per_class):
    return ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest, "--config", cfg,
            "--smoke-subset", subset, "--min-per-class", min_per_class]


# ---- restored cache is really used; cold decode timing on a separate empty cache
cfg0 = write_cfg("c0", epochs=2)
proc = run(EXP + ["real-preflight", *common(cfg0, 64, 8), "--work-dir", OUT / "preflight_restored"], "preflight_restored")
pre = json.loads((OUT / "preflight_restored" / "real_preflight.json").read_text())
SUMMARY["preflight_restored"] = {k: pre[k] for k in ("passed", "studies_selected", "decoded_ok", "cache_hits", "cache_misses",
                                                      "decode_seconds_per_study", "tensor_shape", "problems")}
if proc.returncode != 0 or pre["cache_hits"] != pre["studies_selected"] or pre["cache_misses"] != 0:
    stop("restored cache was not used for every study")
cold_cfg = write_cfg("cold", epochs=1, cache_dir=str(pathlib.Path("/tmp") / "cold_cache"))
run(EXP + ["real-preflight", *common(cold_cfg, 16, 2), "--work-dir", OUT / "preflight_cold"], "preflight_cold")
cold = json.loads((OUT / "preflight_cold" / "real_preflight.json").read_text())
SUMMARY["cold_decode"] = {k: cold[k] for k in ("passed", "studies_selected", "cache_hits", "cache_misses", "decode_seconds_per_study")}
if not cold["passed"] or cold["cache_misses"] != cold["studies_selected"]:
    stop("cold decode measurement failed")


def report(work):
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    log = [json.loads(l) for l in (work / "runtime_log.jsonl").read_text().splitlines()]
    epochs = [r for r in log if r["event"] == "epoch"]
    hist = state["metrics"]["history"]
    return {"epoch": state["epoch"], "global_step": state["global_step"], "history": [
                {"epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": h["val_loss"], "invalid": h["invalid_samples"],
                 "train_images": h["train_images"]} for h in hist],
            "epoch_seconds": [round(e["seconds"], 2) for e in epochs], "data_wait_seconds": [e["data_wait_seconds"] for e in epochs],
            "peak_cuda_gb": max((e["cuda_peak_gb"] or 0) for e in epochs), "sessions_in_log": sum(r["event"] == "session_start" for r in log),
            "images_per_second": [round(e["train_images"] / max(e["seconds"], 1e-9), 1) for e in epochs],
            "scaler_scale": (state["scaler"] or {}).get("scale"), "last_pt_sha256": sha(work / "last.pt"),
            "last_best_sha256": sha(work / "last_best.pt"), "finite": all(h["train_loss"] == h["train_loss"] and h["val_loss"] == h["val_loss"]
                                                                       and abs(h["train_loss"]) != float("inf") for h in hist),
            "params_million": round(sum(v.numel() for v in state["model"].values()) / 1e6, 2)}


all_ok = True
for name, extra in (("C0", {}), ("C2", {"sequence": "gru"})):
    SUMMARY["stage"] = name
    cfg = write_cfg(name.lower(), epochs=2, **extra)
    work = OUT / f"{name}_smoke64"
    first = run(EXP + ["train", *common(cfg, 64, 8), "--work-dir", work, "--export-dir", OUT / f"durable_{name}",
                       "--stop-after-epochs", 1], f"train_{name}_epoch1")
    if first.returncode != 0:
        stop(f"{name} epoch 1 failed")
    mid = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    before = {"epoch": mid["epoch"], "step": mid["global_step"], "sha": sha(work / "last.pt")}
    second = run(EXP + ["train", *common(cfg, 64, 8), "--work-dir", work, "--export-dir", OUT / f"durable_{name}"],
                 f"train_{name}_resume")
    if second.returncode != 0:
        stop(f"{name} resume failed")
    r = report(work)
    r["restore"] = {"checkpoint_before_resume": before, "resumed_to_epoch": r["epoch"], "sessions_in_log": r["sessions_in_log"],
                    "step_continued": r["global_step"] > before["step"]}
    exp = run(EXP + ["export", "--work-dir", work, "--export-dir", OUT / f"export_{name}"], f"export_{name}")
    m = json.loads((OUT / f"export_{name}" / "MANIFEST.json").read_text())
    r["export_ok"] = exp.returncode == 0 and all(sha(OUT / f"export_{name}" / n) == i["sha256"] for n, i in m.items() if n != "runtime")
    # one timed epoch on the 256-study subset
    big = run(EXP + ["train", *common(write_cfg(name.lower() + "_256", epochs=1, **extra), 256, 24), "--work-dir", OUT / f"{name}_epoch256"],
              f"train_{name}_256")
    r["epoch_256"] = report(OUT / f"{name}_epoch256") if big.returncode == 0 else {"failed": (big.stderr or big.stdout)[-600:]}
    r["passed"] = bool(r["finite"] and r["export_ok"] and r["restore"]["step_continued"] and big.returncode == 0
                       and all(h["invalid"] == 0 for h in r["history"]))
    SUMMARY[name] = r
    all_ok &= r["passed"]

SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = all_ok
save()
print(json.dumps(SUMMARY, indent=2, default=str))
sys.exit(0 if all_ok else 1)
