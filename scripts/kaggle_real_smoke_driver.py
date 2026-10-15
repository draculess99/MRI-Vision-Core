"""Kaggle kernel driver: bounded real-data smoke test (<= 64 studies, 1 epoch, Models A then B).

Runs inside a PRIVATE Kaggle kernel with the competition data and the code dataset attached and internet off.
It never touches the test studies' images, never evaluates on gold labels, never submits, and stops at the first failed
precondition. The weak-label manifest is regenerated here from the competition's own train.csv reports (nothing
derived is uploaded) and compared with the local reference.

Environment overrides (used by the local dry run): KAGGLE_INPUT, KAGGLE_WORKING, BUNDLE_DIR, SMOKE_STUDIES, SMOKE_TMP, SMOKE_DEVICE_NOTE.
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
STUDIES = int(os.environ.get("SMOKE_STUDIES", "64"))
TMP = pathlib.Path(os.environ.get("SMOKE_TMP", "/tmp"))
OUT = WORKING / "real_smoke"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"stage": "start", "stopped": None, "studies_requested": STUDIES}
T0 = time.perf_counter()


def save():
    SUMMARY["elapsed_seconds"] = round(time.perf_counter() - T0, 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason: str, code: int = 1):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    print(json.dumps(SUMMARY, indent=2, default=str))
    sys.exit(code)


def run(cmd, name, timeout=3000):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    tail = (proc.stdout or "")[-6000:]
    print(tail)
    if proc.stderr.strip():
        print("[stderr]", proc.stderr[-2500:])
    SUMMARY.setdefault("steps", {})[name] = {"exit_code": proc.returncode, "seconds": round(time.perf_counter() - started, 1)}
    (OUT / f"{name}.log").write_text((proc.stdout or "") + "\n[stderr]\n" + (proc.stderr or ""))
    return proc


# ---------------------------------------------------------------------------- 1. code bundle integrity
mounts = sorted(str(p) for p in INPUT.glob("*"))[:10]
print("input mounts:", mounts)
SUMMARY["input_mounts"] = mounts
side = glob.glob(str(pathlib.Path(os.environ.get("BUNDLE_DIR", INPUT)) / "**" / "BUNDLE_FILES.json"), recursive=True)
if not side:
    stop("code dataset with BUNDLE_FILES.json is not attached")
expected = json.loads(pathlib.Path(side[0]).read_text())
root = pathlib.Path(side[0]).parent
zips = glob.glob(str(root / "**" / "rsna_weak_label_code.zip"), recursive=True)
code = WORKING / "code"
if zips:
    import zipfile
    zipfile.ZipFile(zips[0]).extractall(code)
else:  # Kaggle auto-extracts uploaded zips
    shutil.copytree(next(p.parent.parent for p in root.rglob("kaggle_weak_label_experiment.py")), code, dirs_exist_ok=True)
actual = {str(p.relative_to(code)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in code.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
if actual != expected["files"]:
    stop("code differs from the verified bundle: " + str(sorted(set(actual) ^ set(expected["files"]))[:5]))
SUMMARY["bundle"] = {"git_commit": expected.get("git_commit"), "files": len(actual), "zip_sha256": expected.get("zip_sha256")}
print("bundle verified:", SUMMARY["bundle"])
sys.path[:0] = [str(code), str(code / "scripts")]

# ---------------------------------------------------------------------------- 2. data discovery
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402

try:
    meta_dir, dicom_root = discover_data(INPUT)
except Exception as exc:
    stop(f"competition data not found or ambiguous: {exc}")
needed = ["train.csv", "train_series.csv", "test.csv", "test_series.csv", "sample_submission.csv"]
missing = [n for n in needed if not (meta_dir / n).is_file()]
if missing or not (dicom_root / "train_series").is_dir():
    stop(f"missing competition files: {missing or 'train_series/'}")
n_train_dirs = sum(1 for _ in os.scandir(dicom_root / "train_series"))
SUMMARY["data"] = {"metadata_dir": str(meta_dir), "dicom_root": str(dicom_root), "train_series_study_dirs": n_train_dirs,
                   "test_series_present": (dicom_root / "test_series").is_dir(),
                   "free_disk_gb_working": round(shutil.disk_usage(WORKING).free / 1e9, 1),
                   "device": device_info()}
print(json.dumps(SUMMARY["data"], indent=2))
if not SUMMARY["data"]["device"].get("cuda_available") and not os.environ.get("SMOKE_DEVICE_NOTE"):
    stop("no CUDA device available")

# ---------------------------------------------------------------------------- 3. regenerate and verify the manifest
ref = expected["manifest_reference"]
manifest = OUT / "manifest" / "weak_manifest_tier1.jsonl"
proc = run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv",
            "--test-csv", meta_dir / "test.csv", "--sample-submission", meta_dir / "sample_submission.csv",
            "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus", "--output", manifest], "build_manifest")
if proc.returncode != 0:
    stop("manifest regeneration failed")
summary = json.loads(manifest.with_name(manifest.name + ".summary.json").read_text())
got = {"manifest_sha256": summary["manifest_sha256"], "included": summary["counts"]["included"],
       "eligible": summary["counts"]["eligible"], "unsupported_language": summary["counts"]["unsupported_language"],
       "below_min_decided": summary["counts"]["below_min_decided"],
       "trainable": {t: [v["positive"], v["negative"]] for t, v in summary["trainable"].items()},
       "train_csv_sha256": summary["train_csv_sha256"]}
SUMMARY["manifest"] = {"regenerated": got, "reference": {k: ref[k] for k in got}, "leakage_checks": summary["leakage_checks"],
                       "match": {k: got[k] == ref[k] for k in got}}
print(json.dumps(SUMMARY["manifest"]["match"], indent=2))
if not all(SUMMARY["manifest"]["match"].values()):
    stop("regenerated manifest differs from the local reference (see final_summary.json)")

# ---------------------------------------------------------------------------- 4. config and real-data preflight
tier1 = json.loads((code / "configs" / "weak_label_tier1.json").read_text())
config = {**tier1, "epochs": 1, "val_fraction": 0.25, "num_workers": min(4, os.cpu_count() or 2), "amp": True,
          "cache_dir": str(TMP / "decode_cache")}  # cache outside /kaggle/working so derived pixels are not part of the output
cfg_path = OUT / "smoke_config.json"
cfg_path.write_text(json.dumps(config, indent=2))
common = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest, "--config", cfg_path]
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]
proc = run(EXP + ["real-preflight", *common, "--smoke-subset", STUDIES, "--work-dir", OUT / "preflight"], "real_preflight")
pre = json.loads((OUT / "preflight" / "real_preflight.json").read_text()) if (OUT / "preflight" / "real_preflight.json").is_file() else {}
SUMMARY["preflight"] = {k: pre.get(k) for k in ("passed", "studies_selected", "train", "validation", "decoded_ok", "problems",
                                                 "decode_seconds_per_study", "valid_slices_per_study_mean", "cache_bytes",
                                                 "class_balance", "unmet_quota", "free_disk_gb", "gpu_memory_free_gb",
                                                 "excluded_missing_series")}
if proc.returncode != 0 or not pre.get("passed"):
    stop("real-data preflight failed; training not started")

# ---------------------------------------------------------------------------- 5. arms A then B
import torch  # noqa: E402


def train_arm(arm: str) -> bool:
    work, export = OUT / f"arm{arm}", OUT / f"export_{arm}"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    proc = run(EXP + ["train", *common, "--arm", arm, "--smoke-subset", STUDIES, "--epochs-override", 1, "--work-dir", work,
                      "--export-dir", OUT / f"durable_{arm}"], f"train_{arm}")
    entry = {"exit_code": proc.returncode}
    SUMMARY.setdefault("arms", {})[arm] = entry
    if proc.returncode != 0:
        entry["failure"] = (proc.stderr or proc.stdout)[-800:]
        return False
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    latest = state["metrics"]["latest"]
    runtime = [json.loads(l) for l in (work / "runtime_log.jsonl").read_text().splitlines()]
    epoch = next(r for r in runtime if r["event"] == "epoch")
    entry.update({
        "epoch": state["epoch"], "global_step": state["global_step"], "train_loss": latest["train_loss"], "val_loss": latest["val_loss"],
        "val_auc_weak": latest["val_auc_weak"], "invalid_samples": latest["invalid_samples"], "train_images": latest["train_images"],
        "train_studies": len(state["train_study_ids"]), "valid_studies": len(state["valid_study_ids"]),
        "epoch_seconds": round(epoch["seconds"], 2), "data_wait_seconds": epoch["data_wait_seconds"],
        "train_images_per_second": round(latest["train_images"] / max(epoch["seconds"], 1e-9), 1),
        "peak_cuda_gb": epoch["cuda_peak_gb"], "device": epoch["device"],
        "state_keys_present": {k: state.get(k) is not None for k in ("model", "optimizer", "scheduler", "scaler", "torch_rng",
                                                                       "python_rng", "numpy_rng", "cuda_rng", "loader_rng")},
        "dataset_fingerprint": state["dataset_fingerprint"], "checkpoint_on_cpu": all(v.device.type == "cpu" for v in state["model"].values()),
        "recovery": "epoch-boundary only: a crash inside an epoch repeats that epoch"})
    finite = all(x is not None and x == x and abs(x) != float("inf") for x in (entry["train_loss"], entry["val_loss"]))
    exp = run(EXP + ["export", "--work-dir", work, "--export-dir", export], f"export_{arm}")
    hashes_ok = False
    if exp.returncode == 0:
        manifest_json = json.loads((export / "MANIFEST.json").read_text())
        hashes_ok = all(hashlib.sha256((export / n).read_bytes()).hexdigest() == i["sha256"]
                        for n, i in manifest_json.items() if n != "runtime")
        durable_ok = hashlib.sha256((OUT / f"durable_{arm}" / "last.pt").read_bytes()).hexdigest() == \
            hashlib.sha256((work / "last.pt").read_bytes()).hexdigest()
        entry["export"] = {"hashes_match": hashes_ok, "durable_copy_matches": durable_ok,
                           "files": {n: i["bytes"] for n, i in manifest_json.items() if n != "runtime"}}
        hashes_ok = hashes_ok and durable_ok
    entry["checks"] = {"finite_losses": finite, "no_skipped_samples": entry["invalid_samples"] == 0, "epoch_is_1": entry["epoch"] == 1,
                       "full_state_present": all(v for k, v in entry["state_keys_present"].items() if k != "scaler")
                       and (entry["state_keys_present"]["scaler"] or not entry["device"].startswith("cuda")),
                       "export_integrity": hashes_ok}
    entry["passed"] = all(entry["checks"].values())
    return entry["passed"]


SUMMARY["stage"] = "arm_A"
if not train_arm("A"):
    stop("Model A smoke training failed an integrity or data check; Model B was not run")
SUMMARY["stage"] = "arm_B"
train_arm("B")
SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = all(a.get("passed") for a in SUMMARY["arms"].values())
save()
print(json.dumps(SUMMARY, indent=2, default=str))
sys.exit(0 if SUMMARY["all_passed"] else 1)
