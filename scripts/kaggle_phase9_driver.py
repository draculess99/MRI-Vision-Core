"""Kaggle driver for the Phase 9 bounded multi-epoch experiment (256 studies, <= 3 epochs, Models A then B).

Two separate kernels run this file with SESSION = 1 or 2 (the constant is substituted when the kernel folder is built):

  SESSION 1  verify bundle, regenerate + compare the manifest, real-data preflight (populates the decode cache in
             /kaggle/working), train Model A for ONE epoch of a planned three, verify and export the checkpoint.
  SESSION 2  attached to session 1's kernel OUTPUT. Verifies that checkpoint's hash against the marker AND an independently
             recorded hash, restores it into a new directory together with the decode cache, checks compatibility
             (fingerprint, split, config, state), resumes Model A to epoch 3, trains an uninterrupted reference A and Model B,
             and compares the resumed run with the reference.

Private kernel, internet off, competition data attached. Never evaluates on gold, never reads test images, never submits.
Environment overrides exist only for local dry runs.
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

SESSION = 0  # substituted: 1 or 2
EXPECTED_S1_LAST_SHA256 = ""  # substituted in session 2 from the locally downloaded session-1 output
STUDIES, MIN_PER_CLASS, EPOCHS = 256, 24, 3

INPUT = pathlib.Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORKING = pathlib.Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
S1_INPUT = pathlib.Path(os.environ.get("S1_INPUT", INPUT))
OUT = WORKING / f"phase9_s{SESSION}"
CACHE = WORKING / "decode_cache"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"session": SESSION, "stage": "start", "stopped": None}
T0 = time.perf_counter()


def save():
    SUMMARY["session_wall_seconds"] = round(time.perf_counter() - T0, 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason: str):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    print(json.dumps(SUMMARY, indent=2, default=str))
    sys.exit(1)


def run(cmd, name, timeout=3300):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-5000:])
    if proc.stderr.strip():
        print("[stderr]", proc.stderr[-2000:])
    SUMMARY.setdefault("steps", {})[name] = {"exit_code": proc.returncode, "seconds": round(time.perf_counter() - started, 1)}
    (OUT / f"{name}.log").write_text((proc.stdout or "") + "\n[stderr]\n" + (proc.stderr or ""))
    return proc


def sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def dir_bytes(path) -> int:
    return sum(f.stat().st_size for f in pathlib.Path(path).rglob("*") if f.is_file()) if pathlib.Path(path).is_dir() else 0


# ------------------------------------------------------------------ 1. bundle integrity
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
actual = {str(p.relative_to(code)).replace("\\", "/"): sha(p) for p in code.rglob("*")
          if p.is_file() and "__pycache__" not in p.parts}
if actual != expected["files"]:
    stop("code differs from the verified bundle: " + str(sorted(set(actual) ^ set(expected["files"]))[:5]))
SUMMARY["bundle"] = {"git_commit": expected["git_commit"], "files": len(actual), "zip_sha256": expected["zip_sha256"]}
sys.path[:0] = [str(code), str(code / "scripts")]

# ------------------------------------------------------------------ 2. data and manifest
import torch  # noqa: E402
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402

try:
    meta_dir, dicom_root = discover_data(INPUT)
except Exception as exc:
    stop(f"competition data not found or ambiguous: {exc}")
if any(not (meta_dir / n).is_file() for n in ("train.csv", "train_series.csv", "test.csv", "sample_submission.csv")):
    stop("competition metadata files missing")
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1),
                          "cpu_count": os.cpu_count()}
if not SUMMARY["environment"]["cuda_available"] and not os.environ.get("SMOKE_DEVICE_NOTE"):
    stop("no CUDA device available")

ref = expected["manifest_reference"]
manifest = OUT / "manifest" / "weak_manifest_tier1.jsonl"
if run([sys.executable, code / "scripts" / "build_weak_label_manifest.py", "--train-csv", meta_dir / "train.csv",
        "--test-csv", meta_dir / "test.csv", "--sample-submission", meta_dir / "sample_submission.csv",
        "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus", "--output", manifest], "build_manifest").returncode != 0:
    stop("manifest regeneration failed")
msum = json.loads(manifest.with_name(manifest.name + ".summary.json").read_text())
got = {"manifest_sha256": msum["manifest_sha256"], "included": msum["counts"]["included"],
       "trainable": {t: [v["positive"], v["negative"]] for t, v in msum["trainable"].items()}}
SUMMARY["manifest"] = {"sha256": got["manifest_sha256"], "included": got["included"], "leakage_checks": msum["leakage_checks"],
                       "matches_local_reference": all(got[k] == ref[k] for k in got)}
if not SUMMARY["manifest"]["matches_local_reference"]:
    stop("regenerated manifest differs from the local reference")

tier1 = json.loads((code / "configs" / "weak_label_tier1.json").read_text())
config = {**tier1, "epochs": EPOCHS, "val_fraction": 0.2, "amp": True, "num_workers": min(4, os.cpu_count() or 2),
          "cache_dir": str(CACHE)}
cfg = OUT / "config_phase9.json"
cfg.write_text(json.dumps(config, indent=2))
COMMON = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest, "--config", cfg,
          "--smoke-subset", STUDIES, "--min-per-class", MIN_PER_CLASS]
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]


def preflight(tag: str) -> dict:
    proc = run(EXP + ["real-preflight", *COMMON, "--work-dir", OUT / f"preflight_{tag}"], f"preflight_{tag}")
    path = OUT / f"preflight_{tag}" / "real_preflight.json"
    pre = json.loads(path.read_text()) if path.is_file() else {}
    SUMMARY[f"preflight_{tag}"] = {k: pre.get(k) for k in (
        "passed", "studies_selected", "train", "validation", "decoded_ok", "problems", "decode_seconds_per_study",
        "cache_bytes", "class_balance", "unmet_quota", "excluded_missing_series", "free_disk_gb")}
    if proc.returncode != 0 or not pre.get("passed"):
        stop(f"real-data preflight ({tag}) failed; training not started")
    return pre


def arm_report(work: pathlib.Path) -> dict:
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    log = [json.loads(l) for l in (work / "runtime_log.jsonl").read_text().splitlines()]
    epochs = [r for r in log if r["event"] == "epoch"]
    hist = state["metrics"]["history"]
    return {"epoch": state["epoch"], "global_step": state["global_step"],
            "epochs": [{"epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": h["val_loss"],
                        "val_macro_auc_weak": h["val_macro_auc_weak"],
                        "val_auc_weak": {t: a for t, a in h["val_auc_weak"].items() if a is not None},
                        "train_images": h["train_images"], "invalid_samples": h["invalid_samples"], "lr": h["lr"]} for h in hist],
            "epoch_seconds": [round(e["seconds"], 2) for e in epochs], "data_wait_seconds": [e["data_wait_seconds"] for e in epochs],
            "peak_cuda_gb": max((e["cuda_peak_gb"] or 0) for e in epochs) if epochs else None,
            "sessions_in_log": sum(1 for r in log if r["event"] == "session_start"),
            "pos_weight": next((r["pos_weight"] for r in log if r["event"] == "session_start"), None),
            "train_studies": len(state["train_study_ids"]), "valid_studies": len(state["valid_study_ids"]),
            "dataset_fingerprint": state["dataset_fingerprint"], "last_pt_sha256": sha(work / "last.pt"),
            "last_best_sha256": sha(work / "last_best.pt"), "finite": all(
                x == x and abs(x) != float("inf") for h in hist for x in (h["train_loss"], h["val_loss"]))}


def export(work: pathlib.Path, name: str) -> dict:
    dest = OUT / f"export_{name}"
    proc = run(EXP + ["export", "--work-dir", work, "--export-dir", dest], f"export_{name}")
    if proc.returncode != 0:
        return {"ok": False}
    manifest_json = json.loads((dest / "MANIFEST.json").read_text())
    ok = all(sha(dest / n) == i["sha256"] for n, i in manifest_json.items() if n != "runtime")
    return {"ok": ok, "hashes": {n: i["sha256"][:16] for n, i in manifest_json.items() if n != "runtime"}}


def train(arm: str, work: pathlib.Path, *extra) -> subprocess.CompletedProcess:
    return run(EXP + ["train", *COMMON, "--arm", arm, "--work-dir", work, "--export-dir", OUT / f"durable_{work.name}", *extra],
               f"train_{work.name}")


SUMMARY["stage"] = "preflight"
preflight("cold" if SESSION == 1 else "restored_cache")

# ------------------------------------------------------------------ SESSION 1
if SESSION == 1:
    SUMMARY["stage"] = "armA_epoch1"
    workA = OUT / "armA"
    proc = train("A", workA, "--stop-after-epochs", 1)
    if proc.returncode != 0:
        stop("Model A training failed in session 1")
    report = arm_report(workA)
    verify = run(EXP + ["verify-checkpoint", *COMMON, "--checkpoint", workA / "last.pt", "--work-dir", OUT / "verify"], "verify_s1")
    exp = export(workA, "A_s1")
    SUMMARY["armA_session1"] = {**report, "verify_exit": verify.returncode, "export": exp}
    if report["epoch"] != 1 or not report["finite"] or verify.returncode != 0 or not exp["ok"]:
        stop("session-1 checkpoint failed an integrity check")
    marker = {"session": 1, "git_commit": expected["git_commit"], "manifest_sha256": got["manifest_sha256"],
              "arm": "A", "epoch": report["epoch"], "global_step": report["global_step"], "planned_epochs": EPOCHS,
              "last_pt_sha256": report["last_pt_sha256"], "last_best_sha256": report["last_best_sha256"],
              "dataset_fingerprint": report["dataset_fingerprint"], "arm_dir": "phase9_s1/armA",
              "decode_cache_dir": "decode_cache", "decode_cache_entries": len(list(CACHE.glob("*.pt"))),
              "decode_cache_bytes": dir_bytes(CACHE)}
    (WORKING / "SESSION1_MARKER.json").write_text(json.dumps(marker, indent=2))
    SUMMARY["marker"] = marker
    SUMMARY["stage"] = "done"
    save()
    print(json.dumps(SUMMARY, indent=2, default=str))
    sys.exit(0)

# ------------------------------------------------------------------ SESSION 2
SUMMARY["stage"] = "restore"
markers = glob.glob(str(S1_INPUT / "**" / "SESSION1_MARKER.json"), recursive=True)
if not markers:
    stop("session-1 kernel output (SESSION1_MARKER.json) is not attached")
s1_root = pathlib.Path(markers[0]).parent
marker = json.loads(pathlib.Path(markers[0]).read_text())
s1_arm = s1_root / marker["arm_dir"]
restore = {"source": str(s1_root), "marker_last_pt_sha256": marker["last_pt_sha256"],
           "independent_expected_sha256": EXPECTED_S1_LAST_SHA256 or None, "actual_sha256": sha(s1_arm / "last.pt")}
restore["matches_marker"] = restore["actual_sha256"] == marker["last_pt_sha256"]
restore["matches_independent_record"] = (restore["actual_sha256"] == EXPECTED_S1_LAST_SHA256) if EXPECTED_S1_LAST_SHA256 else None
restore["bundle_commit_matches"] = marker["git_commit"] == expected["git_commit"]
restore["manifest_matches"] = marker["manifest_sha256"] == got["manifest_sha256"]
SUMMARY["restore"] = restore
if not (restore["matches_marker"] and restore["matches_independent_record"] is not False and restore["manifest_matches"]):
    stop("restored session-1 checkpoint failed hash/manifest verification")
workA = OUT / "armA_resumed"
shutil.copytree(s1_arm, workA)
restored_cache = s1_root / marker["decode_cache_dir"]
shutil.copytree(restored_cache, CACHE, dirs_exist_ok=True)
SUMMARY["restore"]["decode_cache_entries_restored"] = len(list(CACHE.glob("*.pt")))
SUMMARY["restore"]["decode_cache_entries_expected"] = marker["decode_cache_entries"]
before = torch.load(workA / "last.pt", map_location="cpu", weights_only=True)
SUMMARY["restore"]["state_before_resume"] = {"epoch": before["epoch"], "global_step": before["global_step"],
    "scheduler_last_epoch": (before["scheduler"] or {}).get("last_epoch"), "scaler_scale": (before["scaler"] or {}).get("scale"),
    "has_cuda_rng": bool(before.get("cuda_rng")), "saved_cuda_rng_devices": len(before.get("cuda_rng") or [])}
verify = run(EXP + ["verify-checkpoint", *COMMON, "--checkpoint", workA / "last.pt", "--work-dir", OUT / "verify"], "verify_restored")
SUMMARY["restore"]["verify_exit"] = verify.returncode
try:
    vtxt = (OUT / "verify_restored.log").read_text()
    SUMMARY["restore"]["verify_report"] = json.loads(vtxt[vtxt.index('{\n  "checkpoint_sha256"'):vtxt.index("\n}\n") + 2])
except Exception as exc:
    SUMMARY["restore"]["verify_report"] = f"unparsed: {exc}"
if verify.returncode != 0:
    stop("restored checkpoint is incompatible with this session's manifest/selection/split")

SUMMARY["stage"] = "armA_resume"
if train("A", workA).returncode != 0:
    stop("Model A failed while resuming in session 2")
repA = arm_report(workA)
repA["resumed_from_epoch"] = before["epoch"]
repA["export"] = export(workA, "A_resumed")

SUMMARY["stage"] = "armA_reference"
workRef = OUT / "armA_uninterrupted_reference"
if train("A", workRef).returncode != 0:
    stop("uninterrupted reference run for Model A failed")
repRef = arm_report(workRef)
a = torch.load(workA / "last.pt", map_location="cpu", weights_only=True)
b = torch.load(workRef / "last.pt", map_location="cpu", weights_only=True)
diffs = [float((a["model"][k].float() - b["model"][k].float()).abs().max()) for k in a["model"]]
repA["resume_vs_uninterrupted"] = {
    "same_global_step": a["global_step"] == b["global_step"], "weights_bitwise_equal": all(d == 0.0 for d in diffs),
    "max_abs_weight_diff": max(diffs),
    "loss_history_resumed": [round(h["train_loss"], 6) for h in a["metrics"]["history"]],
    "loss_history_reference": [round(h["train_loss"], 6) for h in b["metrics"]["history"]],
    "note": "epoch 1 of the resumed run came from session 1 (different machine); CUDA/AMP runs are not guaranteed bitwise"}
repA["state_after_resume"] = {"epoch": a["epoch"], "global_step": a["global_step"],
                              "scheduler_last_epoch": (a["scheduler"] or {}).get("last_epoch"),
                              "scaler_scale": (a["scaler"] or {}).get("scale"), "sessions_in_log": repA["sessions_in_log"]}
SUMMARY["armA"] = repA
SUMMARY["armA_reference"] = repRef
a_ok = (repA["epoch"] == EPOCHS and repA["finite"] and repA["export"]["ok"] and repA["state_after_resume"]["global_step"] > before["global_step"]
        and all(e["invalid_samples"] == 0 for e in repA["epochs"]))
if not a_ok:
    stop("Model A did not meet the success criteria; Model B was not run")

SUMMARY["stage"] = "armB"
workB = OUT / "armB"
if train("B", workB).returncode != 0:
    stop("Model B training failed")
repB = arm_report(workB)
repB["export"] = export(workB, "B")
SUMMARY["armB"] = repB
SUMMARY["same_split_for_A_and_B"] = (repA["dataset_fingerprint"] == repB["dataset_fingerprint"]
                                     and repA["train_studies"] == repB["train_studies"])
SUMMARY["total_epoch_seconds_gpu"] = round(sum(sum(r["epoch_seconds"]) for r in (repA, repRef, repB)), 1)
SUMMARY["stage"] = "done"
SUMMARY["all_passed"] = bool(repB["finite"] and repB["epoch"] == EPOCHS and repB["export"]["ok"] and SUMMARY["same_split_for_A_and_B"])
save()
print(json.dumps(SUMMARY, indent=2, default=str))
sys.exit(0 if SUMMARY["all_passed"] else 1)
