"""Kaggle driver for Phase 10: full Tier 1 weak-label training (Models A then B), private kernel, internet off.

Order of operations (the cache is restored BEFORE the preflight, and the preflight proves the restored tensors are used):
  1. verify the code bundle, regenerate + compare the manifest, check disk/GPU
  2. restore decode cache (and, in a continuation session, the previous arm directories) from attached kernel outputs
  3. preflight on the 256 Phase 9 studies: every one must be a cache HIT with zero decodes
  4. cache-warm every selected study with worker processes; unreadable studies go to an exclusion list
  5. train Model A (max 10 epochs, early stopping on weak-validation loss, time-boxed), then Model B on the identical split
  6. verify checkpoints/exports; write a marker so a later session can continue from this one's output

Never uses gold labels, never reads test images, never submits. Epoch-boundary recovery only.
Environment overrides (BUNDLE_DIR, KAGGLE_INPUT, KAGGLE_WORKING, PREV_INPUT, SMOKE_DEVICE_NOTE) are for local dry runs.
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

MAX_EPOCHS = int(os.environ.get("P10_EPOCHS", "10"))
SESSION_BUDGET_SECONDS = float(os.environ.get("P10_BUDGET", str(7 * 3600)))  # leaves headroom under the 9 h platform limit
SAFETY_MARGIN = float(os.environ.get("P10_MARGIN", "900"))
RESTORE_SUBSET, RESTORE_MIN_PER_CLASS = 256, 24  # the Phase 9 selection whose cache entries are restored

INPUT = pathlib.Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORKING = pathlib.Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
PREV = pathlib.Path(os.environ.get("PREV_INPUT", INPUT))
CACHE = WORKING / "decode_cache"
T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# --- continuation: a previous Phase 10 session's output may be attached
prev_markers = sorted(glob.glob(str(PREV / "**" / "PHASE10_MARKER.json"), recursive=True))
PREV_MARKER = json.loads(pathlib.Path(prev_markers[0]).read_text()) if prev_markers else None
SESSION = (PREV_MARKER["session"] + 1) if PREV_MARKER else 1
OUT = WORKING / f"phase10_s{SESSION}"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"session": SESSION, "stage": "start", "stopped": None, "continuation_of": PREV_MARKER and PREV_MARKER["session"]}


def save():
    SUMMARY["session_wall_seconds"] = round(elapsed(), 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason: str):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    print(json.dumps(SUMMARY, indent=2, default=str))
    sys.exit(1)


def run(cmd, name, timeout=30000):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    started = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    print((proc.stdout or "")[-4000:])
    if proc.stderr.strip():
        print("[stderr]", proc.stderr[-1500:])
    SUMMARY.setdefault("steps", {})[name] = {"exit_code": proc.returncode, "seconds": round(time.perf_counter() - started, 1)}
    (OUT / f"{name}.log").write_text((proc.stdout or "") + "\n[stderr]\n" + (proc.stderr or ""))
    return proc


# ------------------------------------------------------------------ 1. bundle, data, manifest
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

import torch  # noqa: E402
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402

try:
    meta_dir, dicom_root = discover_data(INPUT)
except Exception as exc:
    stop(f"competition data not found or ambiguous: {exc}")
SUMMARY["environment"] = {**device_info(), "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1), "cpu_count": os.cpu_count()}
if not SUMMARY["environment"]["cuda_available"] and not os.environ.get("SMOKE_DEVICE_NOTE"):
    stop("no CUDA device available")
if SUMMARY["environment"]["free_disk_gb"] < 6:
    stop("less than 6 GB free in /kaggle/working (the full decode cache needs about 1.6 GB plus checkpoints)")

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
if any(msum["leakage_checks"][k] for k in ("overlap_with_gold", "overlap_with_test", "overlap_with_holdout")):
    stop("leakage check failed")

# ------------------------------------------------------------------ 2. restore (cache BEFORE preflight)
restored = {"sources": []}
cache_sources = []
if PREV_MARKER:
    prev_root = pathlib.Path(prev_markers[0]).parent
    cache_sources.append(prev_root / "decode_cache")
    for arm in ("A", "B"):
        info = PREV_MARKER["arms"].get(arm)
        if info and info.get("dir"):
            src = prev_root / info["dir"]
            if sha(src / "last.pt") != info["last_pt_sha256"]:
                stop(f"restored Model {arm} checkpoint hash differs from the previous session's marker")
            shutil.copytree(src, OUT / f"arm{arm}")
            restored[f"arm{arm}_restored_epoch"] = info["epoch"]
else:  # first session: restore the Phase 9 cache (256 studies) so its tensors are reused instead of re-decoded
    for marker in glob.glob(str(PREV / "**" / "SESSION1_MARKER.json"), recursive=True):
        cache_sources.append(pathlib.Path(marker).parent / json.loads(pathlib.Path(marker).read_text())["decode_cache_dir"])
for source in cache_sources:
    if source.is_dir():
        shutil.copytree(source, CACHE, dirs_exist_ok=True)
        restored["sources"].append({"path": str(source), "entries": len(list(source.glob("*.pt")))})
restored["entries_after_restore"] = len(list(CACHE.glob("*.pt"))) if CACHE.is_dir() else 0
SUMMARY["restore"] = restored

tier1 = json.loads((code / "configs" / "weak_label_full_tier1.json").read_text())
config = {**tier1, "epochs": MAX_EPOCHS, "num_workers": min(4, os.cpu_count() or 2), "cache_dir": str(CACHE)}
cfg = OUT / "config_phase10.json"
cfg.write_text(json.dumps(config, indent=2))
COMMON = ["--data-dir", meta_dir, "--dicom-root", dicom_root, "--manifest", manifest, "--config", cfg]
EXP = [sys.executable, code / "scripts" / "kaggle_weak_label_experiment.py"]

# ------------------------------------------------------------------ 3. preflight proves the restored tensors are used
if restored["entries_after_restore"]:
    proc = run(EXP + ["real-preflight", *COMMON, "--smoke-subset", RESTORE_SUBSET, "--min-per-class", RESTORE_MIN_PER_CLASS,
                      "--work-dir", OUT / "preflight_restored"], "preflight_restored")
    path = OUT / "preflight_restored" / "real_preflight.json"
    pre = json.loads(path.read_text()) if path.is_file() else {}
    SUMMARY["preflight_restored"] = {k: pre.get(k) for k in ("passed", "studies_selected", "decoded_ok", "problems", "cache_hits",
                                                              "cache_misses", "decode_seconds_per_study", "cache_bytes")}
    if proc.returncode != 0 or not pre.get("passed"):
        stop("preflight on restored cache failed")
    if not PREV_MARKER and not os.environ.get("SMOKE_DEVICE_NOTE") and (pre["cache_hits"] != pre["studies_selected"] or pre["cache_misses"] != 0):
        stop(f"restored cache was not used: {pre['cache_hits']} hits / {pre['cache_misses']} misses")

# ------------------------------------------------------------------ 4. warm the cache for every study, find unreadable ones
SUMMARY["stage"] = "cache_warm"
if run(EXP + ["cache-warm", *COMMON, "--work-dir", OUT / "warm"], "cache_warm", timeout=20000).returncode != 0:
    stop("cache warming failed")
warm = json.loads((OUT / "warm" / "cache_warm.json").read_text())
SUMMARY["cache_warm"] = warm
exclude = OUT / "warm" / "excluded_unreadable.json"
n_bad = len(json.loads(exclude.read_text()))
if n_bad > 0.02 * warm["studies"]:
    stop(f"{n_bad} of {warm['studies']} studies unreadable (> 2%); fix the data handling before training")
TRAIN_COMMON = [*COMMON, "--exclude-studies", exclude]


# ------------------------------------------------------------------ 5. training
def arm_report(work: pathlib.Path) -> dict:
    state = torch.load(work / "last.pt", map_location="cpu", weights_only=True)
    log = [json.loads(l) for l in (work / "runtime_log.jsonl").read_text().splitlines()]
    epochs = [r for r in log if r["event"] == "epoch"]
    hist = state["metrics"]["history"]
    best = min(hist, key=lambda h: h["val_loss"])
    return {
        "epoch": state["epoch"], "global_step": state["global_step"], "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "epochs": [{"epoch": h["epoch"], "train_loss": h["train_loss"], "val_loss": h["val_loss"],
                    "val_macro_auc_weak": h["val_macro_auc_weak"],
                    "val_auc_weak": {t: a for t, a in h["val_auc_weak"].items() if a is not None},
                    "train_images": h["train_images"], "invalid_samples": h["invalid_samples"], "lr": h["lr"]} for h in hist],
        "epoch_seconds": [round(e["seconds"], 1) for e in epochs], "data_wait_seconds": [e["data_wait_seconds"] for e in epochs],
        "peak_cuda_gb": max((e["cuda_peak_gb"] or 0) for e in epochs) if epochs else None,
        "sessions_in_log": sum(1 for r in log if r["event"] == "session_start"),
        "pos_weight": next((r["pos_weight"] for r in log if r["event"] == "session_start"), None),
        "trainable_entries": next((r["trainable_entries"] for r in log if r["event"] == "session_start"), None),
        "train_studies": len(state["train_study_ids"]), "valid_studies": len(state["valid_study_ids"]),
        "dataset_fingerprint": state["dataset_fingerprint"], "last_pt_sha256": sha(work / "last.pt"),
        "last_best_sha256": sha(work / "last_best.pt"), "invalid_total": sum(h["invalid_samples"] for h in hist),
        "finite": all(x == x and abs(x) != float("inf") for h in hist for x in (h["train_loss"], h["val_loss"]))}


def export(work: pathlib.Path, name: str) -> dict:
    dest = OUT / f"export_{name}"
    if run(EXP + ["export", "--work-dir", work, "--export-dir", dest], f"export_{name}").returncode != 0:
        return {"ok": False}
    m = json.loads((dest / "MANIFEST.json").read_text())
    return {"ok": all(sha(dest / n) == i["sha256"] for n, i in m.items() if n != "runtime"),
            "hashes": {n: i["sha256"][:16] for n, i in m.items() if n != "runtime"}}


def train_arm(arm: str) -> dict:
    work = OUT / f"arm{arm}"
    remaining = SESSION_BUDGET_SECONDS - elapsed() - SAFETY_MARGIN
    entry = SUMMARY.setdefault("arms", {}).setdefault(arm, {})
    if remaining < 120:
        entry["status"] = "not_started_time_budget"
        return entry
    proc = run(EXP + ["train", *TRAIN_COMMON, "--arm", arm, "--work-dir", work, "--export-dir", OUT / f"durable_arm{arm}",
                      "--max-seconds", int(remaining)], f"train_{arm}", timeout=int(remaining) + 1500)
    entry["exit_code"] = proc.returncode
    if proc.returncode != 0:
        entry["failure"] = (proc.stderr or proc.stdout)[-900:]
        entry["status"] = "failed"
        return entry
    ends = [json.loads(l) for l in (work / "runtime_log.jsonl").read_text().splitlines() if '"session_end"' in l]
    status = ends[-1]["status"] if ends else None
    entry.update(arm_report(work))
    entry["status"] = status or "unknown"
    entry["export"] = export(work, arm)
    verify = run(EXP + ["verify-checkpoint", *TRAIN_COMMON, "--arm", arm, "--checkpoint", work / "last.pt",
                        "--work-dir", OUT / f"verify_{arm}"], f"verify_{arm}")
    entry["verify_exit"] = verify.returncode
    entry["dir"] = f"phase10_s{SESSION}/arm{arm}"
    entry["integrity_ok"] = bool(entry["finite"] and entry["export"]["ok"] and verify.returncode == 0 and entry["invalid_total"] == 0)
    return entry


def finished(arm):  # complete or early stopped in this or an earlier session
    e = SUMMARY.get("arms", {}).get(arm, {})
    return e.get("status") in ("complete", "early_stopped")


if PREV_MARKER:
    for arm, info in PREV_MARKER["arms"].items():
        SUMMARY.setdefault("arms", {})[arm] = {"status": info["status"], "carried_from_session": PREV_MARKER["session"]}

SUMMARY["stage"] = "armA"
if not finished("A"):
    a = train_arm("A")
    if a.get("status") in ("failed",) or (a.get("status") not in ("not_started_time_budget", "paused_time_budget") and not a.get("integrity_ok")):
        stop("Model A failed an integrity or training check; Model B was not run")
SUMMARY["stage"] = "armB"
if finished("A") and not finished("B"):
    b = train_arm("B")
    if b.get("status") == "failed" or (b.get("status") not in ("not_started_time_budget", "paused_time_budget") and not b.get("integrity_ok")):
        stop("Model B failed an integrity or training check")

# ------------------------------------------------------------------ 6. marker + comparison
arms = SUMMARY["arms"]
marker = {"session": SESSION, "git_commit": expected["git_commit"], "manifest_sha256": got["manifest_sha256"],
          "decode_cache_dir": "decode_cache", "decode_cache_entries": len(list(CACHE.glob("*.pt"))),
          "arms": {arm: {"status": e.get("status"), "epoch": e.get("epoch"), "dir": e.get("dir"), "last_pt_sha256": e.get("last_pt_sha256")}
                   for arm, e in arms.items() if e.get("dir")}}
for arm, e in arms.items():  # arms carried over from a previous session keep their old directory reference
    if arm not in marker["arms"] and PREV_MARKER and arm in PREV_MARKER["arms"]:
        marker["arms"][arm] = PREV_MARKER["arms"][arm]
(WORKING / "PHASE10_MARKER.json").write_text(json.dumps(marker, indent=2))
if "A" in arms and "B" in arms and arms["A"].get("epochs") and arms["B"].get("epochs"):
    SUMMARY["comparison"] = {
        "same_dataset_fingerprint": arms["A"]["dataset_fingerprint"] == arms["B"]["dataset_fingerprint"],
        "best_val_loss": {"A": arms["A"]["best_val_loss"], "B": arms["B"]["best_val_loss"]},
        "best_epoch": {"A": arms["A"]["best_epoch"], "B": arms["B"]["best_epoch"]},
        "best_epoch_weak_auc": {a: arms[a]["epochs"][arms[a]["best_epoch"] - 1]["val_auc_weak"] for a in ("A", "B")}}
SUMMARY["marker"] = marker
SUMMARY["stage"] = "done"
SUMMARY["all_finished"] = finished("A") and finished("B")
save()
print(json.dumps(SUMMARY, indent=2, default=str))
sys.exit(0)
