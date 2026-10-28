"""Phase 19B bounded GPU smoke test (PREPARED, NOT LAUNCHED): pretrained ConvNeXt-T arm D, TRAINING studies only, no labels used.

  verify code bundle, the torchvision ImageNet-1K weights (SHA-256) and the frozen seed-201 partition; build arm D with the pretrained trunk and fresh heads; for batch sizes 2, 3, 4
  run two AMP train steps (finite loss and gradients, peak VRAM, speed; an out-of-memory is recorded, not fatal); save a full resumable checkpoint and reload it into fresh objects.
Smoke labels are synthetic coin flips on two targets: they exist only to drive a backward pass and are never used for any result. At most 8 registered TRAINING studies are read.
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
DRY = bool(os.environ.get("P19_DRY"))
SEED = 201
SIZE = int(os.environ.get("P19_IMAGE_SIZE", "256"))
EXPECTED_WEIGHTS_SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"  # torchvision convnext_tiny IMAGENET1K_V1
OUT = WORKING / "phase19_smoke"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = {"experiment": "phase19b_gpu_smoke", "stage": "start", "stopped": None, "labels": "synthetic (smoke only)"}
T0 = time.perf_counter()


def save():
    SUMMARY["wall_seconds"] = round(time.perf_counter() - T0, 1)
    (OUT / "final_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def stop(reason):
    SUMMARY["stopped"] = reason
    save()
    print("STOP:", reason)
    sys.exit(0)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def _find_bundle(base, max_depth=3, prune=("train_series", "test_series")):
    found, frontier = [], [base]
    for depth in range(max_depth + 1):
        nxt = []
        for directory in frontier:
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                if entry.is_file() and entry.name == "BUNDLE_FILES.json":
                    found.append(entry.path)
                elif entry.is_dir() and entry.name not in prune and depth < max_depth:
                    nxt.append(entry.path)
        frontier = nxt
    return sorted(found)


side = _find_bundle(pathlib.Path(os.environ.get("BUNDLE_DIR", INPUT)))
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
    for rel in expected["files"]:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(next(root.rglob("kaggle_weak_label_experiment.py")).parent.parent / rel, code / rel)
actual = {str(p.relative_to(code)).replace("\\", "/"): sha(p) for p in code.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
if actual != expected["files"]:
    stop("code differs from the verified bundle")
SUMMARY["bundle"] = {"git_commit": expected["git_commit"], "files": len(actual)}
sys.path[:0] = [str(code), str(code / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from kaggle_rsna_runner import device_info, discover_data  # noqa: E402
from mri_core.kaggle_inputs import InputsError, locate_inputs_package  # noqa: E402
from mri_core.rsna_checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from mri_core.rsna_knee_dataset import TARGET_COLUMNS, load_rsna_metadata  # noqa: E402
from mri_core.weak_label_data import WeakLabelDicomDataset, assert_no_leakage, hash_split, load_weak_manifest  # noqa: E402
from mri_core.weak_label_models import build_model  # noqa: E402
from mri_core.weak_label_train import _scheduler  # noqa: E402

env = device_info()
SUMMARY["environment"] = {**env, "free_disk_gb": round(shutil.disk_usage(WORKING).free / 1e9, 1)}
if not env["cuda_available"] and not DRY:
    stop("no CUDA device available")
device = torch.device("cuda" if env["cuda_available"] else "cpu")
REG = json.loads((code / "configs" / "phase17_registry.json").read_text())

# ---- weights: official torchvision ImageNet-1K file, hash-pinned
try:
    WDIR, WPROV = locate_inputs_package(INPUT, "phase19_weights", override=os.environ.get("P19_WEIGHTS"))
except InputsError as error:
    SUMMARY["weights_diagnosis"] = {"code": error.code, **error.diagnostics}
    stop(f"weights package problem ({error.code}): {error}")
WEIGHTS = WDIR / "convnext_tiny-983f1562.pth"
if sha(WEIGHTS) != EXPECTED_WEIGHTS_SHA256:
    stop("the weights file is not the expected torchvision convnext_tiny IMAGENET1K_V1 file")
SUMMARY["weights"] = {"sha256": EXPECTED_WEIGHTS_SHA256, "source": WPROV.get("source_url"), "license_note": WPROV.get("license_note")}

# ---- inputs and the frozen seed-201 partition (training studies only)
try:
    IN, _ = locate_inputs_package(INPUT, "phase17a_inputs", override=os.environ.get("P19_INPUTS"))
except InputsError as error:
    SUMMARY["inputs_diagnosis"] = {"code": error.code, **error.diagnostics}
    stop(f"inputs package problem ({error.code}): {error}")
selection = json.loads((IN / "selected_studies.json").read_text())
if sha_ids(selection) != REG["training_pool"]["sha256"] and not DRY:
    stop("selection differs from the registered training pool")
meta_dir, dicom_root = discover_data(INPUT)
metadata = load_rsna_metadata(meta_dir, dicom_root)
manifest_path = OUT / "manifest" / "weak_manifest_tier1.jsonl"  # same regeneration as the Phase 19A driver, so the split is the registered one
proc = subprocess.run([sys.executable, str(code / "scripts" / "build_weak_label_manifest.py"), "--train-csv", str(meta_dir / "train.csv"), "--test-csv", str(meta_dir / "test.csv"),
                       "--sample-submission", str(meta_dir / "sample_submission.csv"), "--active-targets", "ACL,Medial Meniscus,Lateral Meniscus", "--output", str(manifest_path)],
                      capture_output=True, text=True)
if proc.returncode != 0:
    SUMMARY["manifest_stderr"] = proc.stderr[-800:]
    stop("manifest regeneration failed")
manifest = load_weak_manifest(manifest_path)
if manifest.sha256 != REG["manifest_sha256"] and not DRY:
    stop("manifest differs from the registered manifest")
order = [u for u in manifest.ids if u in set(selection)]
try:
    tr, va = hash_split(order, 0.2, SEED)
except ValueError:
    stop("empty split (rehearsal data too small)")
train_ids, val_ids = [order[i] for i in tr], [order[i] for i in va]
if sha_ids(val_ids) != REG["partitions"][str(SEED)]["validation_sha256"] and not DRY:
    stop("validation partition differs from the registry")
gold = set(metadata.complete_label_train["StudyInstanceUID"].astype(str))
test = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
assert_no_leakage(train_ids, val_ids, gold_ids=gold, test_ids=test)


def present(uid):
    rows = metadata.series_for(uid)
    return all(metadata.dicom_series_dir(uid, str(s)).is_dir() for s in rows["SeriesInstanceUID"]) and len(rows) > 0


smoke_ids = [u for u in train_ids if present(u)][:8]
if len(smoke_ids) < 4:
    stop(f"only {len(smoke_ids)} registered training studies are available")
assert not set(smoke_ids) & (set(val_ids) | gold | test)
SUMMARY["partition"] = {"train": len(train_ids), "validation": len(val_ids), "validation_sha256": sha_ids(val_ids), "smoke_studies": len(smoke_ids), "smoke_all_training": True,
                        "overlap_with_validation_gold_test": 0}
save()

# ---- model: pretrained trunk, fresh heads
cfg = {"arm": "D", "image_size": SIZE, "slices_per_series": 9, "planes": ["Axial", "Coronal", "Sagittal"], "dropout": 0.2, "drop_path": 0.1, "pretrained_weights": str(WEIGHTS),
       "learning_rate": 1e-4, "weight_decay": 0.05, "warmup_epochs": 1, "epochs": 12, "amp": True, "seed": SEED, "batch_size": 2, "cache_dir": None}
torch.manual_seed(SEED)
model = build_model("D", cfg)
torch.manual_seed(SEED)
fresh = build_model("D", {**cfg, "pretrained_weights": None})
file_state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
trunk_equal = all(torch.equal(model.trunk.state_dict()[k.replace("classifier.0.", "out_norm.")], v) for k, v in file_state.items() if k.startswith(("features.", "classifier.0.")))
heads_fresh = all(torch.equal(a, b) for (n, a), (_, b) in zip(model.state_dict().items(), fresh.state_dict().items()) if not n.startswith("trunk."))
trunk_differs = not all(torch.equal(a, b) for (n, a), (_, b) in zip(model.state_dict().items(), fresh.state_dict().items()) if n.startswith("trunk."))
SUMMARY["model"] = {"parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 3), "trunk_equals_weights_file": trunk_equal,
                    "heads_equal_a_fresh_seeded_init": heads_fresh, "trunk_differs_from_random_init": trunk_differs}
if not (trunk_equal and heads_fresh and trunk_differs):
    stop("pretrained trunk / fresh head initialisation check failed")
model.to(device)

# ---- data: <= 8 training studies, synthetic labels for two targets
rng = np.random.default_rng(SEED)
values = np.zeros((len(smoke_ids), 12), np.float32)
masks = np.zeros((len(smoke_ids), 12), np.float32)
for name in ("ACL", "Medial Meniscus"):
    j = TARGET_COLUMNS.index(name)
    values[:, j], masks[:, j] = rng.integers(0, 2, len(smoke_ids)), 1
dataset = WeakLabelDicomDataset(metadata, smoke_ids, values, masks, cfg, on_error="raise", cache_dir=None)
t = time.perf_counter()
samples = [dataset[i] for i in range(len(dataset))]
SUMMARY["decode_seconds_per_study"] = round((time.perf_counter() - t) / len(samples), 3)
images = torch.stack([s["images"] for s in samples])
mask = torch.stack([s["mask"] for s in samples])
labels = torch.stack([s["labels"] for s in samples])
tmask = torch.stack([s["target_mask"] for s in samples])
SUMMARY["input"] = {"image_tensor_per_study": list(images.shape[1:]), "valid_slices_min": int(mask.sum(dim=(1, 2)).min())}

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = _scheduler(optimizer, cfg)
scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
bce = torch.nn.BCEWithLogitsLoss(reduction="none")
runs = {}
for bs in ([2, 3, 4] if device.type == "cuda" else [2]):
    if bs > len(samples):
        continue
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    record = {"batch_size": bs, "images_per_step": int(mask[:bs].sum())}
    try:
        steps = []
        for step in range(2):  # the second step also holds the optimizer state
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images[:bs].to(device), mask[:bs].to(device))
            loss = (bce(logits.float(), labels[:bs].to(device)) * tmask[:bs].to(device)).sum() / tmask[:bs].sum().clamp(min=1).to(device)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            bad = sum(int(not torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            steps.append({"loss": float(loss), "loss_finite": bool(torch.isfinite(loss)), "grad_norm": norm, "nonfinite_grad_tensors": bad, "seconds": round(time.perf_counter() - started, 3)})
        record.update({"steps": steps, "ok": all(s["loss_finite"] and s["nonfinite_grad_tensors"] == 0 and np.isfinite(s["grad_norm"]) for s in steps)})
        if device.type == "cuda":
            record.update({"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)})
    except torch.cuda.OutOfMemoryError:
        record.update({"ok": False, "oom": True})
        optimizer.zero_grad(set_to_none=True)
    runs[f"bs{bs}"] = record
    save()
    if record.get("oom"):
        break
SUMMARY["runs"] = runs
scheduler.step()

# ---- full resumable checkpoint: save, reload into fresh objects, compare
gen = torch.Generator().manual_seed(SEED)
ckpt = OUT / "smoke_checkpoint.pt"
save_checkpoint(ckpt, model=model, optimizer=optimizer, config=cfg, target_columns=TARGET_COLUMNS, epoch=1, global_step=4, train_study_ids=smoke_ids, valid_study_ids=[],
                dataset_fingerprint="smoke", metrics={"history": []}, scheduler=scheduler, scaler=scaler, loader_generator=gen)
model2 = build_model("D", {**cfg, "pretrained_weights": None}).to(device)
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=0.05)
sched2 = _scheduler(opt2, cfg)
scaler2 = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
state = load_checkpoint(ckpt, model=model2, optimizer=opt2, scheduler=sched2, scaler=scaler2, expected_target_columns=TARGET_COLUMNS, expected_dataset_fingerprint="smoke", device=device)
model.eval()
model2.eval()
with torch.no_grad():
    same_predictions = bool(torch.equal(model(images[:2].to(device), mask[:2].to(device)), model2(images[:2].to(device), mask[:2].to(device))))
opt_equal = all(torch.equal(a["exp_avg"], b["exp_avg"]) for a, b in zip(optimizer.state.values(), opt2.state.values())) and len(optimizer.state) == len(opt2.state)
SUMMARY["checkpoint"] = {"bytes": ckpt.stat().st_size, "sha256": sha(ckpt), "epoch": state.epoch, "global_step": state.global_step, "same_predictions_after_reload": same_predictions,
                         "optimizer_state_equal": opt_equal, "scheduler_last_epoch_equal": sched2.last_epoch == scheduler.last_epoch, "scaler_present": scaler is None or scaler2.get_scale() == scaler.get_scale(),
                         "has_rng_states": state.loader_rng_state is not None}
ok = [r for r in runs.values() if r.get("ok")]
SUMMARY["verdict"] = {"all_finite": bool(ok) and all(r["ok"] for r in runs.values() if not r.get("oom")), "largest_ok_batch": max((r["batch_size"] for r in ok), default=None),
                      "checkpoint_roundtrip": same_predictions and opt_equal}
SUMMARY["stage"] = "done"
save()
(ckpt).unlink(missing_ok=True)  # the smoke checkpoint is a throw-away; its hash is recorded above
print(json.dumps({k: SUMMARY[k] for k in ("model", "runs", "checkpoint", "verdict")}, indent=1, default=str)[:3000])
sys.exit(0)
