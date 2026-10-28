"""Resumable masked-loss training, gold evaluation and prediction for the weak-label experiment.

Design points
  * Loss: masked multi-label BCE-with-logits. A target contributes only where the manifest mask is 1, scaled by a
    configurable per-target reliability weight; positive-class weights come from the TRAINING mask only.
  * Checkpoint selection uses only weak-label VALIDATION loss on held-out manifest studies. Gold studies are never
    loaded during training; `evaluate_gold` is a separate step guarded by a one-look ledger and a leakage assertion.
  * Full state (model, optimizer, scheduler, AMP scaler, Python/NumPy/torch/CUDA/loader RNG, epoch, step, config,
    metrics, study IDs, dataset fingerprint) is written atomically after every epoch via rsna_checkpoint. An
    interrupted run resumes from the last completed epoch and, on CPU, reproduces an uninterrupted run exactly.
  * Runtime/GPU usage is appended to runtime_log.jsonl so quota can be tracked across sessions.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .rsna_checkpoint import (CheckpointIncompatibleError, CheckpointMalformedError, load_checkpoint,
                              save_checkpoint)
from .rsna_knee_dataset import TARGET_COLUMNS
from .rsna_knee_train import RSNAKneeDicomDataset, seed_everything
from .weak_label_data import assert_no_leakage
from .weak_label_manifest import LeakageError
from .weak_label_models import build_model

INVALID_LOG = "invalid_studies.json"
AMP_DEVICE_TYPES = ("cuda",)  # tests may add "cpu" to exercise the scaler path without a GPU
LAST = "last.pt"
RUNTIME_LOG = "runtime_log.jsonl"
UPDATE_LOG = "update_log.jsonl"
IDENTITY_KEYS = ("arm", "seed", "learning_rate", "weight_decay", "batch_size", "image_size", "slices_per_series",
                 "planes", "dropout", "val_fraction", "reliability_weights", "max_pos_weight", "warmup_epochs", "amp")
DEFAULT_CONFIG = {
    "arm": "A", "seed": 0, "epochs": 15, "batch_size": 8, "learning_rate": 1e-3, "weight_decay": 1e-4,
    "warmup_epochs": 1, "grad_clip": 1.0, "early_stopping_patience": 0, "early_stopping_min_delta": 0.002, "dropout": 0.2, "image_size": 128, "slices_per_series": 9,
    "planes": ["Axial", "Coronal", "Sagittal"], "num_workers": 2, "val_fraction": 0.1, "max_pos_weight": 10.0,
    "reliability_weights": {}, "amp": False, "device": "auto", "max_invalid_fraction": 0.05, "deterministic": True,
    # optimizer-update accounting: gradients of `grad_accum_steps` batches are summed into ONE update; the update count (not the batch count) is the global step
    "grad_accum_steps": 1, "amp_init_scale": 65536.0, "amp_min_scale": 1.0, "max_consecutive_skipped_updates": 20, "amp_growth_interval": 2000,
}


def early_stop_due(history: Sequence[dict], patience: int, min_delta: float) -> bool:
    """True when the last `patience` epochs failed to beat the best weak-validation loss by more than `min_delta`.

    Derived purely from the recorded history, so it needs no extra checkpoint state and survives resume."""
    if not patience:
        return False
    best, since = float("inf"), 0
    for record in history:
        if record["val_loss"] < best - min_delta:
            best, since = record["val_loss"], 0
        else:
            since += 1
    return since >= patience


class TrainingAborted(RuntimeError):
    pass


class GoldAlreadyScoredError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- loss
def reliability_vector(config: dict) -> torch.Tensor:
    weights = config.get("reliability_weights") or {}
    unknown = set(weights) - set(TARGET_COLUMNS)
    if unknown:
        raise ValueError(f"reliability_weights has unknown targets: {sorted(unknown)}")
    vector = torch.tensor([float(weights.get(t, 1.0)) for t in TARGET_COLUMNS])
    if (vector < 0).any():
        raise ValueError("reliability weights must be non-negative")
    return vector


def compute_pos_weight(values: np.ndarray, masks: np.ndarray, max_pos_weight: float) -> torch.Tensor:
    """negatives/positives among trainable entries, per target; 1.0 where a class is absent; clipped to max."""
    on = masks == 1
    pos = ((values == 1) & on).sum(axis=0).astype(np.float64)
    neg = ((values == 0) & on).sum(axis=0).astype(np.float64)
    weight = np.where((pos > 0) & (neg > 0), neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(weight, 1.0, max_pos_weight), dtype=torch.float32)


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, target_mask: torch.Tensor, pos_weight: torch.Tensor,
               reliability: torch.Tensor) -> torch.Tensor:
    """Weighted mean BCE over trainable entries; exactly 0 (with a gradient path) when nothing is trainable."""
    per_entry = nn.functional.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none",
                                                               pos_weight=pos_weight)
    weight = target_mask * reliability
    total = weight.sum()
    if float(total) == 0.0:
        return logits.sum() * 0.0
    return (per_entry * weight).sum() / total


def macro_auc(per_target: dict) -> Optional[float]:
    """Mean over the targets whose AUC is defined; None (unavailable) when no target has both classes. Never a placeholder value."""
    defined = [a for a in per_target.values() if a is not None]
    return float(np.mean(defined)) if defined else None


def masked_auc(labels: np.ndarray, probs: np.ndarray, masks: np.ndarray) -> dict:
    """Per-target ROC-AUC over trainable entries only; None when a target lacks both classes."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for j, target in enumerate(TARGET_COLUMNS):
        on = masks[:, j] == 1
        y = labels[on, j]
        out[target] = float(roc_auc_score(y, probs[on, j])) if on.sum() and 0 < y.sum() < len(y) else None
    return out


# ----------------------------------------------------------------------------------------------- helpers
class InMemoryWeakDataset(Dataset):
    """Tensor-backed dataset with the WeakLabelDicomDataset interface (smoke tests / cached features)."""

    def __init__(self, ids, images, mask, values, masks):
        self.ids, self.images, self.mask = list(ids), images, mask
        self.values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
        self.masks = np.asarray(masks, dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"images": self.images[i], "mask": self.mask[i], "labels": torch.from_numpy(self.values[i]),
                "target_mask": torch.from_numpy(self.masks[i]), "valid": torch.tensor(True), "study_uid": self.ids[i]}


def _roi_inputs(batch, device) -> dict:
    """Optional ROI boxes carried by a batch (Phase 19A ROI models); empty for every earlier arm, whose forward() takes only (images, mask)."""
    return {k: batch[k].to(device) for k in ("rois", "roi_valid") if k in batch}


def _device(config: dict, override: Optional[str]) -> torch.device:
    name = override or config.get("device", "auto")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def _scheduler(optimizer, config: dict):
    warmup, epochs = int(config["warmup_epochs"]), max(int(config["epochs"]), 1)

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(epochs - warmup, 1))) * 0.99 + 0.01

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _write_update_rows(path: Path, rows: Sequence[dict]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=float) + "\n")


def _trim_update_log(path: Path, keep_epochs: int) -> None:
    """After a resume, keep only the update rows of epochs whose checkpoint exists."""
    if not path.is_file():
        return
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if json.loads(line)["epoch"] <= keep_epochs]
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def analyze_update_log(out_dir: Path, *, calibration_updates: int = 20, max_skipped: int = 10) -> dict:
    """Summarise update_log.jsonl: counts, scale trajectory, skips after start-up calibration and internal consistency. Descriptive; nothing here feeds back into training."""
    path = Path(out_dir) / UPDATE_LOG
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []
    if not rows:
        return {"updates": 0, "stable": False, "reason": "no update rows"}
    applied = sum(1 for r in rows if r["finite"])
    skipped_at = [r["update"] for r in rows if r["skipped"]]
    scales = [r["scale_after"] for r in rows if r["scale_after"] is not None]
    changes = [(r["update"], r["scale_before"], r["scale_after"]) for r in rows if r["scale_before"] != r["scale_after"]]
    norms = sorted(r["grad_norm"] for r in rows if r["grad_norm"] is not None)
    consistent = ([r["update"] for r in rows] == list(range(1, len(rows) + 1)) and applied + len(skipped_at) == len(rows)
                  and all(a["epoch"] <= b["epoch"] for a, b in zip(rows, rows[1:])) and rows[-1]["applied_total"] == applied)
    late = [u for u in skipped_at if u > calibration_updates]
    return {"updates": len(rows), "applied": applied, "skipped": len(skipped_at), "skipped_updates": skipped_at[:30], "skips_after_calibration": len(late),
            "scale_first": rows[0]["scale_before"], "scale_last": scales[-1] if scales else None, "scale_min": min(scales) if scales else None, "scale_max": max(scales) if scales else None,
            "scale_changes": changes[:40], "grad_norm_median": norms[len(norms) // 2] if norms else None, "grad_norm_max": norms[-1] if norms else None,
            "epochs": sorted({r["epoch"] for r in rows}), "consistent": consistent,
            "stable": bool(consistent and len(skipped_at) <= max_skipped and not late and (min(scales) >= 1.0 if scales else True))}


def _log(out_dir: Path, record: dict) -> None:
    with open(out_dir / RUNTIME_LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=float) + "\n")


def _session_number(out_dir: Path) -> int:
    path = out_dir / RUNTIME_LOG
    if not path.is_file():
        return 1
    return 1 + sum(1 for line in path.read_text(encoding="utf-8").splitlines() if '"session_start"' in line)


def summarize_runtime(out_dir: Path) -> dict:
    path = Path(out_dir) / RUNTIME_LOG
    summary = {"sessions": 0, "epochs": 0, "epoch_seconds": 0.0, "gpu_epoch_seconds": 0.0, "peak_cuda_gb": 0.0}
    if not path.is_file():
        return summary
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["event"] == "session_start":
            summary["sessions"] += 1
        elif row["event"] == "epoch":
            summary["epochs"] += 1
            summary["epoch_seconds"] += row["seconds"]
            if row["device"].startswith("cuda"):
                summary["gpu_epoch_seconds"] += row["seconds"]
            summary["peak_cuda_gb"] = max(summary["peak_cuda_gb"], row.get("cuda_peak_gb") or 0.0)
    return summary


def _durable_copy(out_dir: Path, export_dir: Optional[Path]) -> None:
    if export_dir is None:
        return
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    for name in (LAST, "last_best.pt", RUNTIME_LOG, UPDATE_LOG):
        source = out_dir / name
        if source.is_file():
            temp = export_dir / f".{name}.tmp"
            shutil.copy2(source, temp)
            temp.replace(export_dir / name)


def _check_compatible(saved: dict, current: dict) -> None:
    diffs = [k for k in IDENTITY_KEYS if saved.get(k) != current.get(k)]
    if int(saved.get("grad_accum_steps", 1)) != int(current.get("grad_accum_steps", 1)):  # not in IDENTITY_KEYS so that older checkpoints stay valid
        diffs.append("grad_accum_steps")
    if diffs:
        raise CheckpointIncompatibleError(f"Config changed since the checkpoint was written: {diffs}")


def _prepare(model, config):
    if config.get("deterministic", True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return model


# ----------------------------------------------------------------------------------------------- training
def train_weak_label(train_ds, val_ds, config: dict, out_dir: Path, *, fingerprint: str, train_ids: Sequence[str],
                     val_ids: Sequence[str], resume: bool = True, scaler=None, device: Optional[str] = None,
                     batch_hook: Optional[Callable[[int, int], None]] = None, max_seconds: Optional[float] = None,
                     export_dir: Optional[Path] = None, model_factory: Optional[Callable] = None,
                     stop_after_epochs: Optional[int] = None) -> dict:
    """Train one arm. Returns a summary dict; status is "complete", "paused_time_budget" or raises."""
    config = {**DEFAULT_CONFIG, **config}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_device = _device(config, device)
    seed_everything(int(config["seed"]))
    model = _prepare((model_factory or (lambda: build_model(config["arm"], config)))(), config).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    scheduler = _scheduler(optimizer, config)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True,
                        num_workers=int(config["num_workers"]), generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(config["batch_size"]), shuffle=False,
                            num_workers=int(config["num_workers"]))
    pos_weight = compute_pos_weight(train_ds.values, train_ds.masks, float(config["max_pos_weight"])).to(torch_device)
    reliability = reliability_vector(config).to(torch_device)
    use_amp = bool(config["amp"]) and torch_device.type in AMP_DEVICE_TYPES
    amp_dtype = torch.float16 if torch_device.type == "cuda" else torch.bfloat16
    # float16 only on CUDA: T4/P100 have no bf16 units, and emulated bf16 is ~50x slower and can silently change results
    if use_amp and scaler is None:
        scaler = torch.amp.GradScaler(torch_device.type, init_scale=float(config["amp_init_scale"]), growth_interval=int(config["amp_growth_interval"]))

    accum = int(config["grad_accum_steps"])
    if accum < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    start_epoch, step, best, history, resumed_from = 0, 0, None, [], None
    counters = {"updates_applied": 0, "updates_skipped": 0, "nonfinite_tensors": 0}
    last = out_dir / LAST
    if resume and last.is_file():
        peek = torch.load(last, map_location="cpu", weights_only=True)
        _check_compatible(peek.get("config", {}), config)
        if use_amp != (peek.get("scaler") is not None):
            raise CheckpointIncompatibleError("AMP scaler state in the checkpoint does not match this run's amp setting; "
                                              "resume with the same amp/device class or start a new work dir")
        state = load_checkpoint(last, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                expected_target_columns=TARGET_COLUMNS, expected_dataset_fingerprint=fingerprint,
                                device=torch_device)
        start_epoch, step, best = state.epoch, state.global_step, state.best_metric
        history = list(state.metrics.get("history", []))
        counters = {**counters, **state.metrics.get("counters", {})}
        _trim_update_log(out_dir / UPDATE_LOG, state.epoch)  # drop rows of an epoch that was lost with its checkpoint
        if state.loader_rng_state is not None:
            generator.set_state(state.loader_rng_state)
        resumed_from = state.epoch

    consecutive_skips = 0
    session = _session_number(out_dir)
    _log(out_dir, {"event": "session_start", "session": session, "device": str(torch_device),
                   "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   "resumed_from_epoch": resumed_from, "arm": config["arm"],
                   "pos_weight": [round(float(x), 3) for x in pos_weight.cpu()],
                   "trainable_entries": [int(x) for x in train_ds.masks.sum(axis=0)]})
    session_started = time.perf_counter()
    status, seen, invalid = "complete", 0, 0
    invalid_ids: set = set()
    invalid_path = out_dir / INVALID_LOG
    if invalid_path.is_file():
        invalid_ids = set(json.loads(invalid_path.read_text(encoding="utf-8")))

    patience, min_delta = int(config["early_stopping_patience"]), float(config["early_stopping_min_delta"])
    if early_stop_due(history, patience, min_delta):  # resuming a run that had already stopped early
        status, last_epoch = "early_stopped", start_epoch
    else:
        last_epoch = int(config["epochs"])
    for epoch in range(start_epoch, last_epoch):
        epoch_started = time.perf_counter()
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        losses, grad_norms, epoch_rows = [], [], []
        epoch_skipped, epoch_applied, epoch_bad_tensors = 0, 0, 0
        n_batches = len(loader)
        scale_start = scaler.get_scale() if scaler is not None and use_amp else None
        data_wait, epoch_images, wait_from = 0.0, 0, time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            data_wait += time.perf_counter() - wait_from  # time the GPU loop spent waiting for decoded batches
            if batch_hook is not None:
                batch_hook(epoch, step)
            group = min(accum, n_batches - (i // accum) * accum)  # batches summed into this update (the last group of an epoch may be smaller)
            with torch.autocast(device_type=torch_device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(batch["images"].to(torch_device), batch["mask"].to(torch_device), **_roi_inputs(batch, torch_device))
            loss = masked_bce(logits, batch["labels"].to(torch_device), batch["target_mask"].to(torch_device),
                              pos_weight, reliability)
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise TrainingAborted(f"non-finite training loss at update {step + 1}; stopping before a bad checkpoint is written")
            scaled = loss / group
            (scaler.scale(scaled) if scaler is not None and use_amp else scaled).backward()
            losses.append(loss_value)
            epoch_images += int(batch["mask"][batch["valid"]].sum())
            seen += len(batch["valid"])
            bad = ~batch["valid"]
            invalid += int(bad.sum())
            invalid_ids.update(u for u, b in zip(batch["study_uid"], bad.tolist()) if b)
            wait_from = time.perf_counter()
            if (i + 1) % accum and i + 1 != n_batches:
                continue  # keep accumulating
            # ---- one optimizer update. Non-finite gradients never reach optimizer.step(): the update is skipped, weights and Adam moments stay untouched.
            amp_on = scaler is not None and use_amp
            scale_before = scaler.get_scale() if amp_on else None
            if amp_on:
                scaler.unscale_(optimizer)
            norm_value = float(nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"])))  # pre-clip global norm (after unscaling under AMP)
            finite = math.isfinite(norm_value)
            bad_tensors = 0 if finite else sum(int(not torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if finite:
                if amp_on:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
                counters["updates_applied"] += 1
                epoch_applied += 1
                grad_norms.append(norm_value)
                consecutive_skips = 0
            else:
                counters["updates_skipped"] += 1
                counters["nonfinite_tensors"] += bad_tensors
                epoch_skipped += 1
                epoch_bad_tensors += bad_tensors
                consecutive_skips += 1
            if amp_on:
                scaler.update()  # an overflow lowers the scale; clean updates eventually raise it
            scale_after = scaler.get_scale() if amp_on else None
            optimizer.zero_grad(set_to_none=True)
            step += 1
            epoch_rows.append({"epoch": epoch + 1, "update": step, "micro_batches": group, "scale_before": scale_before, "scale_after": scale_after,
                               "grad_norm": norm_value if finite else None, "finite": finite, "skipped": not finite, "nonfinite_tensors": bad_tensors,
                               "applied_total": counters["updates_applied"], "skipped_total": counters["updates_skipped"],
                               "lr": float(optimizer.param_groups[0]["lr"]), "train_loss": loss_value})
            if consecutive_skips >= int(config["max_consecutive_skipped_updates"]):
                raise TrainingAborted(f"{consecutive_skips} consecutive optimizer updates were skipped for non-finite gradients (update {step}); training is unstable")
            if amp_on and scale_after < float(config["amp_min_scale"]):
                raise TrainingAborted(f"GradScaler scale fell to {scale_after} (< {config['amp_min_scale']}) at update {step}; persistent overflow, not start-up calibration")
        scheduler.step()
        if invalid_ids:  # attribution: which studies were skipped (full UIDs stay in the work dir, not in printed logs)
            invalid_path.write_text(json.dumps(sorted(invalid_ids)), encoding="utf-8")
        if seen and invalid / seen > float(config["max_invalid_fraction"]):
            raise TrainingAborted(f"{invalid}/{seen} training samples were unreadable (limit "
                                  f"{config['max_invalid_fraction']:.0%}); fix the data before continuing")

        model.eval()
        v_losses, v_labels, v_probs, v_masks = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["images"].to(torch_device), batch["mask"].to(torch_device), **_roi_inputs(batch, torch_device))
                v_losses.append(float(masked_bce(logits, batch["labels"].to(torch_device),
                                                 batch["target_mask"].to(torch_device), pos_weight, reliability).cpu()))
                v_labels.append(batch["labels"].numpy())
                v_probs.append(logits.float().sigmoid().cpu().numpy())
                v_masks.append(batch["target_mask"].numpy())
        if not np.isfinite(v_losses).all():
            raise TrainingAborted("non-finite validation loss; stopping before a bad checkpoint is written")
        auc = masked_auc(np.vstack(v_labels), np.vstack(v_probs), np.vstack(v_masks))
        defined = [a for a in auc.values() if a is not None]
        val_loss = float(np.mean(v_losses))
        probs_all = np.vstack(v_probs)
        mask_all = np.vstack(v_masks)
        grad_clip = float(config["grad_clip"])
        diagnostics = {  # health indicators only; nothing here feeds back into training
            "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
            "grad_norm_max": float(np.max(grad_norms)) if grad_norms else None,
            "grad_clipped_fraction": float(np.mean(np.asarray(grad_norms) > grad_clip)) if grad_norms else None,
            "nonfinite_grad_steps": epoch_skipped, "updates_applied": epoch_applied, "updates_skipped": epoch_skipped, "nonfinite_tensors": epoch_bad_tensors,
            "optimizer_updates_total": counters["updates_applied"], "global_step_updates": step, "scale_start": scale_start,
            "scale_end": scaler.get_scale() if scaler is not None and use_amp else None, "grad_accum_steps": accum,
            "val_prob_mean": {t: float(probs_all[mask_all[:, j] == 1, j].mean()) for j, t in enumerate(TARGET_COLUMNS) if (mask_all[:, j] == 1).any()},
            "val_prob_std": {t: float(probs_all[mask_all[:, j] == 1, j].std()) for j, t in enumerate(TARGET_COLUMNS) if (mask_all[:, j] == 1).any()}}
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **diagnostics, "val_loss": val_loss,
                  "val_macro_auc_weak": macro_auc(auc),
                  "val_auc_weak": auc, "lr": float(optimizer.param_groups[0]["lr"]), "invalid_samples": invalid, "invalid_studies": len(invalid_ids),
                  "train_images": epoch_images, "data_wait_seconds": round(data_wait, 3)}
        history.append(record)
        score = -val_loss  # higher is better, as in the existing trainers
        is_best = best is None or score > best
        best = score if is_best else best
        save_checkpoint(last, model=model, optimizer=optimizer, config=config, target_columns=TARGET_COLUMNS,
                        epoch=epoch + 1, global_step=step, train_study_ids=list(train_ids),
                        valid_study_ids=list(val_ids), dataset_fingerprint=fingerprint,
                        metrics={"history": history, "latest": record, "counters": dict(counters)}, best_metric=best, is_best=is_best,
                        scheduler=scheduler, scaler=scaler if use_amp else None, loader_generator=generator)
        _write_update_rows(out_dir / UPDATE_LOG, epoch_rows)
        _durable_copy(out_dir, export_dir)
        seconds = time.perf_counter() - epoch_started
        _log(out_dir, {"event": "epoch", "session": session, "epoch": epoch + 1, "seconds": seconds,
                       "data_wait_seconds": round(data_wait, 3), "train_images": epoch_images,
                       "device": str(torch_device), "val_loss": val_loss,
                       "cuda_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch_device.type == "cuda" else None})
        if early_stop_due(history, patience, min_delta):
            status = "early_stopped"
            break
        if stop_after_epochs is not None and epoch + 1 < int(config["epochs"]) and \
                (epoch + 1 - start_epoch) >= stop_after_epochs:
            status = "paused_epoch_limit"
            break
        if max_seconds is not None and epoch + 1 < int(config["epochs"]) and \
                (time.perf_counter() - session_started) + 1.2 * seconds > max_seconds:
            status = "paused_time_budget"
            break
    _log(out_dir, {"event": "session_end", "session": session, "status": status,
                   "seconds": time.perf_counter() - session_started})
    return {"status": status, "checkpoint": str(last), "final_epoch": history[-1]["epoch"] if history else 0,
            "resumed_from_epoch": resumed_from, "best_val_loss": -best if best is not None else None,
            "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"] if history else None,
            "history": history, "invalid_samples": invalid, "invalid_study_ids_file": str(invalid_path) if invalid_ids else None, "device": str(torch_device),
            "runtime": summarize_runtime(out_dir)}


# ----------------------------------------------------------------------------------------------- inference / gold
def predict_probabilities(model: nn.Module, dataset, device: torch.device, batch_size: int = 8, num_workers: int = 0):
    """Sigmoid probabilities [N, 12] in dataset order, plus study IDs. Deterministic: eval mode, no shuffling."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    uids, rows = [], []
    with torch.inference_mode():
        for batch in loader:
            rows.append(model(batch["images"].to(device), batch["mask"].to(device), **_roi_inputs(batch, device)).float().sigmoid().cpu().numpy())
            uids.extend(batch["study_uid"])
    return uids, np.vstack(rows)


def auc_report(labels: np.ndarray, probs: np.ndarray, *, rounds: int = 1000, seed: int = 0,
               headline_targets: Sequence[str] = ()) -> dict:
    """Per-target and macro ROC-AUC with percentile-bootstrap 95% CIs (studies are resampled)."""
    full = np.ones_like(labels)

    def evaluate(idx):
        auc = masked_auc(labels[idx], probs[idx], full[idx])
        defined = [v for v in auc.values() if v is not None]
        head = [auc[t] for t in headline_targets if auc.get(t) is not None]
        return auc, (np.mean(defined) if defined else None), (np.mean(head) if head else None)

    auc, macro, head = evaluate(np.arange(len(labels)))
    rng = np.random.default_rng(seed)
    draws = {t: [] for t in TARGET_COLUMNS}
    macros, heads = [], []
    for _ in range(rounds):
        idx = rng.integers(0, len(labels), len(labels))
        a, m, h = evaluate(idx)
        for t, v in a.items():
            if v is not None:
                draws[t].append(v)
        if m is not None:
            macros.append(m)
        if h is not None:
            heads.append(h)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None
    return {"n_studies": int(len(labels)), "per_target_auc": auc, "per_target_ci95": {t: ci(v) for t, v in draws.items()},
            "macro_auc": None if macro is None else float(macro), "macro_ci95": ci(macros),
            "headline_targets": list(headline_targets), "headline_macro_auc": None if head is None else float(head),
            "headline_ci95": ci(heads), "bootstrap_rounds": rounds}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_gold(checkpoint: Path, metadata, gold_frame, *, ledger_path: Optional[Path], force: bool = False,
                  device: Optional[str] = None, batch_size: int = 8, rounds: int = 1000,
                  headline_targets: Sequence[str] = ("ACL", "Medial Meniscus", "Lateral Meniscus"),
                  dataset=None) -> dict:
    """Score a checkpoint on the gold studies. Refuses if any gold study was in training/validation, and allows one
    look per checkpoint file hash unless forced (forced looks are recorded as such)."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    gold_ids = list(gold_frame["StudyInstanceUID"].astype(str))
    assert_no_leakage(state["train_study_ids"], state["valid_study_ids"], gold_ids=gold_ids)
    digest = sha256_file(checkpoint)
    spent = False
    if ledger_path is not None:
        ledger = Path(ledger_path)
        spent = ledger.is_file() and any(json.loads(l)["checkpoint_sha256"] == digest
                                         for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip())
        if spent and not force:
            raise GoldAlreadyScoredError("This checkpoint was already scored on the gold studies; pick checkpoints "
                                         "and epochs from weak-label validation only, or pass force=True.")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"checkpoint_sha256": digest, "arm": state["config"].get("arm"),
                                     "forced": bool(spent and force), "n_gold": len(gold_ids)}) + "\n")
    config = state["config"]
    torch_device = _device(config, device)
    model = build_model(config["arm"], {**config, "pretrained_weights": None}).to(torch_device)
    model.load_state_dict(state["model"])
    dataset = dataset or RSNAKneeDicomDataset(metadata, gold_frame, config, split="train")
    uids, probs = predict_probabilities(model, dataset, torch_device, batch_size)
    order = {u: i for i, u in enumerate(uids)}
    labels = gold_frame.set_index(gold_frame["StudyInstanceUID"].astype(str)).loc[uids, list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)
    report = auc_report(labels, probs, rounds=rounds, headline_targets=headline_targets)
    report.update({"arm": config["arm"], "checkpoint_sha256": digest, "epoch": int(state["epoch"]),
                   "dataset_fingerprint": state["dataset_fingerprint"], "forced": bool(spent and force)})
    return report
