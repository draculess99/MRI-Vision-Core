"""Resumable MRNet training, built alongside (not replacing) mri_core/cnn_train.py.

cnn_train.py's train_one_seed() already writes a complete checkpoint (model, optimizer,
RNG states) after every epoch, but train_one_seed() always calls
`directory.mkdir(parents=True, exist_ok=False)` and never looks for an existing checkpoint
to resume from -- interrupting and re-running it with the same directory simply fails.
This module reuses MRNetDataset/MRNetCNN/manifest_fingerprint from the existing, tested
cnn_train.py/cnn_dataset.py and adds: (a) resume-on-restart via mri_core.rsna_checkpoint's
generic save/load, and (b) a SIGINT/SIGTERM handler that stops cleanly at the next batch
boundary instead of being killed mid-epoch.

Limits of the interruption handler (stated explicitly, per design requirement): SIGINT and
SIGTERM can be caught and will trigger a clean stop before the next batch starts. A forced
kill (SIGKILL / Windows `taskkill /F`) or a power loss cannot be intercepted by any Python
signal handler and may lose progress made since the last fully-completed epoch's checkpoint.
The checkpoint *file itself* remains safe either way, because mri_core.rsna_checkpoint.
save_checkpoint always writes to a temp file and renames atomically.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cnn_dataset import MRNetDataset, training_class_weights
from .cnn_model import MRNetCNN
from .cnn_train import manifest_fingerprint, seed_everything, seed_worker
from .labels import TARGETS
from .rsna_checkpoint import CheckpointMalformedError, load_checkpoint, save_checkpoint

LAST_CHECKPOINT_NAME = "last.pt"


class TrainingInterrupted(Exception):
    """Raised when a caught SIGINT/SIGTERM stops training cleanly before all epochs ran."""


class GracefulInterruptHandler:
    """Catches SIGINT and SIGTERM for the duration of the `with` block.

    Cannot catch SIGKILL or power loss -- see module docstring.
    """

    def __init__(self):
        self.requested = False
        self._previous = {}

    def __enter__(self):
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is not None:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
        return self

    def _handle(self, signum, frame):
        self.requested = True

    def __exit__(self, exc_type, exc_value, traceback):
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        return False


def train_mrnet_resumable(
    manifest,
    config: dict,
    checkpoint_dir: Path,
    *,
    resume: bool = True,
    device: str = "cpu",
    install_signal_handler: bool = True,
):
    """Train MRNetCNN with a fully resumable checkpoint written after every completed epoch.

    A caught SIGINT/SIGTERM stops cleanly before the next batch and raises
    TrainingInterrupted; the checkpoint from the last fully-completed epoch is left intact
    (no partial-epoch state is ever written).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / LAST_CHECKPOINT_NAME
    torch_device = torch.device(device)

    seed = int(config["seed"]) if "seed" in config else int(config["seeds"][0])
    seed_everything(seed)

    fingerprint = manifest_fingerprint(manifest.rows)

    training = MRNetDataset(manifest, "train", config, augment=True, seed=seed)
    validation = MRNetDataset(manifest, "valid", config, augment=False, seed=seed)

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], worker_init_fn=seed_worker,
        generator=loader_generator, persistent_workers=False,
    )
    valid_loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    model = MRNetCNN(len(config["planes"]), config["dropout"], num_targets=len(TARGETS)).to(torch_device)
    weights = training_class_weights(manifest.rows).to(torch_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    train_ids = sorted(row["exam_id"] for row in manifest.rows if row["split"] == "train")
    valid_ids = sorted(row["exam_id"] for row in manifest.rows if row["split"] == "valid")

    start_epoch, global_step, best_metric, history = 0, 0, None, []
    resumed_from = None

    if resume and checkpoint_path.is_file():
        try:
            state = load_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer,
                expected_target_columns=list(TARGETS), expected_dataset_fingerprint=fingerprint,
                device=torch_device,
            )
        except CheckpointMalformedError:
            state = None
        if state is not None:
            start_epoch = state.epoch
            global_step = state.global_step
            best_metric = state.best_metric
            history = list(state.metrics.get("history", []))
            if state.loader_rng_state is not None:
                loader_generator.set_state(state.loader_rng_state)
            resumed_from = state.epoch

    target_epochs = int(config["epochs"])
    started = time.perf_counter()
    interrupted = False

    with GracefulInterruptHandler() if install_signal_handler else _NullInterruptHandler() as interrupt:
        for epoch in range(start_epoch, target_epochs):
            model.train()
            training.set_epoch(epoch)
            total_loss, example_count = 0.0, 0
            for batch in train_loader:
                if interrupt.requested:
                    interrupted = True
                    break
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["images"].to(torch_device), batch["mask"].to(torch_device))
                loss = criterion(logits, batch["labels"].to(torch_device))
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(batch["exam_id"])
                example_count += len(batch["exam_id"])
                global_step += 1

            if interrupted:
                break  # last.pt still holds the last fully-completed epoch; nothing partial is written

            model.eval()
            valid_losses, valid_labels, valid_scores = [], [], []
            with torch.inference_mode():
                for batch in valid_loader:
                    logits = model(batch["images"].to(torch_device), batch["mask"].to(torch_device))
                    valid_losses.append(float(criterion(logits, batch["labels"].to(torch_device)).cpu()) * len(batch["exam_id"]))
                    valid_labels.append(batch["labels"].numpy())
                    valid_scores.append(logits.sigmoid().cpu().numpy())
            labels_array = np.vstack(valid_labels)
            scores_array = np.vstack(valid_scores)

            from sklearn.metrics import roc_auc_score

            per_target_auc = {}
            for index, target in enumerate(TARGETS):
                column = labels_array[:, index]
                per_target_auc[target] = (
                    float(roc_auc_score(column, scores_array[:, index])) if len(np.unique(column)) >= 2 else None
                )
            defined = [v for v in per_target_auc.values() if v is not None]
            macro_auc = float(np.mean(defined)) if defined else None

            record = {
                "epoch": epoch + 1,
                "training_loss": total_loss / example_count,
                "validation_loss": sum(valid_losses) / len(validation),
                "per_target_auc": per_target_auc,
                "macro_auc": macro_auc,
            }
            history.append(record)

            score = macro_auc if macro_auc is not None else -np.inf
            is_best = best_metric is None or score > best_metric
            if is_best:
                best_metric = score

            save_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer, config=config,
                target_columns=list(TARGETS), epoch=epoch + 1, global_step=global_step,
                train_study_ids=train_ids, valid_study_ids=valid_ids,
                dataset_fingerprint=fingerprint, metrics={"history": history, "latest": record},
                best_metric=best_metric, is_best=is_best, loader_generator=loader_generator,
            )

    if interrupted:
        raise TrainingInterrupted(
            f"Stopped cleanly by signal after epoch {history[-1]['epoch'] if history else start_epoch}; "
            f"last.pt reflects that last fully-completed epoch. Re-run with resume=True to continue."
        )

    return {
        "checkpoint": str(checkpoint_path),
        "device": str(torch_device),
        "train_examples": len(training),
        "validation_examples": len(validation),
        "seconds": time.perf_counter() - started,
        "history": history,
        "resumed_from_epoch": resumed_from,
        "final_epoch": target_epochs,
        "best_metric": best_metric,
        "dataset_fingerprint": fingerprint,
    }


class _NullInterruptHandler:
    """No-op stand-in when install_signal_handler=False (e.g. inside a worker thread,
    where signal.signal() is not permitted)."""

    requested = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
