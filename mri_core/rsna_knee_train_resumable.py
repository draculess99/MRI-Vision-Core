"""Resumable RSNA training loop.

This sits alongside, and does not modify, mri_core/rsna_knee_train.py's train_rsna():
that function's checkpoint format (model/config/target_columns/epoch/validation) keeps
working for run_inference()/run_ensemble_inference() exactly as before. This module adds
a separate, fully resumable training path (optimizer/scheduler/scaler/RNG/dataset
fingerprint/exact study IDs) built from the same dataset, model, and CV-split primitives.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .rsna_checkpoint import (
    CheckpointMalformedError,
    load_checkpoint,
    labeled_frame_fingerprint,
    save_checkpoint,
)
from .rsna_knee_dataset import PLANE_NAMES, TARGET_COLUMNS, RSNAKneeMetadata
from .rsna_knee_model import RSNAKneeCNN
from .rsna_knee_train import (
    RSNAKneeDicomDataset,
    _macro_loss,
    calculate_auc,
    seed_everything,
    split_labeled_studies,
    split_labeled_studies_cv,
)

LAST_CHECKPOINT_NAME = "last.pt"


def _resolve_split(metadata: RSNAKneeMetadata, config: dict):
    if "cv_folds" in config:
        return split_labeled_studies_cv(
            metadata, int(config["cv_folds"]), int(config.get("cv_fold", 0)), int(config["seed"])
        )
    return split_labeled_studies(metadata, float(config["validation_fraction"]), int(config["seed"]))


def train_rsna_resumable(
    metadata: RSNAKneeMetadata,
    config: dict,
    checkpoint_dir: Path,
    *,
    resume: bool = True,
    scheduler_factory=None,
    scaler=None,
    device: Optional[str] = None,
    initial_model=None,
):
    """Train RSNAKneeCNN with a fully resumable checkpoint written after every epoch.

    scheduler_factory, if given, is called as scheduler_factory(optimizer) once at the
    start of a fresh run (not on resume, where the scheduler's own state is restored).

    initial_model, if given, is used instead of a freshly-constructed RSNAKneeCNN for a
    *fresh* run (e.g. a model produced by rsna_mrnet_transfer.transfer_mrnet_encoder_to_rsna).
    Ignored when resuming from an existing checkpoint, since the checkpoint's own weights
    take precedence -- this only controls how a brand-new run's weights start out.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / LAST_CHECKPOINT_NAME

    seed_everything(int(config["seed"]))

    train_frame, valid_frame = _resolve_split(metadata, config)
    dataset_fingerprint = labeled_frame_fingerprint(metadata.complete_label_train)

    device_name = config.get("device", "auto") if device is None else device
    torch_device = (
        torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )

    train_data = RSNAKneeDicomDataset(metadata, train_frame, config, "train")
    valid_data = RSNAKneeDicomDataset(metadata, valid_frame, config, "train")

    loader_generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_data, batch_size=int(config["batch_size"]), shuffle=True,
        num_workers=int(config["num_workers"]), generator=loader_generator,
    )
    valid_loader = DataLoader(
        valid_data, batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]),
    )

    model = (initial_model if initial_model is not None
              else RSNAKneeCNN(len(config.get("planes", PLANE_NAMES)), float(config["dropout"]))).to(torch_device)
    labels = train_frame[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)
    positive = labels.sum(axis=0)
    pos_weight = torch.tensor(
        (len(labels) - positive) / np.maximum(positive, 1), dtype=torch.float32, device=torch_device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None

    start_epoch, global_step, best_metric, history = 0, 0, None, []
    resumed_from = None

    if resume and checkpoint_path.is_file():
        try:
            state = load_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                expected_target_columns=TARGET_COLUMNS,
                expected_dataset_fingerprint=dataset_fingerprint,
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

    started = time.perf_counter()
    target_epochs = int(config["epochs"])

    for epoch in range(start_epoch, target_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["images"].to(torch_device), batch["mask"].to(torch_device))
            loss = criterion(logits, batch["labels"].to(torch_device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
        if scheduler is not None:
            scheduler.step()

        model.eval()
        valid_losses, valid_labels, valid_scores = [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                logits = model(batch["images"].to(torch_device), batch["mask"].to(torch_device))
                valid_losses.append(float(criterion(logits, batch["labels"].to(torch_device)).cpu()))
                valid_labels.append(batch["labels"].numpy())
                valid_scores.append(logits.sigmoid().cpu().numpy())
        labels_array, scores_array = np.vstack(valid_labels), np.vstack(valid_scores)
        auc = calculate_auc(labels_array, scores_array)
        score = auc["macro_roc_auc"] if auc["macro_roc_auc"] is not None else -np.inf
        record = {
            "epoch": epoch + 1,
            "loss": _macro_loss(losses),
            "validation_loss": _macro_loss(valid_losses),
            **auc,
        }
        history.append(record)

        is_best = best_metric is None or score > best_metric
        if is_best:
            best_metric = score

        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            target_columns=TARGET_COLUMNS,
            epoch=epoch + 1,
            global_step=global_step,
            train_study_ids=train_frame["StudyInstanceUID"].tolist(),
            valid_study_ids=valid_frame["StudyInstanceUID"].tolist(),
            dataset_fingerprint=dataset_fingerprint,
            metrics={"history": history, "latest": record},
            best_metric=best_metric,
            is_best=is_best,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=loader_generator,
        )

    return {
        "checkpoint": str(checkpoint_path),
        "device": str(torch_device),
        "train_studies": len(train_frame),
        "validation_studies": len(valid_frame),
        "seconds": time.perf_counter() - started,
        "history": history,
        "resumed_from_epoch": resumed_from,
        "final_epoch": target_epochs,
        "best_metric": best_metric,
        "dataset_fingerprint": dataset_fingerprint,
    }
