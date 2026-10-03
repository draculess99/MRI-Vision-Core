"""Write an RSNA Kaggle submission using one or more trained checkpoints.

Supports single-checkpoint inference or averaging predictions across multiple
checkpoints. The output schema is forced to match sample_submission.csv exactly.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from .rsna_knee_dataset import (
    SUBMISSION_COLUMNS,
    TARGET_COLUMNS,
    load_rsna_metadata,
)
from .rsna_knee_model import RSNAKneeCNN
from .rsna_knee_train import RSNAKneeDicomDataset


def _load_checkpoint(path: Path):
    """Load and validate one checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} did not contain a dictionary")

    required = {"model", "config", "target_columns"}
    missing = required.difference(checkpoint.keys())
    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing required fields: {sorted(missing)}"
        )

    if list(checkpoint["target_columns"]) != list(TARGET_COLUMNS):
        raise ValueError(
            f"Checkpoint {path} target order does not match RSNA schema"
        )

    if not isinstance(checkpoint["config"], dict):
        raise ValueError(f"Checkpoint {path} config is not a dictionary")

    return checkpoint


def _build_model(checkpoint, selected_device):
    """Construct a model matching a checkpoint and load its weights."""
    config = checkpoint["config"]

    if "planes" in config:
        num_planes = len(config["planes"])
    else:
        num_planes = int(config.get("num_planes", 3))

    dropout = float(config["dropout"])

    model = RSNAKneeCNN(
        num_planes=num_planes,
        dropout=dropout,
    ).to(selected_device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def write_submission(
    data_dir,
    checkpoints,
    output,
    batch_size=8,
    num_workers=0,
    device="auto",
    dicom_root=None,
):
    """Generate an RSNA submission CSV.

    Args:
        data_dir:
            Directory containing train.csv, test.csv, train_series.csv,
            test_series.csv, and sample_submission.csv.

        checkpoints:
            One checkpoint path or an iterable of checkpoint paths.
            Predictions are averaged across all supplied checkpoints.

        output:
            Output CSV path.

        batch_size:
            Number of studies per inference batch.

        num_workers:
            PyTorch DataLoader worker count.

        device:
            "auto", "cpu", "cuda", etc.

        dicom_root:
            Optional root containing train_series/ and test_series/.

    Returns:
        pandas.DataFrame containing the submission.
    """

    data_dir = Path(data_dir)
    output = Path(output)

    if isinstance(checkpoints, (str, Path)):
        checkpoint_paths = [Path(checkpoints)]
    else:
        checkpoint_paths = [Path(p) for p in checkpoints]

    if not checkpoint_paths:
        raise ValueError("At least one checkpoint must be supplied")

    metadata = load_rsna_metadata(data_dir, dicom_root)

    loaded_checkpoints = [
        _load_checkpoint(path)
        for path in checkpoint_paths
    ]

    if device == "auto":
        selected_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        selected_device = torch.device(device)

    models = [
        _build_model(checkpoint, selected_device)
        for checkpoint in loaded_checkpoints
    ]

    # All checkpoints must use compatible preprocessing.
    base_config = dict(loaded_checkpoints[0]["config"])

    for index, checkpoint in enumerate(loaded_checkpoints[1:], start=1):
        config = checkpoint["config"]

        keys_to_match = (
            "image_size",
            "slices_per_series",
            "slices_per_plane",
            "planes",
            "num_planes",
        )

        for key in keys_to_match:
            if key in base_config or key in config:
                if base_config.get(key) != config.get(key):
                    raise ValueError(
                        f"Checkpoint preprocessing mismatch for '{key}' "
                        f"between checkpoint 0 and checkpoint {index}"
                    )

    # Dataset expects slices_per_series.
    dataset_config = dict(base_config)

    if (
        "slices_per_series" not in dataset_config
        and "slices_per_plane" in dataset_config
    ):
        dataset_config["slices_per_series"] = dataset_config["slices_per_plane"]

    dataset = RSNAKneeDicomDataset(
        metadata,
        metadata.test,
        dataset_config,
        split="test",
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    predictions = {}

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(selected_device)
            mask = batch["mask"].to(selected_device)

            model_scores = []

            for model in models:
                logits = model(images, mask)
                probabilities = logits.sigmoid()
                model_scores.append(probabilities)

            ensemble_scores = torch.stack(
                model_scores,
                dim=0,
            ).mean(dim=0)

            ensemble_scores = ensemble_scores.cpu().numpy()

            for uid, row in zip(
                batch["study_uid"],
                ensemble_scores,
            ):
                predictions[str(uid)] = row

    template = metadata.sample_submission.copy()

    if tuple(template.columns) != tuple(SUBMISSION_COLUMNS):
        raise ValueError(
            "Submission template column order changed"
        )

    result = pd.DataFrame(
        {
            "StudyInstanceUID":
                template["StudyInstanceUID"].astype("string")
        }
    )

    missing_uids = [
        uid
        for uid in result["StudyInstanceUID"].astype(str)
        if uid not in predictions
    ]

    if missing_uids:
        raise ValueError(
            f"Missing predictions for {len(missing_uids)} test studies. "
            f"First missing UID: {missing_uids[0]}"
        )

    for index, target in enumerate(TARGET_COLUMNS):
        result[target] = (
            result["StudyInstanceUID"]
            .astype(str)
            .map(lambda uid: float(predictions[uid][index]))
        )

    # Final sanity checks.
    if tuple(result.columns) != tuple(SUBMISSION_COLUMNS):
        raise ValueError(
            "Generated submission column order does not match RSNA schema"
        )

    probability_values = result[list(TARGET_COLUMNS)]

    if probability_values.isnull().any().any():
        raise ValueError(
            "Generated submission contains missing probability values"
        )

    if (
        probability_values.lt(0.0).any().any()
        or probability_values.gt(1.0).any().any()
    ):
        raise ValueError(
            "Generated submission contains probabilities outside [0, 1]"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output,
        index=False,
    )

    return result