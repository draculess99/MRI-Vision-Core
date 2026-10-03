"""Kaggle entry point for RSNA Knee Abnormality Detection inference.

Runs one or more trained RSNA checkpoints against the competition test set,
averages their predictions, validates the result, and writes submission.csv.

Designed to work both:

1. Locally, with explicit paths.
2. Inside a Kaggle notebook/runtime.

Example:

    python scripts/kaggle_rsna_submit.py \
        --data-dir /kaggle/input/rsna-knee-abnormality-detection \
        --dicom-root /kaggle/input/rsna-knee-abnormality-detection \
        --checkpoint /kaggle/input/mri-core-rsna-checkpoints/fold0_checkpoint_best.pt \
        --checkpoint /kaggle/input/mri-core-rsna-checkpoints/fold1_checkpoint_best.pt \
        --output /kaggle/working/submission.csv \
        --device auto
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Make project root importable when this script is executed directly.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from mri_core.rsna_knee_dataset import (  # noqa: E402
    SUBMISSION_COLUMNS,
    TARGET_COLUMNS,
)
from mri_core.rsna_knee_submit import write_submission  # noqa: E402


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_data_dir(data_dir: Path) -> None:
    """Validate required RSNA competition metadata files."""

    required_files = (
        "test.csv",
        "test_series.csv",
        "sample_submission.csv",
    )

    missing = [
        name
        for name in required_files
        if not (data_dir / name).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"RSNA data directory {data_dir} is missing required files: "
            f"{missing}"
        )


def validate_dicom_root(dicom_root: Path) -> None:
    """Validate that the test-series directory is available."""

    test_series_dir = dicom_root / "test_series"

    if not test_series_dir.is_dir():
        raise FileNotFoundError(
            f"RSNA test_series directory not found: {test_series_dir}"
        )


def validate_checkpoints(checkpoints: list[Path]) -> None:
    """Ensure all supplied checkpoint files exist."""

    if not checkpoints:
        raise ValueError("At least one checkpoint must be supplied")

    missing = [
        path
        for path in checkpoints
        if not path.is_file()
    ]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)

        raise FileNotFoundError(
            "One or more checkpoint files were not found:\n"
            f"{formatted}"
        )


def validate_submission(
    submission: pd.DataFrame,
    template_path: Path,
) -> None:
    """Perform final structural and probability validation."""

    template = pd.read_csv(template_path)

    if tuple(submission.columns) != tuple(SUBMISSION_COLUMNS):
        raise ValueError(
            "Generated submission columns do not match expected "
            "SUBMISSION_COLUMNS"
        )

    if tuple(submission.columns) != tuple(template.columns):
        raise ValueError(
            "Generated submission columns do not exactly match "
            "sample_submission.csv"
        )

    if len(submission) != len(template):
        raise ValueError(
            f"Submission row count {len(submission)} does not match "
            f"sample submission row count {len(template)}"
        )

    expected_uids = template["StudyInstanceUID"].astype(str).tolist()
    actual_uids = submission["StudyInstanceUID"].astype(str).tolist()

    if actual_uids != expected_uids:
        raise ValueError(
            "StudyInstanceUID ordering does not match sample_submission.csv"
        )

    probabilities = submission[list(TARGET_COLUMNS)]

    if probabilities.isnull().any().any():
        raise ValueError(
            "Submission contains NaN probability values"
        )

    minimum = float(probabilities.min().min())
    maximum = float(probabilities.max().max())

    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"Probability range invalid: min={minimum}, max={maximum}"
        )

    if submission["StudyInstanceUID"].duplicated().any():
        raise ValueError(
            "Submission contains duplicate StudyInstanceUID values"
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_environment(device: str) -> None:
    """Print useful runtime information."""

    print("=" * 72)
    print("RSNA Knee Abnormality Detection - Kaggle Inference")
    print("=" * 72)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA devices    : {torch.cuda.device_count()}")
        print(f"GPU             : {torch.cuda.get_device_name(0)}")

    print(f"Requested device: {device}")
    print()


def print_submission_summary(
    submission: pd.DataFrame,
    output: Path,
    elapsed_seconds: float,
    checkpoint_count: int,
) -> None:
    """Print a concise final submission report."""

    probabilities = submission[list(TARGET_COLUMNS)]

    print()
    print("=" * 72)
    print("SUBMISSION COMPLETE")
    print("=" * 72)

    print(f"Checkpoints : {checkpoint_count}")
    print(f"Studies     : {len(submission)}")
    print(f"Targets     : {len(TARGET_COLUMNS)}")
    print(f"Shape       : {submission.shape}")
    print(
        f"Probability range: "
        f"{probabilities.min().min():.6f} "
        f"to {probabilities.max().max():.6f}"
    )
    print(f"Elapsed     : {elapsed_seconds:.1f} seconds")
    print(f"Output      : {output.resolve()}")

    print()
    print("First submission rows:")
    print(submission.head().to_string(index=False))

    print()
    print("SUCCESS: submission.csv passed structural validation.")


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the full Kaggle inference pipeline."""

    data_dir = Path(args.data_dir)
    dicom_root = (
        Path(args.dicom_root)
        if args.dicom_root
        else data_dir
    )
    output = Path(args.output)
    checkpoints = [
        Path(path)
        for path in args.checkpoint
    ]

    print_environment(args.device)

    print(f"Data directory : {data_dir}")
    print(f"DICOM root     : {dicom_root}")
    print(f"Output         : {output}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Workers        : {args.num_workers}")

    print()
    print("Checkpoints:")

    for index, checkpoint in enumerate(checkpoints):
        print(f"  [{index}] {checkpoint}")

    print()

    print("Validating input paths...")

    validate_data_dir(data_dir)
    validate_dicom_root(dicom_root)
    validate_checkpoints(checkpoints)

    print("Input validation OK.")
    print()

    start = time.perf_counter()

    print(
        f"Running ensemble inference with "
        f"{len(checkpoints)} checkpoint(s)..."
    )

    submission = write_submission(
        data_dir=data_dir,
        checkpoints=checkpoints,
        output=output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dicom_root=dicom_root,
    )

    elapsed = time.perf_counter() - start

    print()
    print("Validating generated submission...")

    validate_submission(
        submission,
        data_dir / "sample_submission.csv",
    )

    print_submission_summary(
        submission=submission,
        output=output,
        elapsed_seconds=elapsed,
        checkpoint_count=len(checkpoints),
    )

    return submission


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an RSNA Knee Abnormality Detection Kaggle submission "
            "using one or more trained checkpoints."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Directory containing test.csv, test_series.csv, "
            "sample_submission.csv, etc."
        ),
    )

    parser.add_argument(
        "--dicom-root",
        default=None,
        help=(
            "Directory containing test_series/. "
            "Defaults to --data-dir."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help=(
            "Checkpoint file. Repeat --checkpoint for ensemble inference."
        ),
    )

    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output CSV path. Default: submission.csv",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size. Default: 8",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers. Default: 0",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            'Torch device, e.g. "auto", "cpu", or "cuda". '
            'Default: "auto".'
        ),
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print()
        print("=" * 72, file=sys.stderr)
        print("RSNA SUBMISSION FAILED", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())