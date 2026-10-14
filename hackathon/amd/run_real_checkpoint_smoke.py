"""Smoke test: one real local DICOM study through a real externally-trained checkpoint.

Uses mri_core.rsna_integration only for study/series *discovery* (which studies and planes
are actually downloaded locally) -- not for model loading or preprocessing, since the
checkpoint here is not mri_core's own RSNAKneeCNN. See hackathon/amd/kaggle_mil_bridge.py
for why a separate model/preprocessing path is required.

Fails with a non-zero exit code rather than writing an invented result if the checkpoint,
study data, or decode step is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from mri_core.rsna_integration import (
    RSNADiscoveryError,
    RSNAStudyNotAvailable,
    discover_available_studies,
    get_available_planes,
    load_rsna_metadata_safe,
)
from mri_core.rsna_knee_dataset import select_series

from .kaggle_mil_bridge import BridgeError, load_bridge_checkpoint, run_study_inference

NON_DIAGNOSTIC_NOTICE = (
    "Research prototype output, not a diagnostic device. This checkpoint is an externally "
    "trained Kaggle baseline bridged into this adapter; it has not been evaluated, calibrated, "
    "or validated by this project. Probabilities are raw, uncalibrated sigmoid outputs. "
    "Nothing here is a clinical finding."
)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pick_study_and_series(data_root: Path, study_uid: str | None) -> tuple[str, str, str]:
    metadata = load_rsna_metadata_safe(data_root)
    if study_uid is None:
        available = discover_available_studies(data_root, metadata)
        if not available:
            raise RSNAStudyNotAvailable(f"No fully downloaded labeled study under {data_root}")
        study_uid = available[0]

    planes = get_available_planes(data_root, metadata, study_uid)
    if not planes:
        raise RSNAStudyNotAvailable(f"No downloaded series for study {study_uid} under {data_root}")
    plane = "Sagittal" if "Sagittal" in planes else planes[0]

    series_rows = metadata.series_for(study_uid, split="train")
    plane_candidates = series_rows[series_rows["Anatomical_Plane"].astype(str) == plane]
    series_uid = select_series(plane_candidates)
    return study_uid, series_uid, plane


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="RSNA knee data dir (train.csv, raw/train_series/...)")
    parser.add_argument("--checkpoint", required=True, help="Path to a foldN-knee-mil-best.ckpt file")
    parser.add_argument("--study-uid", help="StudyInstanceUID; default: first fully downloaded labeled study")
    parser.add_argument("--out", default="outputs/amd/real_checkpoint_prediction.json")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    checkpoint_path = Path(args.checkpoint)

    try:
        study_uid, series_uid, plane = pick_study_and_series(data_root, args.study_uid)
    except (RSNADiscoveryError, RSNAStudyNotAvailable) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    series_dir = data_root / "raw" / "train_series" / study_uid / series_uid

    try:
        model, model_provenance = load_bridge_checkpoint(checkpoint_path)
    except BridgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    start = time.monotonic()
    try:
        inference = run_study_inference(model, series_dir)
    except BridgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    latency_seconds = time.monotonic() - start

    result = {
        "prediction": {
            "study_uid": study_uid,
            "series_uid": series_uid,
            "plane": plane,
            "n_slices_sampled": inference["n_slices"],
            "image_size": inference["image_size"],
            "probabilities": inference["probabilities"],
            "latency_seconds": latency_seconds,
            "device": "cpu",
        },
        "provenance": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_of_file(checkpoint_path),
            **model_provenance,
            "data_root": str(data_root),
        },
        "non_diagnostic_notice": NON_DIAGNOSTIC_NOTICE,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
