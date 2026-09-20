"""Metadata and optional DICOM access for the RSNA Knee Abnormality dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TARGET_COLUMNS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
)
SERIES_COLUMNS = ("StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
                  "Fat_Suppression", "Anatomical_Plane")
SUBMISSION_COLUMNS = ("StudyInstanceUID", *TARGET_COLUMNS)
PLANE_NAMES = ("Axial", "Coronal", "Sagittal")


def _read_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing RSNA metadata file: {path}")
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": "string", "SeriesInstanceUID": "string"})
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if frame["StudyInstanceUID"].isna().any() or (frame["StudyInstanceUID"].astype(str).str.len() == 0).any():
        raise ValueError(f"{path.name}: StudyInstanceUID contains missing values")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype("string")
    return frame


@dataclass(frozen=True)
class RSNAKneeMetadata:
    data_dir: Path
    train: pd.DataFrame
    train_series: pd.DataFrame
    test: pd.DataFrame
    test_series: pd.DataFrame
    sample_submission: pd.DataFrame

    def series_for(self, study_uid: str, split: str = "train") -> pd.DataFrame:
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        frame = self.train_series if split == "train" else self.test_series
        return frame[frame["StudyInstanceUID"] == str(study_uid)].copy()

    def study_series_map(self, split: str = "train") -> dict[str, tuple[dict, ...]]:
        frame = self.train_series if split == "train" else self.test_series
        return {str(uid): tuple(group.to_dict("records"))
                for uid, group in frame.groupby("StudyInstanceUID", sort=False)}

    def target_vector(self, study_uid: str) -> np.ndarray:
        matches = self.train[self.train["StudyInstanceUID"] == str(study_uid)]
        if len(matches) != 1:
            raise KeyError(f"Expected one training row for StudyInstanceUID={study_uid!r}")
        return matches.iloc[0][list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)

    @property
    def complete_label_train(self) -> pd.DataFrame:
        return self.train.dropna(subset=list(TARGET_COLUMNS)).copy()

    def dicom_series_dir(self, study_uid: str, series_uid: str, split: str = "train") -> Path:
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        return self.data_dir / f"{split}_series" / str(study_uid) / str(series_uid)


def load_rsna_metadata(data_dir: Path | str) -> RSNAKneeMetadata:
    root = Path(data_dir)
    train = _read_csv(root / "train.csv", ("StudyInstanceUID", "Report", *TARGET_COLUMNS))
    train_series = _read_csv(root / "train_series.csv", SERIES_COLUMNS)
    test = _read_csv(root / "test.csv", ("StudyInstanceUID",))
    test_series = _read_csv(root / "test_series.csv", SERIES_COLUMNS)
    sample = _read_csv(root / "sample_submission.csv", SUBMISSION_COLUMNS)
    if tuple(sample.columns) != SUBMISSION_COLUMNS:
        raise ValueError("sample_submission.csv columns are not in the required order")
    for frame, name in ((train_series, "train_series"), (test_series, "test_series")):
        if frame["SeriesInstanceUID"].isna().any():
            raise ValueError(f"{name}: SeriesInstanceUID contains missing values")
        unknown = set(frame["Anatomical_Plane"].dropna().astype(str)) - set(PLANE_NAMES)
        if unknown:
            raise ValueError(f"{name}: unknown Anatomical_Plane values {sorted(unknown)}")
    return RSNAKneeMetadata(root, train, train_series, test, test_series, sample)


def split_labeled_studies(metadata: RSNAKneeMetadata, validation_fraction: float, seed: int):
    """Return disjoint study-level train/validation dataframes with complete labels only."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    labeled = metadata.complete_label_train
    if len(labeled) < 2:
        raise ValueError("At least two complete-label studies are required")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labeled))
    valid_count = max(1, int(round(len(labeled) * validation_fraction)))
    valid_indices = indices[:valid_count]
    train_indices = indices[valid_count:]
    return labeled.iloc[train_indices].reset_index(drop=True), labeled.iloc[valid_indices].reset_index(drop=True)


def metadata_report(metadata: RSNAKneeMetadata) -> dict:
    series = metadata.train_series
    series_per_study = series.groupby("StudyInstanceUID").size()
    prevalence, missing = {}, {}
    for target in TARGET_COLUMNS:
        prevalence[target] = float(metadata.train[target].dropna().mean()) if metadata.train[target].notna().any() else None
        missing[target] = int(metadata.train[target].isna().sum())
    dicom_train = metadata.data_dir / "train_series"
    dicom_test = metadata.data_dir / "test_series"
    return {
        "training_studies": int(metadata.train["StudyInstanceUID"].nunique()),
        "test_studies": int(metadata.test["StudyInstanceUID"].nunique()),
        "training_rows": int(len(metadata.train)),
        "series_rows": int(len(series)),
        "series_per_study": {"min": int(series_per_study.min()) if len(series_per_study) else 0,
                             "max": int(series_per_study.max()) if len(series_per_study) else 0,
                             "mean": float(series_per_study.mean()) if len(series_per_study) else 0.0,
                             "median": float(series_per_study.median()) if len(series_per_study) else 0.0},
        "anatomical_plane_counts": {str(k): int(v) for k, v in series["Anatomical_Plane"].value_counts(dropna=False).items()},
        "fluid_sensitive_counts": {str(k): int(v) for k, v in series["Fluid_Sensitive"].value_counts(dropna=False).items()},
        "fat_suppression_counts": {str(k): int(v) for k, v in series["Fat_Suppression"].value_counts(dropna=False).items()},
        "label_prevalence": prevalence,
        "missing_label_counts": missing,
        "complete_label_studies": int(len(metadata.complete_label_train)),
        "dicom_directories": {"train": dicom_train.is_dir(), "test": dicom_test.is_dir(),
                              "train_path": str(dicom_train), "test_path": str(dicom_test)},
    }
