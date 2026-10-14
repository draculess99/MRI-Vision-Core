"""Kaggle-native RSNA Knee research runner (inspect / benchmark / train / export).

Reads the competition data in place from the mounted /kaggle/input tree -- nothing is copied or
re-published -- and drives the existing, unmodified mri_core training stack
(train_rsna_resumable, RSNAKneeCNN, study-level splits). Nothing here trains unless the
`benchmark` or `train` subcommand is run; `discover` and `inspect` are read-only.

Typical notebook use:  python scripts/kaggle_rsna_runner.py inspect
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mri_core.rsna_knee_dataset import (
    PLANE_NAMES, TARGET_COLUMNS, RSNAKneeMetadata, load_rsna_metadata, metadata_report, select_series,
)

INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
REQUIRED_CSVS = ("train.csv", "train_series.csv", "test.csv", "test_series.csv", "sample_submission.csv")
DEFAULT_CONFIG = {
    "image_size": 128, "slices_per_series": 9, "planes": list(PLANE_NAMES), "dropout": 0.2,
    "batch_size": 4, "num_workers": 2, "learning_rate": 1e-3, "epochs": 20,
    "validation_fraction": 0.2, "seed": 0, "device": "auto",
}
# Only these artifact types may leave the notebook via `export`; raw competition data may not.
EXPORTABLE_NAMES = ("last.pt", "last_best.pt", "run_plan.json", "selection.json", "benchmark.json", "inspect.json")


class DiscoveryError(RuntimeError):
    pass


# --------------------------------------------------------------------------- discovery
def discover_data(input_root: Path = INPUT_ROOT, max_depth: int = 4) -> tuple[Path, Path]:
    """Return (metadata_dir, dicom_root) by searching for the RSNA files, not a fixed folder name."""
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise DiscoveryError(f"Input root not found: {input_root}. Attach the competition under 'Add Input'.")
    hits = []
    for current, dirs, files in os.walk(input_root):
        depth = len(Path(current).relative_to(input_root).parts)
        if depth >= max_depth:
            dirs[:] = []
        if "train.csv" in files and "train_series.csv" in files:
            missing = [name for name in REQUIRED_CSVS if name not in files]
            hits.append((Path(current), missing))
    if not hits:
        raise DiscoveryError(f"No directory under {input_root} contains train.csv and train_series.csv.")
    complete = [path for path, missing in hits if not missing]
    if len(complete) != 1:
        found = [f"{p} (missing: {m or 'none'})" for p, m in hits]
        raise DiscoveryError("Expected exactly one complete RSNA metadata directory, found: " + "; ".join(found)
                             + ". Pass --data-dir explicitly.")
    metadata_dir = complete[0]
    return metadata_dir, find_dicom_root(metadata_dir)


def find_dicom_root(metadata_dir: Path) -> Path:
    """The folder that holds train_series/ (usually the metadata dir itself, otherwise a near neighbour)."""
    for candidate in (metadata_dir, metadata_dir.parent, *[p for p in metadata_dir.iterdir() if p.is_dir()]):
        if (candidate / "train_series").is_dir():
            return candidate
    raise DiscoveryError(f"train_series/ directory not found at or near {metadata_dir}")


# --------------------------------------------------------------------------- selection
def _series_dir_ok(metadata: RSNAKneeMetadata, study_uid: str, planes) -> bool:
    rows = metadata.series_for(study_uid)
    for plane in planes:
        candidates = rows[rows["Anatomical_Plane"].astype(str).str.lower() == plane.lower()]
        if candidates.empty:
            return False
        directory = metadata.dicom_series_dir(study_uid, select_series(candidates))
        if not directory.is_dir() or not any(directory.iterdir()):
            return False
    return True


def available_labeled_studies(metadata: RSNAKneeMetadata, planes=PLANE_NAMES) -> list[str]:
    """Complete-label studies whose chosen series for every required plane exist on disk (sorted)."""
    return sorted(str(uid) for uid in metadata.complete_label_train["StudyInstanceUID"]
                  if _series_dir_ok(metadata, str(uid), planes))


def restrict_metadata(metadata: RSNAKneeMetadata, study_uids) -> RSNAKneeMetadata:
    """New metadata object limited to the given labeled studies (train only; test is left alone)."""
    keep = set(study_uids)
    train = metadata.train[metadata.train["StudyInstanceUID"].astype(str).isin(keep)].reset_index(drop=True)
    series = metadata.train_series[metadata.train_series["StudyInstanceUID"].astype(str).isin(keep)]
    return dataclasses.replace(metadata, train=train, train_series=series.reset_index(drop=True))


def select_subset(available: list[str], limit: int | None, seed: int) -> list[str]:
    if not limit or limit >= len(available):
        return list(available)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(available, size=limit, replace=False).tolist())


def load_or_create_selection(path: Path, available: list[str], limit: int | None, seed: int) -> list[str]:
    """Persist the exact study list so every resumed session trains on the same studies."""
    if path.is_file():
        saved = json.loads(path.read_text())["study_uids"]
        missing = sorted(set(saved) - set(available))
        if missing:
            raise RuntimeError(f"{len(missing)} studies from the saved selection are not available now")
        return saved
    chosen = select_subset(available, limit, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "limit": limit, "study_uids": chosen}, indent=2))
    return chosen


# --------------------------------------------------------------------------- inspect (read-only)
def inspect_data(metadata: RSNAKneeMetadata, *, max_list: int = 20, size_sample: int = 25) -> dict:
    """Read-only: counts, labels and (sampled) on-disk sizes. Never decodes pixels or writes anything."""
    report = metadata_report(metadata)
    report["targets"] = list(TARGET_COLUMNS)
    report["target_count"] = len(TARGET_COLUMNS)
    available = available_labeled_studies(metadata)
    report["labeled_studies_available_on_disk"] = len(available)
    studies = []
    for uid in available[:max_list]:
        rows = metadata.series_for(uid)
        studies.append({"study": uid, "series": len(rows),
                        "planes": sorted(rows["Anatomical_Plane"].astype(str).unique().tolist()),
                        "positive_targets": [t for t in TARGET_COLUMNS if metadata.target_vector(uid)[TARGET_COLUMNS.index(t)] == 1]})
    report["study_listing"] = {"shown": len(studies), "of": len(available), "studies": studies}
    sizes = []
    for uid in available[:size_sample]:
        total = files = 0
        for series_uid in metadata.series_for(uid)["SeriesInstanceUID"].astype(str):
            directory = metadata.dicom_series_dir(uid, series_uid)
            if directory.is_dir():
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.is_file():
                            files += 1
                            total += entry.stat().st_size
        sizes.append((files, total))
    if sizes:
        report["sampled_study_size"] = {"studies_sampled": len(sizes),
                                        "mean_files": float(np.mean([s[0] for s in sizes])),
                                        "mean_megabytes": float(np.mean([s[1] for s in sizes])) / 1e6}
    return report


# --------------------------------------------------------------------------- training
def device_info() -> dict:
    """Describe whatever accelerator is actually present; assume nothing."""
    import torch
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "gpus": []}
    if info["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append({"index": index, "name": props.name, "memory_gb": round(props.total_memory / 1e9, 1)})
    return info


def _peek_epoch(checkpoint: Path) -> int:
    import torch
    return int(torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"])


def _stage_resume(resume_from: Path | None, out_dir: Path) -> None:
    """Copy last.pt from a previous notebook's output (attached as an input) into the working dir."""
    if resume_from is None:
        return
    source = resume_from / "last.pt" if resume_from.is_dir() else resume_from
    if not source.is_file():
        raise FileNotFoundError(f"No checkpoint at {source}")
    target = out_dir / "last.pt"
    if target.exists():
        raise FileExistsError(f"{target} already exists; refusing to overwrite it with {source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def run_training(metadata, config: dict, out_dir: Path, *, study_limit, resume_from, epochs_this_session):
    from mri_core.rsna_knee_train_resumable import train_rsna_resumable
    out_dir.mkdir(parents=True, exist_ok=True)
    _stage_resume(resume_from, out_dir)
    available = available_labeled_studies(metadata, config["planes"])
    if len(available) < 2:
        raise RuntimeError(f"Only {len(available)} usable labeled studies found; need at least 2.")
    chosen = load_or_create_selection(out_dir / "selection.json", available, study_limit, int(config["seed"]))
    subset = restrict_metadata(metadata, chosen)
    last = out_dir / "last.pt"
    done = _peek_epoch(last) if last.is_file() else 0
    total = int(config["epochs"])
    config = dict(config, epochs=min(total, done + epochs_this_session) if epochs_this_session else total)
    (out_dir / "run_plan.json").write_text(json.dumps(
        {"config": config, "planned_total_epochs": total, "epochs_done_before_session": done,
         "studies": len(chosen), "device": device_info()}, indent=2))
    # scaler=None / no scheduler: the resumable trainer is used exactly as-is (no new architecture).
    return train_rsna_resumable(subset, config, out_dir, resume=True)


def run_benchmark(metadata, config: dict, out_dir: Path, *, studies: int, epochs: int):
    """Tiny, fixed-size timing run on a seeded subset; reports time/epoch and peak GPU memory."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = run_training(metadata, dict(config, epochs=epochs), out_dir,
                          study_limit=studies, resume_from=None, epochs_this_session=0)
    seconds = time.perf_counter() - started
    summary = {
        "studies": result["train_studies"] + result["validation_studies"], "epochs": epochs,
        "seconds_total": round(seconds, 1), "seconds_per_epoch": round(seconds / max(epochs, 1), 1),
        "peak_cuda_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "device": result["device"], "final_metrics": result["history"][-1] if result["history"] else None,
        "note": "Smoke/timing run only. Metrics on a handful of studies are not evidence of model quality.",
    }
    (out_dir / "benchmark.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


# --------------------------------------------------------------------------- export
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_artifacts(out_dir: Path, export_dir: Path) -> dict:
    """Bundle checkpoints + run metadata (never competition data) into export/ with a SHA-256 manifest and a zip."""
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in EXPORTABLE_NAMES:
        source = out_dir / name
        if source.is_file():
            shutil.copy2(source, export_dir / name)
            manifest[name] = {"sha256": sha256_file(export_dir / name), "bytes": source.stat().st_size}
    if "last.pt" not in manifest:
        raise FileNotFoundError(f"No last.pt in {out_dir}; nothing to export")
    (export_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    archive = export_dir / "rsna_run_export.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in (*manifest, "MANIFEST.json"):
            bundle.write(export_dir / name, name)
    return {"export_dir": str(export_dir), "archive": str(archive), "files": manifest}


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("discover", "inspect", "benchmark", "train", "export"):
        command = sub.add_parser(name)
        command.add_argument("--input-root", type=Path, default=INPUT_ROOT)
        command.add_argument("--data-dir", type=Path, default=None, help="override metadata directory discovery")
        command.add_argument("--dicom-root", type=Path, default=None)
        command.add_argument("--work-dir", type=Path, default=WORKING_ROOT / "outputs" / "rsna-knee")
        if name in ("benchmark", "train"):
            command.add_argument("--config", type=Path, default=None, help="JSON overrides for the default config")
    sub.choices["inspect"].add_argument("--max-list", type=int, default=20)
    sub.choices["benchmark"].add_argument("--studies", type=int, default=24)
    sub.choices["benchmark"].add_argument("--epochs", type=int, default=2)
    sub.choices["train"].add_argument("--studies", type=int, default=None, help="cap on labeled studies (default: all available)")
    sub.choices["train"].add_argument("--resume-from", type=Path, default=None)
    sub.choices["train"].add_argument("--epochs-this-session", type=int, default=0, help="0 = run to the planned total")
    sub.choices["export"].add_argument("--export-dir", type=Path, default=WORKING_ROOT / "export")
    return parser


def _load_config(path: Path | None) -> dict:
    return {**DEFAULT_CONFIG, **(json.loads(path.read_text()) if path else {})}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        print(json.dumps(export_artifacts(args.work_dir, args.export_dir), indent=2))
        return 0
    if args.data_dir is not None:
        metadata_dir, dicom_root = args.data_dir, args.dicom_root or find_dicom_root(args.data_dir)
    else:
        metadata_dir, dicom_root = discover_data(args.input_root)
    if args.command == "discover":
        print(json.dumps({"metadata_dir": str(metadata_dir), "dicom_root": str(dicom_root), "device": device_info()}, indent=2))
        return 0
    metadata = load_rsna_metadata(metadata_dir, dicom_root)
    if args.command == "inspect":
        report = inspect_data(metadata, max_list=args.max_list)
        report["device"] = device_info()
        print(json.dumps(report, indent=2, default=str))
        return 0
    config = _load_config(args.config)
    if args.command == "benchmark":
        print(json.dumps(run_benchmark(metadata, config, args.work_dir / "benchmark", studies=args.studies, epochs=args.epochs), indent=2, default=float))
    else:
        print(json.dumps(run_training(metadata, config, args.work_dir, study_limit=args.studies, resume_from=args.resume_from,
                                      epochs_this_session=args.epochs_this_session), indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
