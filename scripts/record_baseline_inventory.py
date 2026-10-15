"""Record a reproducible, read-only inventory of the preserved baseline (checkpoints, benchmark artifacts, gold labels).

Run it before and after weak-label work and compare the two outputs with --compare to prove nothing
under outputs/ or the original train.csv changed. It only reads; it writes just the one JSON file named by --output.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXCLUDED_DIRS = ("weak-labels",)  # artifacts produced by the weak-label pipeline itself


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def gold_summary(train_csv: Path) -> dict:
    import pandas as pd
    from mri_core.rsna_knee_dataset import TARGET_COLUMNS

    frame = pd.read_csv(train_csv, dtype={"StudyInstanceUID": "string"})
    targets = list(TARGET_COLUMNS)
    complete = frame.dropna(subset=targets)
    ids = sorted(complete["StudyInstanceUID"].astype(str))
    return {
        "studies": int(len(frame)),
        "gold_studies": len(ids),
        "unlabeled_studies": int(frame[targets].isna().all(axis=1).sum()),
        "partially_labeled_studies": int(len(frame) - len(ids) - frame[targets].isna().all(axis=1).sum()),
        "reports_present": int(frame["Report"].notna().sum()),
        "gold_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }


def build_inventory(root: Path, data_dir: Path) -> dict:
    outputs = root / "outputs"
    files = {}
    for path in sorted(outputs.rglob("*")):
        if path.is_file() and not any(part in EXCLUDED_DIRS for part in path.relative_to(outputs).parts):
            files[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    train_csv = data_dir / "train.csv"
    return {
        "git": {"branch": _git(root, "branch", "--show-current"), "head": _git(root, "rev-parse", "HEAD"),
                "dirty_paths": [line for line in _git(root, "status", "--short").splitlines()]},
        "train_csv": {"sha256": sha256_file(train_csv), **gold_summary(train_csv)},
        "metadata_files": {name: sha256_file(data_dir / name) for name in
                           ("test.csv", "train_series.csv", "test_series.csv", "sample_submission.csv")
                           if (data_dir / name).is_file()},
        "outputs_file_count": len(files),
        "checkpoint_count": sum(1 for name in files if name.endswith(".pt")),
        "outputs": files,
    }


def compare(before: dict, after: dict) -> list[str]:
    problems = []
    if before["train_csv"] != after["train_csv"]:
        problems.append("train.csv summary changed")
    if before["metadata_files"] != after["metadata_files"]:
        problems.append("metadata files changed")
    for name, info in before["outputs"].items():
        if name not in after["outputs"]:
            problems.append(f"removed: {name}")
        elif after["outputs"][name] != info:
            problems.append(f"modified: {name}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None, help="earlier inventory to compare against")
    args = parser.parse_args(argv)
    inventory = build_inventory(args.root, args.data_dir or args.root / "data" / "rsna-knee")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: inventory[k] for k in ("git", "outputs_file_count", "checkpoint_count")}, indent=2))
    if args.compare:
        problems = compare(json.loads(args.compare.read_text(encoding="utf-8")), inventory)
        print("PRESERVED: no changes" if not problems else "CHANGES DETECTED:\n  " + "\n  ".join(problems))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
