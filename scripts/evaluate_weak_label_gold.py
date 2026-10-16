"""One-look, paired gold evaluation of frozen weak-label checkpoints (see mri_core/weak_label_gold_report.py).

Pre-registered settings (do not change after seeing results): trained targets only, threshold 0.5, 2000 bootstrap rounds, seed 0.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_core.rsna_knee_dataset import load_rsna_metadata
from mri_core.weak_label_gold_report import DEFAULT_THRESHOLDS, TIER1, score_models_on_gold
from mri_core.weak_label_train import GoldAlreadyScoredError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path, default=None)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    metadata = load_rsna_metadata(args.data_dir, args.dicom_root)
    gold = metadata.complete_label_train.reset_index(drop=True)
    try:
        report = score_models_on_gold(checkpoints, gold, metadata, ledger_path=args.ledger, force=args.force, rounds=args.rounds,
                                      targets=TIER1, thresholds=DEFAULT_THRESHOLDS)
    except GoldAlreadyScoredError as exc:
        print(f"REFUSED: {exc}")
        return 2
    arrays = report.pop("_arrays")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), uids=np.array(arrays["uids"]), labels=arrays["labels"],
             **{f"prob_{n}": p for n, p in arrays.items() if False}, **{f"prob_{n}": p for n, p in arrays["probabilities"].items()})
    args.output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.output} and {args.output.with_suffix('.npz').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
