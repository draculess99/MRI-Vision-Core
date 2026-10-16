"""Build the blinded ~100-excerpt label audit sample from TRAINING-ONLY reports (gold studies excluded).

Outputs go to an explicit directory (use a git-ignored one): a blind sheet for the adjudicator and a key with the automated states.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_core.report_label_audit import build_audit_sample, write_audit_files
from mri_core.rsna_knee_dataset import TARGET_COLUMNS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--old-profile", default="0.1.0")
    parser.add_argument("--new-profile", default="0.2.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-csv", type=Path, default=None,
                        help="optional third-party label file (StudyInstanceUID + finding columns, probabilities); key only")
    args = parser.parse_args(argv)
    train = pd.read_csv(args.train_csv, dtype={"StudyInstanceUID": str})
    gold = set(train.dropna(subset=list(TARGET_COLUMNS)).StudyInstanceUID.astype(str))
    sample = build_audit_sample(train, gold, old_profile=args.old_profile, new_profile=args.new_profile, seed=args.seed)
    reference = None
    if args.reference_csv:
        ref = pd.read_csv(args.reference_csv).set_index("StudyInstanceUID")
        state = lambda p: "not_addressed" if abs(p - 0.5) < 0.02 else ("positive" if p >= 0.7 else "negative" if p <= 0.3 else "uncertain")
        reference = {(r["study_uid"], r["target"]): state(float(ref.loc[r["study_uid"], r["target"]])) for r in sample}
    sheet, key = write_audit_files(sample, args.out_dir, reference=reference)
    counts = {}
    for r in sample:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
    print(f"wrote {sheet.name} and {key.name}: {len(sample)} excerpts")
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
