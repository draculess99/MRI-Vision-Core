"""Build a versioned weak-label manifest from radiology reports (deterministic rules, English/Spanish).

--dry-run computes and prints counts only and writes nothing. A real run needs an explicit --output (.jsonl) and
refuses to overwrite without --overwrite. train.csv and the DICOMs are only read. Logs contain counts, never
report text or full study UIDs.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_core.report_label_rules import RuleBasedExtractor
from mri_core.weak_label_manifest import build_manifest, sha256_file, write_manifest


def _ids(path):
    return [] if path is None else pd.read_csv(path, usecols=["StudyInstanceUID"], dtype=str)["StudyInstanceUID"].tolist()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--holdout-csv", type=Path, default=None, help="CSV with a StudyInstanceUID column to keep out of training")
    parser.add_argument("--output", type=Path, default=None, help="manifest .jsonl path (required unless --dry-run)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-evidence", action="store_true", help="store evidence sentences (default: hashes only)")
    parser.add_argument("--min-decided", type=int, default=1,
                        help="minimum decided (positive/negative) ACTIVE targets a study needs to be included")
    parser.add_argument("--active-targets", default=None,
                        help='comma-separated targets allowed to train, e.g. "ACL,Medial Meniscus,Lateral Meniscus"; '
                             "all other targets keep their extracted state but are masked (default: all 12)")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run")
    train = pd.read_csv(args.train_csv, dtype={"StudyInstanceUID": str})
    test_ids = _ids(args.test_csv) + _ids(args.sample_submission)
    rows, summary = build_manifest(train, RuleBasedExtractor(), test_ids=test_ids, holdout_ids=_ids(args.holdout_csv),
                                   include_evidence=args.include_evidence, min_decided=args.min_decided,
                                   train_csv_sha256=sha256_file(args.train_csv),
                                   active_targets=None if args.active_targets is None else
                                   [t.strip() for t in args.active_targets.split(",") if t.strip()])
    print(json.dumps({k: summary[k] for k in ("extractor_version", "counts", "leakage_checks", "languages_eligible", "trainable",
                                         "manifest_sha256")},
                     indent=2))
    if args.dry_run:
        print("DRY RUN: nothing written")
        return 0
    protected = [p for p in (args.train_csv, args.test_csv, args.sample_submission, args.holdout_csv) if p]
    out, summary_path = write_manifest(rows, summary, args.output, overwrite=args.overwrite, protected=protected)
    print(f"wrote {out.name} and {summary_path.name} ({len(rows)} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
