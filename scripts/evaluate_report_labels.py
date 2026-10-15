"""Score the report-label extractor against the 58 gold studies (DEV/EVAL discipline enforced).

Use --split dev while changing rules. --split eval scores the untouched half once per extractor version
(recorded in the ledger); --split all is exploratory and also counts as an EVAL look.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mri_core.report_label_eval import EvalAlreadySpentError, format_summary, run_gold_evaluation
from mri_core.report_label_rules import RuleBasedExtractor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "eval", "all"), default="dev")
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.split != "dev" and args.ledger is None:
        parser.error("--ledger is required for eval/all so the single look is recorded")
    train = pd.read_csv(args.train_csv, dtype={"StudyInstanceUID": str})
    try:
        report = run_gold_evaluation(RuleBasedExtractor(), train, split=args.split, ledger_path=args.ledger,
                                     force=args.force, rounds=args.rounds)
    except EvalAlreadySpentError as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(format_summary(report))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
