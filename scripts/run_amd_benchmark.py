"""CLI for the automated 22-scenario AMD curated benchmark (hackathon/amd/benchmark_runner.py).

Usage:
    python scripts/run_amd_benchmark.py [--data-root DIR] [--checkpoint PATH] [--device cpu]
        [--output-dir benchmarks/amd/results] [--threshold 0.5] [--scenario-id AMD-001] [--dry-run]

Defaults for --data-root/--checkpoint come from the same AMD_KAGGLE_RSNA_ROOT /
AMD_KAGGLE_CHECKPOINT environment configuration (optionally via .env) already used by
hackathon/amd/streamlit_app.py -- see hackathon/amd/data_config.py. No API key or LLM
provider is required; this script never imports hackathon.amd.llm_config/providers.

--dry-run loads the 22 scenarios, checks local availability, and prints what would run --
it never loads a checkpoint and never runs inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hackathon.amd.benchmark_runner import DEFAULT_THRESHOLD, build_report, dry_run_report, render_markdown, run_scenarios
from hackathon.amd.benchmark_scenarios import BenchmarkScenarioError, load_scenarios
from hackathon.amd.data_config import resolve_data_paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="RSNA knee data dir; defaults to AMD_KAGGLE_RSNA_ROOT")
    parser.add_argument("--checkpoint", help="Path to a foldN-knee-mil-best.ckpt file; defaults to AMD_KAGGLE_CHECKPOINT")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="benchmarks/amd/results")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--scenario-id", help="Run only this one scenario, e.g. AMD-001")
    parser.add_argument("--dry-run", action="store_true", help="Check availability only; never runs inference")
    args = parser.parse_args(argv)

    defaults = resolve_data_paths()
    data_root = Path(args.data_root) if args.data_root else defaults.data_root
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else defaults.checkpoint

    try:
        scenario_set = load_scenarios()
    except BenchmarkScenarioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.scenario_id:
        matching = [s for s in scenario_set.scenarios if s["scenario_id"] == args.scenario_id]
        if not matching:
            print(f"ERROR: no scenario with scenario_id={args.scenario_id!r}", file=sys.stderr)
            return 1
        from dataclasses import replace

        scenario_set = replace(scenario_set, scenarios=tuple(matching), scenario_count=1)

    if args.dry_run:
        rows = dry_run_report(scenario_set, data_root)
        print(f"Dry run: {len(rows)} scenario(s), data_root={data_root}")
        for row in rows:
            status = "WOULD RUN" if row["would_run"] else "unavailable"
            print(f"  {row['scenario_id']}  {status:<11}  {row['study_id']}  {row['title']}")
        if any(row["availability_check_error"] for row in rows):
            print(f"Availability check error: {rows[0]['availability_check_error']}", file=sys.stderr)
        return 0

    if data_root is None or checkpoint_path is None:
        print("ERROR: --data-root/--checkpoint (or AMD_KAGGLE_RSNA_ROOT/AMD_KAGGLE_CHECKPOINT) are required "
              "for a real run; use --dry-run to only check availability.", file=sys.stderr)
        return 1

    results, checkpoint_provenance, checkpoint_load_error = run_scenarios(
        scenario_set, data_root=data_root, checkpoint_path=checkpoint_path, device=args.device,
        threshold=args.threshold,
    )
    report = build_report(
        scenario_set=scenario_set, results=results, checkpoint_path=checkpoint_path,
        checkpoint_provenance=checkpoint_provenance, checkpoint_load_error=checkpoint_load_error,
        data_root=data_root, device=args.device, threshold=args.threshold,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "amd_benchmark_latest.json"
    md_path = out_dir / "amd_benchmark_latest.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    coverage = report["coverage"]
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(
        f"total={coverage['total_curated_scenarios']} available={coverage['locally_available_scenarios']} "
        f"inferred={coverage['successfully_inferred_scenarios']} unavailable={coverage['unavailable_scenarios']} "
        f"failed={coverage['failed_scenarios']} coverage={coverage['inference_coverage_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
