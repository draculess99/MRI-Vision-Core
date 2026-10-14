"""Run one real RSNA study through an existing trained checkpoint and write a structured result.

All inference goes through the shared engine (mri_core.rsna_inference_engine); this module
only picks a study, calls it, and lays the output out in three deliberately separate
top-level sections so nothing downstream can blur them:

  prediction         -- the engine's StudyInferenceResult, verbatim (raw sigmoid probabilities)
  evidence_review    -- deterministic evidence/consistency checks from mri_core.agents, plus an
                        explicit statement that no uncertainty estimate exists
  human_review_gate  -- the simulated review state machine (starts at pending_review)

Nothing here fabricates a prediction: if the checkpoint or study is missing the run fails
with an error report and a non-zero exit code.

Usage:
  python -m hackathon.amd.run_inference --data-root data/rsna-knee --checkpoint last.pt \
      [--study-uid UID] [--device cpu|cuda] [--output outputs/amd/prediction.json]
Exit codes: 0 ok, 1 configuration/checkpoint error, 3 study unable to be evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from mri_core.agents.contracts import QualityGateStatus
from mri_core.agents.orchestrator import WorkflowResult, process_inference_result
from mri_core.rsna_inference_engine import (
    STATUS_OK,
    run_single_model_inference,
)

from ._common import add_data_args, load_metadata_and_study
from .rocm_env import check_environment

SCHEMA = "amd-inference-result/v1"

UNCERTAINTY_NOT_ESTIMATED = {
    "status": "not_estimated",
    "note": "No uncertainty or calibration method exists for these checkpoints. Probabilities "
            "are raw, uncalibrated sigmoid outputs and must not be read as confidence.",
}


def evidence_section(workflow: WorkflowResult) -> dict:
    data = workflow.to_dict()
    return {
        "evidence_fingerprint": data["bundle"]["fingerprint"],
        "quality_gate_status": data["bundle"]["quality_gate_status"],
        "limitations": data["bundle"]["limitations"],
        "consensus": data["consensus"],
        "critic_findings": data["critic_findings"],
        "specialist_summaries": data["specialist_summaries"],
        "uncertainty": UNCERTAINTY_NOT_ESTIMATED,
    }


def review_gate_section(workflow: WorkflowResult) -> dict:
    gate = workflow.to_dict()["review_gate"]
    return {**gate, "simulated": True, "note": "Simulated workflow state; no clinician approval is implied."}


def build_report(result, workflow: WorkflowResult, environment: dict, device: str) -> dict:
    return {
        "schema": SCHEMA,
        "requested_device": device,
        "environment": environment,
        "prediction": result.to_dict(),
        "evidence_review": evidence_section(workflow),
        "human_review_gate": review_gate_section(workflow),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_args(parser)
    parser.add_argument("--checkpoint", required=True, help="trained RSNAKneeCNN checkpoint (.pt)")
    parser.add_argument("--device", default="cpu", help="torch device for the forward pass (cpu, or cuda for ROCm)")
    parser.add_argument("--output", default="outputs/amd/prediction.json")
    args = parser.parse_args(argv)

    environment = check_environment()
    try:
        metadata, study_uid = load_metadata_and_study(args.data_root, args.dicom_root, args.study_uid)
        result = run_single_model_inference(metadata, args.checkpoint, study_uid, device=args.device)
    except Exception as exc:  # report, never fabricate
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    workflow = process_inference_result(result, quality_gate_status=QualityGateStatus.NOT_ASSESSED)
    report = build_report(result, workflow, environment, args.device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}  status={result.status} study={study_uid} latency={result.latency_seconds:.3f}s")
    return 0 if result.status == STATUS_OK else 3


if __name__ == "__main__":
    sys.exit(main())
