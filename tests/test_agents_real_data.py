"""Real-data integration smoke test for mri_core.agents, using the existing local RSNA
checkpoints and a real local study. No retraining, no checkpoint modification, no
network calls. Skips cleanly when the local data/checkpoints aren't available.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mri_core.rsna_inference_engine import STATUS_OK, run_single_model_inference, sha256_of_file
from mri_core.rsna_integration import discover_available_studies, load_rsna_metadata_safe
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

from mri_core.agents.contracts import QualityGateStatus, WorkflowStatus
from mri_core.agents.orchestrator import process_inference_result
from mri_core.agents.review_gate import ReviewState

pytestmark = pytest.mark.rsna

ROOT = Path(__file__).resolve().parents[1]
RSNA_ROOT = ROOT / "data" / "rsna-knee"
CHECKPOINT = ROOT / "outputs" / "rsna-fivefold-benchmark" / "random" / "fold0" / "last.pt"


def _real_data_available():
    return (RSNA_ROOT / "train.csv").is_file() and CHECKPOINT.is_file()


pytestmark = pytest.mark.skipif(not _real_data_available(), reason="Local RSNA data or Phase 5 checkpoint not available")


@pytest.fixture(scope="module")
def real_result():
    metadata = load_rsna_metadata_safe(RSNA_ROOT)
    for study_uid in discover_available_studies(RSNA_ROOT, metadata):
        result = run_single_model_inference(metadata, CHECKPOINT, study_uid)
        if result.status == STATUS_OK:
            return result
    pytest.skip("No fully-downloaded RSNA study produced a valid result")


def test_real_checkpoint_hash_before_and_after_is_unchanged(real_result):
    before = sha256_of_file(CHECKPOINT)
    process_inference_result(real_result, quality_gate_status=QualityGateStatus.NOT_ASSESSED)
    after = sha256_of_file(CHECKPOINT)
    assert before == after


def test_real_result_processes_without_mutation_and_preserves_all_twelve_labels(real_result):
    original_probs = dict(real_result.probabilities)
    wr = process_inference_result(
        real_result, quality_gate_status=QualityGateStatus.OK,
        evidence_references=(str(CHECKPOINT),),
    )
    # original result untouched
    assert real_result.probabilities == original_probs
    # all twelve labels preserved, in order, values identical
    assert list(wr.bundle.probabilities.keys()) == list(TARGET_COLUMNS)
    assert wr.bundle.probabilities == original_probs


def test_real_result_yields_a_defined_workflow_status(real_result):
    wr = process_inference_result(real_result, quality_gate_status=QualityGateStatus.OK)
    assert wr.consensus.workflow_status in WorkflowStatus


def test_real_result_with_quality_not_assessed_requires_more_evidence(real_result):
    """Not passing a real mri_core.decision quality status at all (the default) must
    never be silently treated as OK, even against a real, successful inference."""
    wr = process_inference_result(real_result)  # quality_gate_status defaults to NOT_ASSESSED
    assert wr.bundle.quality_gate_status == QualityGateStatus.NOT_ASSESSED
    assert wr.consensus.workflow_status == WorkflowStatus.MORE_EVIDENCE_REQUIRED


def test_real_result_review_gate_transition_works_end_to_end(real_result):
    wr = process_inference_result(real_result, quality_gate_status=QualityGateStatus.OK)
    fp = wr.bundle.fingerprint
    entry = wr.review_gate.transition(
        ReviewState.REVIEWED, actor="smoke-test", reason="real-data smoke test", evidence_fingerprint=fp
    )
    assert wr.review_gate.state == ReviewState.REVIEWED
    assert entry.evidence_fingerprint == fp


def test_processing_is_deterministic_on_real_data(real_result):
    wr1 = process_inference_result(real_result, quality_gate_status=QualityGateStatus.OK)
    wr2 = process_inference_result(real_result, quality_gate_status=QualityGateStatus.OK)
    assert wr1.bundle.fingerprint == wr2.bundle.fingerprint
    assert wr1.consensus.workflow_status == wr2.consensus.workflow_status


def test_no_network_module_imported_anywhere_in_agents_package():
    import mri_core.agents.contracts, mri_core.agents.specialist, mri_core.agents.critic
    import mri_core.agents.consensus, mri_core.agents.review_gate, mri_core.agents.llm_adapter
    import mri_core.agents.orchestrator, mri_core.agents.evaluation
    import sys
    for mod_name in list(sys.modules):
        if mod_name.startswith("mri_core.agents"):
            mod = sys.modules[mod_name]
            source_file = getattr(mod, "__file__", None)
            if source_file:
                text = Path(source_file).read_text(encoding="utf-8")
                for forbidden in ("import requests", "import urllib", "import socket", "httpx"):
                    assert forbidden not in text, f"{mod_name} unexpectedly imports {forbidden}"
