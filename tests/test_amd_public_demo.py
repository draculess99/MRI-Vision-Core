"""Public synthetic-demo mode: loader validation, the populated 'Select demonstration case'
dropdown, deterministic execution of every case, no model/DICOM/checkpoint/network use, and the
unchanged human-review gate. Never needs real data, a checkpoint, an API key or a network."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pydicom")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from hackathon.amd import public_demo_cases as pdc

APP_PATH = str(Path(__file__).resolve().parents[1] / "hackathon" / "amd" / "streamlit_app.py")
RUN_TIMEOUT = 60
LABELS = [
    "DEMO-001 — Low concern / routine review",
    "DEMO-002 — Multiple evidence items / reviewer attention",
    "DEMO-003 — Uncertain evidence / request more evidence",
    "DEMO-004 — Data-quality limitation",
    "DEMO-005 — Mandatory human escalation",
]


@pytest.fixture
def public_env(monkeypatch, tmp_path):
    for key in ("AMD_KAGGLE_RSNA_ROOT", "AMD_KAGGLE_CHECKPOINT", "GROQ_API_KEY", "AMD_TOKENFACTORY_API_KEY",
                "AMD_TOKENFACTORY_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AMD_PUBLIC_DEMO_MODE", "auto")
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "false")
    import hackathon.amd.llm_config as llm_config

    monkeypatch.setattr(llm_config, "_DEFAULT_DOTENV_PATH", tmp_path / "no.env")


@pytest.fixture
def forbid_everything(monkeypatch):
    """Any network access, model/checkpoint load, inference, or DICOM read fails the test."""
    import pydicom
    import requests.sessions

    import hackathon.amd.kaggle_mil_bridge as bridge

    def boom(*args, **kwargs):
        raise AssertionError("forbidden call in public synthetic-demo mode")

    monkeypatch.setattr(requests.sessions.Session, "send", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(bridge, "load_bridge_checkpoint", boom)
    monkeypatch.setattr(bridge, "run_study_inference", boom)
    monkeypatch.setattr(pydicom, "dcmread", boom)


def _doc():
    return json.loads(pdc.DEFAULT_CASES_PATH.read_text(encoding="utf-8"))


class TestCaseFile:
    def test_exactly_five_valid_synthetic_cases_with_unique_ids(self):
        case_set = pdc.load_public_demo_cases()
        assert [pdc.case_label(c) for c in case_set.cases] == LABELS
        assert len({c.case_id for c in case_set.cases}) == 5
        assert "not clinical advice or diagnosis" in case_set.disclaimer

    def test_contains_no_real_study_ids_paths_or_competition_data(self):
        text = pdc.DEFAULT_CASES_PATH.read_text(encoding="utf-8")
        for fragment in ("1.2.826", ".dcm", ".ckpt", "train_series", "StudyInstanceUID", "Report"):
            assert fragment not in text

    def test_missing_file_fails_cleanly(self, tmp_path):
        with pytest.raises(pdc.PublicDemoCasesError, match="not found"):
            pdc.load_public_demo_cases(tmp_path / "nope.json")

    @pytest.mark.parametrize("mutate", [
        lambda d: d["cases"].pop(),
        lambda d: d["cases"].append(dict(d["cases"][0], case_id="DEMO-006", title="x")),
        lambda d: d["cases"][1].update(case_id="DEMO-001"),
        lambda d: d["cases"][0].pop("expected_reason"),
        lambda d: d["cases"][0].update(synthetic=False),
        lambda d: d["cases"][0].update(expected_workflow_status="APPROVED"),
        lambda d: d["cases"][0]["scores"].pop("ACL"),
        lambda d: d["cases"][0]["scores"].update(ACL=1.5),
        lambda d: d["cases"][0].update(evidence_references=["1.2.826.0.1.3680043.8.498.1"]),
        lambda d: d.update(disclaimer="synthetic"),
        lambda d: d.pop("cases"),
    ])
    def test_malformed_content_fails_cleanly(self, tmp_path, mutate):
        doc = _doc()
        mutate(doc)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(pdc.PublicDemoCasesError):
            pdc.load_public_demo_cases(path)

    def test_invalid_json_fails_cleanly(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{nope", encoding="utf-8")
        with pytest.raises(pdc.PublicDemoCasesError, match="not valid JSON"):
            pdc.load_public_demo_cases(path)


class TestDeterministicExecution:
    def test_every_case_yields_its_expected_status_and_starts_pending(self):
        from mri_core.agents.contracts import ReviewState
        from mri_core.agents.orchestrator import process_inference_result

        for case in pdc.load_public_demo_cases().cases:
            runs = []
            for _ in range(2):
                result, quality, refs = pdc.build_case_inputs(case)
                workflow = process_inference_result(result, quality_gate_status=quality, evidence_references=refs)
                runs.append((workflow.consensus.workflow_status.value, workflow.bundle.fingerprint,
                             tuple(f.code for f in workflow.critic_findings)))
                assert workflow.review_gate.state == ReviewState.PENDING_REVIEW
            assert runs[0] == runs[1]
            assert runs[0][0] == case.expected_workflow_status, case.case_id


class TestPublicModeApp:
    def test_dropdown_banner_and_no_real_scenarios(self, public_env, forbid_everything):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        selector = at.selectbox(key="public_demo_case_select")
        assert selector.label == "Select demonstration case" and selector.options == LABELS
        assert any("Synthetic workflow demonstration" in w.value for w in at.warning)
        assert any("Research/demo only—not clinical advice or diagnosis." in c.value for c in at.caption)
        assert "benchmark_scenario_select" not in [s.key for s in at.selectbox]
        assert "study_select" not in [s.key for s in at.selectbox]
        assert "run_inference_button" not in [b.key for b in at.button]
        assert "public_demo_run_button" in [b.key for b in at.button]

    @pytest.mark.parametrize("index", range(5))
    def test_each_case_runs_without_model_dicom_checkpoint_or_network(self, public_env, forbid_everything, index):
        expected = pdc.load_public_demo_cases().cases[index].expected_workflow_status
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.selectbox(key="public_demo_case_select").set_value(LABELS[index]).run(timeout=RUN_TIMEOUT)
        at.button(key="public_demo_run_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any(f"workflow_status: `{expected}`" in m.value for m in at.markdown)
        assert any("Current state: `PENDING_REVIEW`" in m.value for m in at.markdown)
        assert not at.table  # no raw scores table
        assert not at.json  # no scores / provenance JSON
        assert "public_llm_provider_choice" not in [r.key for r in at.radio]  # local-only text

    def test_human_gate_still_requires_a_reason_and_a_named_actor_action(self, public_env, forbid_everything):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="public_demo_run_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="public_approve_button").click().run(timeout=RUN_TIMEOUT)
        assert any("A reason is required" in e.value for e in at.error)
        assert any("Current state: `PENDING_REVIEW`" in m.value for m in at.markdown)
        at.text_area(key="public_reason_input").set_value("Synthetic walkthrough only").run(timeout=RUN_TIMEOUT)
        at.button(key="public_escalate_button").click().run(timeout=RUN_TIMEOUT)
        assert any("Recorded 'escalate'" in s.value for s in at.success)
        assert len(at.table) == 1  # the audit history now has an entry

    def test_llm_generation_stays_local_even_if_env_requests_an_external_provider(
        self, public_env, forbid_everything, monkeypatch,
    ):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-never-used")
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="public_demo_run_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="public_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Produced by: local" in c.value for c in at.caption)

    def test_malformed_case_file_shows_a_clean_error_and_no_fabricated_result(self, public_env, monkeypatch, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{nope", encoding="utf-8")
        monkeypatch.setattr(pdc, "DEFAULT_CASES_PATH", bad)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Could not load the public demonstration cases" in e.value for e in at.error)
        assert "public_demo_run_button" not in [b.key for b in at.button]

    def test_opt_out_variable_restores_the_metadata_only_behavior(self, public_env, monkeypatch):
        monkeypatch.setenv("AMD_PUBLIC_DEMO_MODE", "false")
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert "benchmark_scenario_select" in [s.key for s in at.selectbox]
        assert "public_demo_case_select" not in [s.key for s in at.selectbox]

    @pytest.mark.parametrize("root, ckpt", [
        ("/app/data/rsna-knee", "/app/models/fold0-knee-mil-best.ckpt"),  # stale Railway values
        ("/app/data/rsna-knee", ""),
    ])
    def test_non_empty_but_invalid_paths_select_public_mode(self, public_env, forbid_everything, monkeypatch, root, ckpt):
        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", root)
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", ckpt)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert at.selectbox(key="public_demo_case_select").label == "Select demonstration case"
        assert "benchmark_scenario_select" not in [s.key for s in at.selectbox]
        assert not at.error

    def test_valid_root_with_missing_checkpoint_selects_public_mode(self, public_env, monkeypatch, tmp_path):
        root = tmp_path / "rsna"
        root.mkdir()
        (root / "train.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", str(root))
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", str(tmp_path / "missing.ckpt"))
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert "public_demo_case_select" in [s.key for s in at.selectbox]

    def test_valid_real_paths_keep_local_real_inference_mode(self, public_env, monkeypatch, tmp_path):
        root = tmp_path / "rsna"
        root.mkdir()
        (root / "train.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
        ckpt = tmp_path / "w.ckpt"
        ckpt.write_bytes(b"x")
        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", str(root))
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", str(ckpt))
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert "public_demo_case_select" not in [s.key for s in at.selectbox]
        assert "benchmark_scenario_select" in [s.key for s in at.selectbox]
