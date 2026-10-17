"""End-to-end tests for hackathon/amd/streamlit_app.py using streamlit.testing.v1.AppTest.

The no-config tests always run. The full-pipeline tests (inference -> evidence review ->
human review gate) need real local assets and are gated the same way as
tests/test_amd_kaggle_bridge.py: AMD_KAGGLE_CHECKPOINT and AMD_KAGGLE_RSNA_ROOT.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("pydicom")
streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "hackathon" / "amd" / "streamlit_app.py")
RUN_TIMEOUT = 60

requires_real_assets = pytest.mark.skipif(
    not (os.environ.get("AMD_KAGGLE_CHECKPOINT") and os.environ.get("AMD_KAGGLE_RSNA_ROOT")),
    reason="set AMD_KAGGLE_CHECKPOINT and AMD_KAGGLE_RSNA_ROOT to run the app's full pipeline",
)


@pytest.fixture
def no_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AMD_KAGGLE_RSNA_ROOT", raising=False)
    monkeypatch.delenv("AMD_KAGGLE_CHECKPOINT", raising=False)
    # This project's own real .env (repo root) already has AMD_KAGGLE_RSNA_ROOT/CHECKPOINT
    # set -- clearing the two env vars above isn't enough for a true "nothing configured"
    # scenario, since resolve_data_paths() would still load them from that real file. Point
    # the shared dotenv loader at an empty tmp_path instead, for this test only.
    import hackathon.amd.llm_config as llm_config_module

    monkeypatch.setattr(llm_config_module, "_DEFAULT_DOTENV_PATH", tmp_path / "does-not-exist.env")


@pytest.fixture
def real_env(monkeypatch):
    monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", os.environ.get("AMD_KAGGLE_RSNA_ROOT", ""))
    monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", os.environ.get("AMD_KAGGLE_CHECKPOINT", ""))


@pytest.fixture(autouse=True)
def _no_ambient_llm_credentials(monkeypatch):
    """This project's own shell has a real GROQ_API_KEY set -- never let any AppTest run in
    this file see it by accident; each test opts in to a fake key explicitly if it needs one."""
    for key in ("GROQ_API_KEY", "GROQ_MODEL", "AMD_TOKENFACTORY_BASE_URL", "AMD_TOKENFACTORY_API_KEY"):
        monkeypatch.delenv(key, raising=False)


class TestNoConfiguration:
    def test_app_loads_without_crashing(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception

    def test_prompts_for_data_root_and_checkpoint_and_shows_no_action_buttons(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert any("Enter a real data_root and checkpoint path" in info.value for info in at.info)
        assert "run_inference_button" not in [b.key for b in at.button]


class TestSyntheticDemoNeedsNoRealData:
    """This is also the Railway-readiness smoke test: with zero AMD_KAGGLE_*/real-data
    configuration (exactly the state of a fresh deploy before any real dataset/checkpoint is
    bundled or authorized), the app must still load cleanly and the synthetic demonstration
    path must work end to end -- real inference staying unavailable is expected, not a bug."""

    def test_app_loads_and_synthetic_section_is_expanded_with_zero_real_config(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert "synthetic_demo_button" in [b.key for b in at.button]

    def test_running_the_synthetic_demo_never_claims_a_real_bridged_checkpoint(self, no_env):
        """Regression test for a real bug found during development: the synthetic result
        briefly inherited agent_bridge_adapter's "externally trained Kaggle checkpoint"
        warning, which would have falsely implied a real checkpoint was used."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("SYNTHETIC WORKFLOW DEMONSTRATION" in e.value for e in at.error)
        assert not any("externally trained Kaggle checkpoint" in w.value for w in at.warning)
        assert any("no real neural-network inference occurred" in w.value for w in at.warning)

    def test_synthetic_demo_runs_the_full_evidence_and_review_workflow(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)

        buttons = {b.key for b in at.button}
        assert {"synthetic_approve_button", "synthetic_request_more_evidence_button", "synthetic_escalate_button"} <= buttons

        at.text_area(key="synthetic_reason_input").set_value("synthetic demo review").run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_approve_button").click().run(timeout=RUN_TIMEOUT)
        assert any("Recorded 'approve'" in s.value for s in at.success)

    def test_synthetic_demo_probabilities_are_deterministic_not_randomized(self):
        from hackathon.amd.synthetic_demo import build_synthetic_demo_result

        first = build_synthetic_demo_result().probabilities
        second = build_synthetic_demo_result().probabilities
        assert first == second

    def test_real_inference_path_is_unaffected_by_the_synthetic_section_existing(self, no_env):
        """The synthetic section must not consume the real-path gate or prevent the real
        "enter a data_root" prompt from still appearing when no real config is set."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert any("Enter a real data_root and checkpoint path" in info.value for info in at.info)


class TestCuratedBenchmarkScenarios:
    """The 22-scenario selector added to streamlit_app.py. Metadata browsing must work with
    zero real configuration; only the local-availability check and running inference need a
    real data_root/checkpoint, and neither ever runs automatically just from selecting a
    scenario."""

    def test_selector_appears_and_all_22_scenarios_load_with_zero_real_config(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        selector = at.selectbox(key="benchmark_scenario_select")
        assert len(selector.options) == 22
        assert selector.options[0].startswith("AMD-001 — ")
        assert selector.options[-1].startswith("AMD-022 — ")

    def test_disclaimer_is_shown(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert any("Research/demo benchmark only" in c.value for c in at.caption)

    def test_browsing_metadata_requires_no_data_root_no_checkpoint_no_credential_no_network(self, no_env, monkeypatch):
        """Metadata browsing must work with no data_root, no checkpoint, no AMD/Groq
        credential, and -- enforced here -- no network call possible at all."""
        import requests

        def _blocked(*args, **kwargs):
            raise AssertionError("requests.post was called just from browsing benchmark scenario metadata.")

        monkeypatch.setattr(requests, "post", _blocked)

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        selector = at.selectbox(key="benchmark_scenario_select")
        selector.set_value(selector.options[5]).run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Scenario ID:" in m.value for m in at.markdown)

    def test_selecting_a_scenario_never_runs_inference_automatically(self, no_env):
        """Selecting any scenario, with zero real config, must never attempt real inference --
        the only way to run inference is the explicit button, gated on local availability."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        selector = at.selectbox(key="benchmark_scenario_select")
        selector.set_value(selector.options[0]).run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert "benchmark_run_inference_button" not in [b.key for b in at.button]
        assert "last_result" not in at.session_state

    def test_no_data_root_shows_cannot_check_availability_not_a_crash(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        selector = at.selectbox(key="benchmark_scenario_select")
        selector.set_value(selector.options[0]).run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("cannot be checked yet" in i.value for i in at.info)

    @requires_real_assets
    def test_missing_underlying_study_shows_clean_informational_state(self, real_env):
        """With a real data_root configured, a scenario whose study is NOT present locally
        must show the exact clean message specified, never crash, never fabricate output."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        selector = at.selectbox(key="benchmark_scenario_select")
        # AMD-002 is not among the studies present in this session's real local data_root.
        amd_002 = next(o for o in selector.options if o.startswith("AMD-002 — "))
        selector.set_value(amd_002).run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any(
            "The underlying MRI study is not available on this machine, so live inference "
            "cannot be run for this scenario" in w.value
            for w in at.warning
        )
        assert "benchmark_run_inference_button" not in [b.key for b in at.button]

    @requires_real_assets
    def test_available_study_shows_success_and_offers_to_run_real_inference(self, real_env):
        """AMD-001's study is present in this session's real local data_root -- confirmed
        separately via mri_core.rsna_integration.discover_available_studies()."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        selector = at.selectbox(key="benchmark_scenario_select")
        amd_001 = next(o for o in selector.options if o.startswith("AMD-001 — "))
        selector.set_value(amd_001).run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("is available locally under the configured data_root" in s.value for s in at.success)
        assert "benchmark_run_inference_button" in [b.key for b in at.button]

    @requires_real_assets
    def test_running_inference_from_a_scenario_reuses_the_existing_real_inference_path(self, real_env):
        """Clicking the benchmark's run-inference button must populate the exact same
        session-state the main real-inference path reads from -- proving this is the same
        underlying call, not a second, duplicate inference implementation."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        selector = at.selectbox(key="benchmark_scenario_select")
        amd_001 = next(o for o in selector.options if o.startswith("AMD-001 — "))
        selector.set_value(amd_001).run(timeout=RUN_TIMEOUT)
        at.button(key="benchmark_run_inference_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Real inference complete" in s.value for s in at.success)
        # The main real-inference panel (rendered by the same render_workflow() used
        # everywhere else) is now showing this scenario's result.
        assert any("real_approve_button" == b.key for b in at.button)


class TestNoConfiguration:
    def test_amd_token_factory_is_unchecked_by_default(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert at.checkbox(key="amd_enabled_checkbox").value is False
        assert any("disabled by default" in c.value for c in at.caption)


class TestDataPathPrefillAndValidation:
    """Always runs -- these don't need real RSNA assets, only existence/extension checks."""

    def test_sidebar_is_prefilled_from_environment_variables(self, monkeypatch, tmp_path):
        root = tmp_path / "rsna-knee"
        root.mkdir()
        (root / "train.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
        checkpoint = tmp_path / "fold0-knee-mil-best.ckpt"
        checkpoint.write_bytes(b"not real weights")

        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", str(root))
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", str(checkpoint))

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert at.text_input(key="data_root_input").value == str(root)
        assert at.text_input(key="checkpoint_input").value == str(checkpoint)

    def test_missing_environment_variables_leave_sidebar_empty(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert at.text_input(key="data_root_input").value == ""
        assert at.text_input(key="checkpoint_input").value == ""

    def test_invalid_data_directory_shows_a_clean_error_without_crashing(self, monkeypatch, tmp_path):
        missing_root = tmp_path / "does-not-exist"
        checkpoint = tmp_path / "fold0-knee-mil-best.ckpt"
        checkpoint.write_bytes(b"not real weights")

        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", str(missing_root))
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", str(checkpoint))

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Data directory not found" in e.value for e in at.error)
        assert "run_inference_button" not in [b.key for b in at.button]

    def test_missing_checkpoint_shows_a_clean_error_without_crashing(self, monkeypatch, tmp_path):
        root = tmp_path / "rsna-knee"
        root.mkdir()
        (root / "train.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
        missing_checkpoint = tmp_path / "does-not-exist.ckpt"

        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", str(root))
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", str(missing_checkpoint))

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Checkpoint file not found" in e.value for e in at.error)
        assert "run_inference_button" not in [b.key for b in at.button]

    def test_wrong_checkpoint_extension_shows_a_clean_error_without_crashing(self, monkeypatch, tmp_path):
        root = tmp_path / "rsna-knee"
        root.mkdir()
        (root / "train.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
        wrong_ext_checkpoint = tmp_path / "checkpoint.pt"
        wrong_ext_checkpoint.write_bytes(b"not real weights")

        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", str(root))
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", str(wrong_ext_checkpoint))

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any(".ckpt" in e.value and "extension" in e.value for e in at.error)
        assert "run_inference_button" not in [b.key for b in at.button]

    def test_missing_dataset_metadata_shows_a_clean_error_without_crashing(self, monkeypatch, tmp_path):
        root = tmp_path / "rsna-knee"
        root.mkdir()  # no train.csv
        checkpoint = tmp_path / "fold0-knee-mil-best.ckpt"
        checkpoint.write_bytes(b"not real weights")

        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", str(root))
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", str(checkpoint))

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("train.csv" in e.value for e in at.error)
        assert "run_inference_button" not in [b.key for b in at.button]


@requires_real_assets
class TestManualSidebarOverride:
    def test_manual_override_takes_precedence_over_environment_default(self, real_env, monkeypatch):
        """Set a bogus env default, then manually type the real paths into the sidebar --
        the app must use the typed values, not the environment default."""
        real_data_root = os.environ["AMD_KAGGLE_RSNA_ROOT"]
        real_checkpoint = os.environ["AMD_KAGGLE_CHECKPOINT"]
        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", "this-env-default-is-intentionally-wrong")
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", "this-env-default-is-intentionally-wrong.ckpt")

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert any("Data directory not found" in e.value for e in at.error)

        at.text_input(key="data_root_input").set_value(real_data_root).run(timeout=RUN_TIMEOUT)
        at.text_input(key="checkpoint_input").set_value(real_checkpoint).run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert not any("not found" in e.value for e in at.error)
        assert len(at.button) > 0


@requires_real_assets
class TestFullPipeline:
    def test_run_inference_shows_scores_provenance_and_warnings(self, real_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("not a diagnostic device" in e.value for e in at.error)
        assert any("externally trained Kaggle checkpoint" in w.value for w in at.warning)
        # Review-gate starts PENDING_REVIEW and all three actions are available immediately.
        buttons = {b.key: b for b in at.button}
        assert set(buttons) >= {"real_approve_button", "real_request_more_evidence_button", "real_escalate_button"}
        assert not buttons["real_approve_button"].disabled

    def test_review_action_without_reason_is_rejected(self, real_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="real_approve_button").click().run(timeout=RUN_TIMEOUT)
        assert any("A reason is required" in e.value for e in at.error)
        # The gate must still be PENDING_REVIEW -- the human-approval gate was not bypassed.
        assert not at.button(key="real_approve_button").disabled

    def test_approve_records_audit_entry_and_closes_the_gate(self, real_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.text_area(key="real_reason_input").set_value("evidence looks complete").run(timeout=RUN_TIMEOUT)
        at.button(key="real_approve_button").click().run(timeout=RUN_TIMEOUT)

        assert any("Recorded 'approve'" in s.value for s in at.success)
        rows = at.table[-1].value
        assert rows.iloc[-1]["action"] == "reviewed"
        assert rows.iloc[-1]["to"] == "REVIEWED"

        # One more rerun to see the now-terminal state reflected in the buttons.
        at.run(timeout=RUN_TIMEOUT)
        assert at.button(key="real_approve_button").disabled
        assert at.button(key="real_escalate_button").disabled
        assert at.button(key="real_request_more_evidence_button").disabled

    def test_escalate_is_distinguishable_from_approve_in_the_audit_trail(self, real_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.text_area(key="real_reason_input").set_value("needs senior review").run(timeout=RUN_TIMEOUT)
        at.button(key="real_escalate_button").click().run(timeout=RUN_TIMEOUT)

        rows = at.table[-1].value
        last_row = rows.iloc[-1]
        assert last_row["action"] == "escalate"
        assert last_row["reason"].startswith("ESCALATED:")

    def test_benchmark_panel_never_claims_amd_gpu_without_a_real_measured_file(self, real_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.text_input(key="benchmark_path_input").set_value("outputs/amd/__does_not_exist__.json").run(timeout=RUN_TIMEOUT)
        assert any("not available until measured" in i.value for i in at.info)


@requires_real_assets
class TestOptionalLLMSection:
    def test_default_provider_is_local_and_produces_a_quarantined_result(self, real_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Produced by: local" in c.value for c in at.caption)
        assert any(w.value.startswith("QUARANTINED") for w in at.warning)

    def test_ambient_groq_api_key_alone_does_not_authorize_a_call(self, real_env, monkeypatch):
        """Regression test for the exact incident this feature fixes: a real-looking
        GROQ_API_KEY sitting in the environment, with no LLM_PROVIDER/LLM_ALLOW_EXTERNAL_CALLS
        set, must never result in a live call -- even if the user picks "Groq" in the UI."""
        import requests

        def _blocked(*args, **kwargs):
            raise AssertionError("requests.post was called; an ambient GROQ_API_KEY alone must never authorize this.")

        monkeypatch.setenv("GROQ_API_KEY", "sk-fake-looking-real-key-123456")
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_ALLOW_EXTERNAL_CALLS", raising=False)
        monkeypatch.setattr(requests, "post", _blocked)

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.radio(key="real_llm_provider_choice").set_value("Groq (requires GROQ_API_KEY)").run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Produced by: local" in c.value for c in at.caption)

    def test_groq_without_env_opt_in_falls_back_to_local_with_no_network_call(self, real_env, monkeypatch):
        import requests

        def _blocked(*args, **kwargs):
            raise AssertionError("requests.post was called with no GROQ_API_KEY set.")

        monkeypatch.setattr(requests, "post", _blocked)

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.radio(key="real_llm_provider_choice").set_value("Groq (requires GROQ_API_KEY)").run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("local" in c.value for c in at.caption)

    def test_groq_mocked_success_requires_full_env_opt_in(self, real_env, monkeypatch):
        import json as _json

        import requests

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {
                    "content": _json.dumps({"summary": "Deterministic workflow summary only."}),
                }}]}

        monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.radio(key="real_llm_provider_choice").set_value("Groq (requires GROQ_API_KEY)").run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Produced by: groq" in c.value for c in at.caption)
        assert any("Deterministic workflow summary only." in m.value for m in at.markdown)

    def test_amd_token_factory_falls_back_to_local_when_misconfigured(self, real_env, monkeypatch):
        """Missing AMD_TOKENFACTORY_MODEL (one of the four required fields) must route to
        local without any network call -- build_adapter() itself never constructs a
        live-capable AMDTokenFactoryLLMAdapter in this case."""
        import requests

        def _blocked(*args, **kwargs):
            raise AssertionError("requests.post was called despite AMD being misconfigured.")

        monkeypatch.setattr(requests, "post", _blocked)
        monkeypatch.setenv("LLM_PROVIDER", "amd_token_factory")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_ENABLED", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
        monkeypatch.setenv("AMD_TOKENFACTORY_API_KEY", "fake-amd-key")
        # Explicitly blanked, not just left out of monkeypatch: the real .env on this machine
        # already has AMD_TOKENFACTORY_MODEL set, and load_dotenv's override=False would fill
        # it in from that file for any variable merely absent from os.environ.
        monkeypatch.setenv("AMD_TOKENFACTORY_MODEL", "")

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.radio(key="real_llm_provider_choice").set_value("AMD Token Factory (LLM reasoning, requires .env config)").run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Produced by: local" in c.value for c in at.caption)
        assert any("not all set" in c.value for c in at.caption)

    def test_amd_token_factory_mocked_success_requires_full_env_opt_in(self, real_env, monkeypatch):
        import json as _json

        import requests

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {
                    "content": _json.dumps({"summary": "Mocked AMD Token Factory summary."}),
                }}]}

        monkeypatch.setenv("LLM_PROVIDER", "amd_token_factory")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_ENABLED", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
        monkeypatch.setenv("AMD_TOKENFACTORY_API_KEY", "fake-amd-key")
        monkeypatch.setenv("AMD_TOKENFACTORY_MODEL", "Qwen3.8-27B")
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.radio(key="real_llm_provider_choice").set_value("AMD Token Factory (LLM reasoning, requires .env config)").run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Produced by: amd_token_factory" in c.value for c in at.caption)
        assert any("Mocked AMD Token Factory summary." in m.value for m in at.markdown)

    def test_amd_token_factory_mocked_failure_falls_back_to_local_not_groq(self, real_env, monkeypatch):
        """Even with Groq also fully configured in the environment, an AMD failure must fall
        back to local, never silently try Groq."""
        import requests

        class _FakeErrorResponse:
            status_code = 500
            text = "internal error"

        monkeypatch.setenv("LLM_PROVIDER", "amd_token_factory")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_ENABLED", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
        monkeypatch.setenv("AMD_TOKENFACTORY_API_KEY", "fake-amd-key")
        monkeypatch.setenv("AMD_TOKENFACTORY_MODEL", "Qwen3.8-27B")
        monkeypatch.setenv("GROQ_API_KEY", "also-set-but-must-not-be-used")
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeErrorResponse())

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.radio(key="real_llm_provider_choice").set_value("AMD Token Factory (LLM reasoning, requires .env config)").run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Produced by: local (fallback:" in c.value for c in at.caption)

    def test_llm_section_never_changes_review_gate_state(self, real_env):
        """The LLM interpretation must be structurally separate from the human-approval gate:
        generating one must not move the gate out of PENDING_REVIEW or disable its buttons."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.button(key="real_approve_button").disabled
        assert not at.button(key="real_escalate_button").disabled
        assert not at.button(key="real_request_more_evidence_button").disabled
        assert any("No review actions recorded yet." in c.value for c in at.caption)
