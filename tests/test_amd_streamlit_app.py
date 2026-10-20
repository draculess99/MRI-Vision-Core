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
    # Explicit defaults (the real .env on a developer machine may enable a live provider); a test that
    # needs Groq enabled sets these itself and mocks requests.post.
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "false")


@pytest.fixture(autouse=True)
def _opt_out_of_public_demo_mode(monkeypatch):
    """Existing tests below cover the metadata-only/real-inference behavior; public synthetic-demo
    mode (tests/test_amd_public_demo.py) is opted out of here so they stay unchanged."""
    monkeypatch.setenv("AMD_PUBLIC_DEMO_MODE", "false")


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
        assert "run_inference_button" not in [b.key for b in at.button]
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
        # Not pre-selected in the single "Run trained inference" selectbox below, since this
        # scenario's study is not among the locally available ones.
        assert at.selectbox(key="study_select").value != "1.2.826.0.1.3680043.8.498.69392348385274125385290015404144639002"

    @requires_real_assets
    def test_available_study_shows_success_and_offers_to_run_real_inference(self, real_env):
        """AMD-001's study is present in this session's real local data_root -- confirmed
        separately via mri_core.rsna_integration.discover_available_studies(). There is exactly
        one inference button on the page (no separate benchmark-section button), and it is
        pre-selected to this scenario's study."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        selector = at.selectbox(key="benchmark_scenario_select")
        amd_001 = next(o for o in selector.options if o.startswith("AMD-001 — "))
        selector.set_value(amd_001).run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("is available locally under the configured data_root" in s.value for s in at.success)
        inference_buttons = [b for b in at.button if "inference" in (b.label or "").lower()]
        assert len(inference_buttons) == 1
        assert inference_buttons[0].key == "run_inference_button"
        assert inference_buttons[0].label == "Run trained inference on selected MRI study"
        assert at.selectbox(key="study_select").value == "1.2.826.0.1.3680043.8.498.11287937729196958426538087439102017580"

    @requires_real_assets
    def test_running_inference_from_a_scenario_reuses_the_existing_real_inference_path(self, real_env):
        """Clicking the single, shared inference button after selecting a scenario must run
        inference on THAT scenario's pre-selected study and populate the exact same
        session-state the main real-inference path reads from -- proving this is the same
        underlying call, not a second, duplicate inference implementation."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        selector = at.selectbox(key="benchmark_scenario_select")
        amd_001 = next(o for o in selector.options if o.startswith("AMD-001 — "))
        selector.set_value(amd_001).run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        state = at.session_state["last_result"]
        assert state["study_uid"] == "1.2.826.0.1.3680043.8.498.11287937729196958426538087439102017580"
        # The main real-inference panel (rendered by the same render_workflow() used
        # everywhere else) is now showing this scenario's result.
        assert any("real_approve_button" == b.key for b in at.button)


class TestNoConfiguration:
    def test_hardware_execution_sidebar_block_is_not_shown(self, no_env):
        """The redundant "Execution provider (hardware, not AI text)" sidebar block (an
        unimplemented status/UI seam, unrelated to the real, working LLM provider integration)
        was removed as judge-facing UI cleanup -- it never dispatched inference to AMD hardware
        and only duplicated/confused the actual 'Optional LLM interpretation' section below."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert "amd_enabled_checkbox" not in [cb.key for cb in at.checkbox]
        assert not any("Execution provider" in h.value for h in at.header)
        assert not any("hardware execution" in c.value for c in at.caption)


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
        assert any("LLM interpretation generated — human review required" in i.value for i in at.info)
        assert any(c.value.startswith("Status: QUARANTINED") for c in at.caption)

    def test_ambient_groq_api_key_alone_does_not_authorize_a_call(self, real_env, monkeypatch):
        """Regression test for the exact incident this feature fixes: a real-looking
        GROQ_API_KEY sitting in the environment, with no LLM_PROVIDER/LLM_ALLOW_EXTERNAL_CALLS
        set, must never result in a live call -- even if the user picks "Groq" in the UI."""
        import requests

        def _blocked(*args, **kwargs):
            raise AssertionError("requests.post was called; an ambient GROQ_API_KEY alone must never authorize this.")

        monkeypatch.setenv("GROQ_API_KEY", "sk-fake-looking-real-key-123456")
        # Blank (not deleted): a real .env on a developer machine would otherwise fill these back in.
        monkeypatch.setenv("LLM_PROVIDER", "")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "")
        monkeypatch.setattr(requests, "post", _blocked)

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
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
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("local" in c.value for c in at.caption)

    def test_groq_mocked_success_requires_full_env_opt_in(self, real_env, monkeypatch):
        import json as _json

        import requests

        structured_fields = {
            "specialist_summary": "Specialist grouped 12 scores into 4 categories.",
            "critic_summary": "Critic found 1 issue: QUALITY_NOT_ASSESSED.",
            "consensus_explanation": "Consensus assigned MORE_EVIDENCE_REQUIRED from that finding.",
            "reviewer_focus": "Complete the quality assessment before approval.",
        }

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": _json.dumps(structured_fields)}}]}

        monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse())

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)

        assert not at.exception
        assert any("Produced by: groq" in c.value for c in at.caption)
        # All four structured sections render, each under its own heading.
        headings = [h.value for h in at.markdown if h.value.startswith("#####")]
        assert any("Specialist explanation" in h for h in headings)
        assert any("Critic explanation" in h for h in headings)
        assert any("Consensus explanation" in h for h in headings)
        assert any("Reviewer focus" in h for h in headings)
        for text in structured_fields.values():
            assert any(text in m.value for m in at.markdown)

    def test_amd_token_factory_is_not_offered_and_never_called(self, real_env, monkeypatch):
        """The AMD adapter code is retained, but it is not part of the active provider path: the UI
        offers only Groq and Local, and an AMD environment selection never causes a network call."""
        import requests

        def _blocked(*args, **kwargs):
            raise AssertionError("requests.post must never be called for an AMD Token Factory selection.")

        monkeypatch.setattr(requests, "post", _blocked)
        monkeypatch.setenv("LLM_PROVIDER", "amd_token_factory")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_ENABLED", "true")
        monkeypatch.setenv("AMD_TOKENFACTORY_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1")
        monkeypatch.setenv("AMD_TOKENFACTORY_API_KEY", "fake-amd-key")
        monkeypatch.setenv("AMD_TOKENFACTORY_MODEL", "Qwen3.8-27B")

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.radio  # no provider choice at all: Groq is the only live provider offered
        assert [b.label for b in at.button if b.key == "real_llm_generate_button"] == ["Generate live Groq explanation"]
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Local deterministic fallback" in w.value for w in at.warning)
        assert any("Groq is not enabled" in m.value for m in at.markdown)

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


class TestGroqIsTheOnlyLiveProvider:
    """Groq is the default and only network LLM provider; every outcome is labelled honestly.
    Synthetic path, all network mocked -- no real data, no real key."""

    FIELDS = {
        "specialist_summary": "Groq specialist text.", "critic_summary": "Groq critic text.",
        "consensus_explanation": "Groq consensus text.", "reviewer_focus": "Groq reviewer focus text.",
    }

    @staticmethod
    def _enable_groq(monkeypatch, model="openai/gpt-oss-20b"):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key-for-tests")
        monkeypatch.setenv("GROQ_MODEL", model)

    def _run(self, monkeypatch, response_cls, *, captured=None):
        import requests

        def fake_post(url, headers=None, json=None, timeout=None):
            if captured is not None:
                captured.append({"url": url, "json": json})
            return response_cls()

        monkeypatch.setattr(requests, "post", fake_post)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        return at

    def _ok(self):
        import json as _json

        fields = self.FIELDS

        class _Ok:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": _json.dumps(fields), "reasoning": "hidden"}}]}

        return _Ok

    def test_there_is_no_provider_choice_widget_and_groq_wording_follows_LLM_PROVIDER_groq(self, no_env, monkeypatch):
        self._enable_groq(monkeypatch)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.radio
        assert [b.label for b in at.button if b.key == "synthetic_llm_generate_button"] == ["Generate live Groq explanation"]

    def test_local_provider_uses_accurate_local_wording_and_never_mentions_groq_or_amd(self, no_env, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        assert [b.label for b in at.button if b.key == "synthetic_llm_generate_button"] == ["Generate local deterministic explanation"]
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Local deterministic explanation (no network, no AI model)." in i.value for i in at.info)
        assert not any("Groq is not enabled" in m.value for m in at.markdown)
        assert not any("fallback" in w.value for w in at.warning)

    def test_default_selection_goes_live_with_groq_when_enabled(self, no_env, monkeypatch):
        self._enable_groq(monkeypatch)
        captured = []
        at = self._run(monkeypatch, self._ok(), captured=captured)  # no radio change: the default
        assert len(captured) == 1 and "groq" in captured[0]["url"]
        assert any("Live Groq explanation" in s.value and "openai/gpt-oss-20b" in s.value for s in at.success)
        assert any("Produced by: groq" in c.value for c in at.caption)
        assert not at.warning or not any("Local deterministic fallback" in w.value for w in at.warning)

    def test_valid_output_shows_all_four_sections(self, no_env, monkeypatch):
        self._enable_groq(monkeypatch, model="test-model-x")
        at = self._run(monkeypatch, self._ok())
        assert any("Live Groq explanation — model: test-model-x" in s.value for s in at.success)
        headings = [m.value for m in at.markdown if m.value.startswith("#####")]
        for name in ("Specialist explanation", "Critic explanation", "Consensus explanation", "Reviewer focus"):
            assert any(name in h for h in headings)
        for text in self.FIELDS.values():
            assert any(text in m.value for m in at.markdown)
        assert not any("hidden" in m.value for m in at.markdown)  # reasoning field never shown

    def test_the_current_deterministic_state_reaches_groq_through_the_allowlist(self, no_env, monkeypatch):
        import json as _json

        from hackathon.amd.llm_providers import _ALLOWED_KEYS
        from hackathon.amd.synthetic_demo import SYNTHETIC_CHECKPOINT_PATH, SYNTHETIC_STUDY_UID, build_synthetic_demo_result
        from mri_core.agents.contracts import QualityGateStatus
        from mri_core.agents.orchestrator import process_inference_result

        self._enable_groq(monkeypatch)
        captured = []
        self._run(monkeypatch, self._ok(), captured=captured)
        workflow = process_inference_result(build_synthetic_demo_result(), quality_gate_status=QualityGateStatus.NOT_ASSESSED)
        sent = _json.loads(captured[0]["json"]["messages"][1]["content"])
        assert set(sent) <= _ALLOWED_KEYS
        assert sent["workflow_status"] == workflow.consensus.workflow_status.value
        assert sent["critic_codes"] == [f.code for f in workflow.critic_findings]
        assert sent["critic_severities"] == [f.severity.value for f in workflow.critic_findings]
        assert sent["quality_gate_status"] == workflow.bundle.quality_gate_status.value
        assert sent["is_ensemble"] is workflow.bundle.is_ensemble and sent["model_count"] == workflow.bundle.model_count
        assert sent["limitations"] == list(workflow.bundle.limitations)
        assert [g["group"] for g in sent["specialist_groups"]] == [s.group_name for s in workflow.specialist_summaries]
        raw_request = _json.dumps(captured[0]["json"])
        for forbidden in (SYNTHETIC_STUDY_UID, SYNTHETIC_CHECKPOINT_PATH, "fake-groq-key-for-tests"):
            assert forbidden not in raw_request

    @pytest.mark.parametrize("kind", ["http_error", "invalid_json", "empty_content", "wrong_schema"])
    def test_any_groq_failure_is_labelled_local_deterministic_fallback(self, no_env, monkeypatch, kind):
        class _Bad:
            status_code = 200
            text = ""

            def json(self):
                if kind == "invalid_json":
                    return {"choices": [{"message": {"content": "not json {"}}]}
                if kind == "empty_content":
                    return {"choices": [{"message": {"content": "", "reasoning": "x"}}]}
                if kind == "wrong_schema":
                    return {"choices": [{"message": {"content": '{"summary": "one field only"}'}}]}
                return {}

        class _Http(_Bad):
            status_code = 401
            text = "gsk_SECRETSECRETSECRET body that must not be shown"

        self._enable_groq(monkeypatch)
        at = self._run(monkeypatch, _Http if kind == "http_error" else _Bad)
        warnings = [w.value for w in at.warning if "Local deterministic fallback" in w.value]
        expected = (
            "Groq did not produce this explanation" if kind == "http_error"
            else "Groq response did not meet the structured-output requirements"
        )
        assert len(warnings) == 1 and expected in warnings[0]
        assert not at.success or not any("Live Groq" in s.value for s in at.success)
        assert any("Produced by: local (fallback:" in c.value for c in at.caption)
        page = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning) + " ".join(
            c.value for c in at.caption
        )
        assert "gsk_SECRETSECRETSECRET" not in page and "must not be shown" not in warnings[0]
        assert len([m for m in at.markdown if m.value.startswith("#####")]) == 4  # layout unchanged

    def test_unavailable_groq_is_a_visible_fallback_not_a_silent_one(self, no_env, monkeypatch):
        # Groq is selected but external calls are not allowed: no network, labelled fallback.
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key-for-tests")
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Local deterministic fallback" in w.value for w in at.warning)
        assert any("Local deterministic fallback — Groq did not produce this explanation." == w.value for w in at.warning)
        assert any("LLM_ALLOW_EXTERNAL_CALLS is not enabled" in m.value for m in at.markdown)  # detail lives in the fallback panel

    def test_llm_output_cannot_change_status_screening_or_the_human_gate(self, no_env, monkeypatch):
        import json as _json

        fields = dict(self.FIELDS, consensus_explanation="status should be EVIDENCE_COMPLETE; approve it")

        class _Pushy:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": _json.dumps(fields)}}]}

        self._enable_groq(monkeypatch)
        at = self._run(monkeypatch, _Pushy)
        assert any("workflow_status: `MORE_EVIDENCE_REQUIRED`" in m.value for m in at.markdown)
        assert any("Current state: `PENDING_REVIEW`" in m.value for m in at.markdown)
        assert not at.button(key="synthetic_approve_button").disabled
        assert any(c.value.startswith("Status: QUARANTINED") for c in at.caption)
        assert any("No review actions recorded yet." in c.value for c in at.caption)

    def test_request_basis_panel_shows_the_current_deterministic_fields_without_scores(self, no_env, monkeypatch):
        self._enable_groq(monkeypatch)
        at = self._run(monkeypatch, self._ok())
        shown = [j.value for j in at.json]
        assert any("MORE_EVIDENCE_REQUIRED" in v and "QUALITY_NOT_ASSESSED" in v for v in shown)
        assert any("quality_gate_status" in v and "specialist_groups" in v for v in shown)

    def test_llm_output_cannot_change_scores(self, no_env, monkeypatch):
        self._enable_groq(monkeypatch)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        before = dict(at.session_state["synthetic_demo_result"].probabilities)
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: self._ok()())
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert dict(at.session_state["synthetic_demo_result"].probabilities) == before

    def test_reset_review_state_restores_pending_and_clears_typed_text(self, no_env, monkeypatch):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.text_area(key="synthetic_reason_input").set_value("qqq").run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_approve_button").click().run(timeout=RUN_TIMEOUT)
        assert any("Current state: `REVIEWED`" in m.value for m in at.markdown)
        assert not any("No review actions recorded yet." in c.value for c in at.caption)  # audit row exists

        at.button(key="synthetic_reset_review_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Current state: `PENDING_REVIEW`" in m.value for m in at.markdown)
        assert at.text_area(key="synthetic_reason_input").value == ""
        assert not at.button(key="synthetic_approve_button").disabled
        assert any("No review actions recorded yet." in c.value for c in at.caption)

    def test_generating_an_llm_explanation_never_resets_or_advances_review_state(self, no_env, monkeypatch):
        self._enable_groq(monkeypatch)
        import requests

        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.text_area(key="synthetic_reason_input").set_value("kept").run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_request_more_evidence_button").click().run(timeout=RUN_TIMEOUT)
        monkeypatch.setattr(requests, "post", lambda *a, **k: self._ok()())
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert any("Current state: `MORE_EVIDENCE_REQUESTED`" in m.value for m in at.markdown)
        assert not any("No review actions recorded yet." in c.value for c in at.caption)


class TestSyntheticBannerOnlyForSyntheticFlow:
    SYNTH_LABEL = "Synthetic workflow demonstration"

    def test_synthetic_panel_is_shown_when_no_real_inference_is_available(self, no_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert any(self.SYNTH_LABEL in e.label for e in at.expander)
        assert "synthetic_demo_button" in [b.key for b in at.button]

    def test_synthetic_panel_is_hidden_for_valid_real_paths(self, monkeypatch, tmp_path):
        root = tmp_path / "rsna"
        root.mkdir()
        (root / "train.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
        ckpt = tmp_path / "w.ckpt"
        ckpt.write_bytes(b"x")
        monkeypatch.setenv("AMD_KAGGLE_RSNA_ROOT", str(root))
        monkeypatch.setenv("AMD_KAGGLE_CHECKPOINT", str(ckpt))
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        assert not any(self.SYNTH_LABEL in e.label for e in at.expander)
        assert "synthetic_demo_button" not in [b.key for b in at.button]
        assert not any("fabricated placeholder" in w.value for w in at.warning)

    @requires_real_assets
    def test_real_local_inference_is_labelled_local_trained_inference(self, real_env):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("Local trained inference" in i.value for i in at.info)
        assert not any(self.SYNTH_LABEL in e.label for e in at.expander)
        assert not any("fabricated placeholder" in w.value for w in at.warning)

    @requires_real_assets
    def test_groq_request_for_a_real_run_derives_from_that_real_runs_review(self, real_env, monkeypatch):
        import json as _json

        import requests

        captured = []

        class _Ok:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": _json.dumps({
                    "specialist_summary": "s", "critic_summary": "c", "consensus_explanation": "e", "reviewer_focus": "r",
                })}}]}

        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setattr(requests, "post", lambda url, headers=None, json=None, timeout=None: (captured.append(json), _Ok())[1])
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="run_inference_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="real_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        sent = _json.loads(captured[0]["messages"][1]["content"])
        state = at.session_state["last_result"]
        assert sent["workflow_status"] in (
            "MORE_EVIDENCE_REQUIRED", "EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW", "INCONSISTENT_EVIDENCE", "INFERENCE_UNAVAILABLE",
        )
        assert "ENGINE_WARNINGS_PRESENT" in sent["critic_codes"]  # the bridged-checkpoint warning of THIS real run
        assert state["study_uid"] not in _json.dumps(captured[0])
        assert any("Live Groq explanation" in s.value for s in at.success)


class TestReviewStateUpdatesImmediately:
    """The state line, buttons and audit table must reflect an action in the SAME rerun as the click."""

    @staticmethod
    def _open(no_env_unused=None):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        return at

    @staticmethod
    def _state(at):
        return next(m.value for m in at.markdown if "Current state:" in m.value)

    def test_approve_shows_reviewed_and_the_audit_row_immediately(self, no_env):
        at = self._open()
        assert "PENDING_REVIEW" in self._state(at)
        at.text_area(key="synthetic_reason_input").set_value("Synthetic walkthrough").run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_approve_button").click().run(timeout=RUN_TIMEOUT)  # one click, one rerun
        assert "REVIEWED" in self._state(at)
        assert at.button(key="synthetic_approve_button").disabled
        assert any("Recorded 'approve' by demo_reviewer." in s.value for s in at.success)
        assert not any("No review actions recorded yet." in c.value for c in at.caption)
        audit = [t for t in at.table if "reason" in list(t.value.columns)]
        assert len(audit) == 1 and list(audit[0].value["to"]) == ["REVIEWED"]

    def test_escalate_updates_state_and_audit_immediately(self, no_env):
        at = self._open()
        at.text_area(key="synthetic_reason_input").set_value("Needs a senior reviewer").run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_escalate_button").click().run(timeout=RUN_TIMEOUT)
        assert "REVIEWED" in self._state(at)  # an escalation is recorded as a reviewed outcome with an escalate action
        audit = [t for t in at.table if "reason" in list(t.value.columns)]
        assert len(audit) == 1 and list(audit[0].value["action"]) == ["escalate"]
        assert any("Recorded 'escalate'" in s.value for s in at.success)

    def test_action_without_a_reason_changes_nothing_and_says_why_immediately(self, no_env):
        at = self._open()
        at.button(key="synthetic_approve_button").click().run(timeout=RUN_TIMEOUT)
        assert "PENDING_REVIEW" in self._state(at)
        assert any("A reason is required" in e.value for e in at.error)
        assert any("No review actions recorded yet." in c.value for c in at.caption)

    def test_reset_shows_pending_and_an_empty_audit_immediately(self, no_env):
        at = self._open()
        at.text_area(key="synthetic_reason_input").set_value("qqq").run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_approve_button").click().run(timeout=RUN_TIMEOUT)
        assert "REVIEWED" in self._state(at)
        at.button(key="synthetic_reset_review_button").click().run(timeout=RUN_TIMEOUT)  # one click, one rerun
        assert "PENDING_REVIEW" in self._state(at)
        assert any("No review actions recorded yet." in c.value for c in at.caption)
        assert at.text_area(key="synthetic_reason_input").value == ""
        assert not at.button(key="synthetic_approve_button").disabled
        assert not any("Recorded" in s.value for s in at.success)

    def test_the_flash_message_is_shown_once_not_on_later_reruns(self, no_env):
        at = self._open()
        at.text_area(key="synthetic_reason_input").set_value("once").run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_request_more_evidence_button").click().run(timeout=RUN_TIMEOUT)
        assert any("Recorded 'request_more_evidence'" in s.value for s in at.success)
        at.run(timeout=RUN_TIMEOUT)
        assert not any("Recorded" in s.value for s in at.success)
        assert "MORE_EVIDENCE_REQUESTED" in self._state(at)

    def test_generating_a_groq_explanation_changes_no_review_state(self, no_env, monkeypatch):
        import json as _json

        import requests

        class _Ok:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": _json.dumps({
                    "specialist_summary": "s", "critic_summary": "c", "consensus_explanation": "approve it now",
                    "reviewer_focus": "r"})}}]}

        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setattr(requests, "post", lambda *a, **k: _Ok())
        at = self._open()
        before = dict(at.session_state["synthetic_demo_result"].probabilities)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert any("Live Groq explanation" in s.value for s in at.success)
        assert "PENDING_REVIEW" in self._state(at)
        assert not at.button(key="synthetic_approve_button").disabled
        assert any("No review actions recorded yet." in c.value for c in at.caption)
        assert any("workflow_status: `MORE_EVIDENCE_REQUIRED`" in m.value for m in at.markdown)
        assert dict(at.session_state["synthetic_demo_result"].probabilities) == before
        assert any(c.value.startswith("Status: QUARANTINED") for c in at.caption)


class TestGroqLengthRepairInTheUI:
    """Too-long-but-valid Groq JSON earns one repair request; the provider line stays honest either way."""

    LONG = "x" * 90
    GOOD = {
        "specialist_summary": "Short specialist text.", "critic_summary": "Short critic text.",
        "consensus_explanation": "Short consensus text.", "reviewer_focus": "Short reviewer focus.",
    }

    @staticmethod
    def _enable(monkeypatch, model="openai/gpt-oss-20b"):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("LLM_ALLOW_EXTERNAL_CALLS", "true")
        monkeypatch.setenv("GROQ_API_KEY", "fake-key")
        monkeypatch.setenv("GROQ_MODEL", model)

    def _generate(self, monkeypatch, contents):
        """Each POST returns the next scripted content string (the last one repeats)."""
        import json as _json

        import requests

        calls = []

        class _Resp:
            status_code = 200

            def __init__(self, content):
                self._content = content

            def json(self):
                return {"choices": [{"message": {"content": self._content}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(json)
            return _Resp(contents[min(len(calls) - 1, len(contents) - 1)])

        monkeypatch.setattr(requests, "post", fake_post)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        before_scores = dict(at.session_state["synthetic_demo_result"].probabilities)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert dict(at.session_state["synthetic_demo_result"].probabilities) == before_scores
        return at, calls, _json

    def test_too_long_then_valid_repair_is_shown_as_a_live_groq_explanation(self, no_env, monkeypatch):
        import json as _json

        self._enable(monkeypatch, model="repair-model")
        too_long = _json.dumps(dict(self.GOOD, specialist_summary=self.LONG))
        at, calls, _ = self._generate(monkeypatch, [too_long, _json.dumps(self.GOOD)])
        assert len(calls) == 2
        assert any("Live Groq explanation — model: repair-model" in s.value for s in at.success)
        assert not any("Local deterministic fallback" in w.value for w in at.warning)
        assert any("Produced by: groq" in c.value for c in at.caption)
        for text in self.GOOD.values():
            assert any(text in m.value for m in at.markdown)
        assert any("workflow_status: `MORE_EVIDENCE_REQUIRED`" in m.value for m in at.markdown)
        assert any("Current state: `PENDING_REVIEW`" in m.value for m in at.markdown)
        assert any(c.value.startswith("Status: QUARANTINED") for c in at.caption)

    def test_still_too_long_after_repair_shows_the_structured_output_fallback_line(self, no_env, monkeypatch):
        import json as _json

        self._enable(monkeypatch)
        at, calls, _ = self._generate(monkeypatch, [_json.dumps(dict(self.GOOD, specialist_summary=self.LONG))])
        assert len(calls) == 2  # original + one repair, never more
        assert not at.success or not any("Live Groq" in s.value for s in at.success)
        banner = [w.value for w in at.warning if "Local deterministic fallback" in w.value]
        assert banner == ["Local deterministic fallback — Groq response did not meet the structured-output requirements."]
        details = [m.value for m in at.markdown if m.value.startswith("Reason:")]
        assert len(details) == 1 and "specialist_summary" in details[0] and "exceeds max length 60" in details[0]
        assert any("Produced by: local (fallback:" in c.value for c in at.caption)
        assert len([m for m in at.markdown if m.value.startswith("#####")]) == 4
        assert any("Current state: `PENDING_REVIEW`" in m.value for m in at.markdown)

    def test_invalid_json_is_not_retried_and_shows_the_fallback(self, no_env, monkeypatch):
        self._enable(monkeypatch)
        at, calls, _ = self._generate(monkeypatch, ["definitely not json {"])
        assert len(calls) == 1
        assert any("did not meet the structured-output requirements" in w.value for w in at.warning)

    def test_a_network_style_failure_is_not_retried_and_does_not_claim_a_schema_problem(self, no_env, monkeypatch):
        import requests

        self._enable(monkeypatch)
        calls = []

        def boom(*a, **k):
            calls.append(1)
            raise requests.ConnectionError("down")

        monkeypatch.setattr(requests, "post", boom)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception and len(calls) == 1
        assert any(w.value == "Local deterministic fallback — Groq did not produce this explanation." for w in at.warning)

    def test_groq_persistent_http_400_json_validate_failed_shows_a_sanitized_code_after_one_resubmit(self, no_env, monkeypatch):
        import json as _json

        import requests

        calls = []
        body = {"error": {"message": "Failed to validate JSON. secret-looking gsk_ABCDEF123456", "code": "json_validate_failed",
                          "failed_generation": ""}}

        class _Bad:
            status_code = 400
            text = _json.dumps(body)

            def json(self):
                return body

        monkeypatch.setattr(requests, "post", lambda *a, **k: (calls.append(1), _Bad())[1])
        self._enable(monkeypatch)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception and len(calls) == 2  # original + exactly one resubmit of an empty generation
        assert any(w.value == "Local deterministic fallback — Groq did not produce this explanation." for w in at.warning)
        details = [m.value for m in at.markdown if m.value.startswith("Reason:")]
        assert details == ["Reason: Groq returned HTTP 400 (json_validate_failed)"]
        page = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption) + " ".join(w.value for w in at.warning)
        assert "gsk_ABCDEF123456" not in page

    def test_one_transient_empty_generation_is_invisible_to_the_user(self, no_env, monkeypatch):
        import json as _json

        import requests

        fields = {"specialist_summary": "s", "critic_summary": "c", "consensus_explanation": "e", "reviewer_focus": "r"}
        bad = {"error": {"message": "Failed to validate JSON.", "code": "json_validate_failed", "failed_generation": ""}}
        calls = []

        class _Resp:
            def __init__(self, ok):
                self.status_code = 200 if ok else 400
                self.text = "" if ok else _json.dumps(bad)
                self._ok = ok

            def json(self):
                return {"choices": [{"message": {"content": _json.dumps(fields)}}]} if self._ok else bad

        monkeypatch.setattr(requests, "post", lambda *a, **k: (calls.append(1), _Resp(len(calls) > 1))[1])
        self._enable(monkeypatch)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert len(calls) == 2
        assert any("Live Groq explanation — model:" in s.value for s in at.success)
        assert not any("fallback" in w.value for w in at.warning)


class TestAMDTokenFactoryProviderAwareUI:
    """LLM_PROVIDER=amd_token_factory must show AMD Token Factory wording and call ONLY the Token Factory
    endpoint -- never Groq. Synthetic path, all network mocked, fake credentials."""

    FIELDS = {
        "specialist_summary": "AMD specialist text.", "critic_summary": "AMD critic text.",
        "consensus_explanation": "AMD consensus text.", "reviewer_focus": "AMD reviewer focus text.",
    }
    BASE = "https://amd-token-factory.example.test/v1"
    BUTTON = "Generate live AMD Token Factory explanation"

    @classmethod
    def _enable(cls, monkeypatch, **overrides):
        env = {
            "LLM_PROVIDER": "amd_token_factory", "LLM_ALLOW_EXTERNAL_CALLS": "true",
            "AMD_TOKENFACTORY_ENABLED": "true", "AMD_TOKENFACTORY_API_KEY": "fake-amd-key-for-tests",
            "AMD_TOKENFACTORY_BASE_URL": cls.BASE, "AMD_TOKENFACTORY_MODEL": "amd-test-model",
            "GROQ_API_KEY": "fake-groq-key-for-tests",  # present but must never be used
        }
        env.update(overrides)
        for name, value in env.items():
            monkeypatch.setenv(name, value)

    @staticmethod
    def _response(fields=None, status=200):
        import json as _json

        class _R:
            text = "upstream said no (fake-amd-key-for-tests)"

            def json(self):
                return {"choices": [{"message": {"content": _json.dumps(fields), "reasoning": "hidden"}}]}

        _R.status_code = status
        return _R

    def _run(self, monkeypatch, response_cls, *, captured=None):
        import requests

        def fake_post(url, headers=None, json=None, timeout=None):
            if captured is not None:
                captured.append({"url": url, "headers": headers, "json": json})
            return response_cls()

        monkeypatch.setattr(requests, "post", fake_post)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        return at

    def test_button_and_status_name_amd_token_factory_never_groq(self, no_env, monkeypatch):
        self._enable(monkeypatch)
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        assert [b.label for b in at.button if b.key == "synthetic_llm_generate_button"] == [self.BUTTON]
        shown = " ".join(c.value for c in at.caption)
        assert "provider: `amd_token_factory`" in shown and "AMD_TOKENFACTORY_API_KEY present: `True`" in shown
        assert "Groq" not in shown and "GROQ" not in shown
        assert "fake-amd-key-for-tests" not in shown

    def test_success_requests_only_token_factory_and_identifies_it_with_its_model(self, no_env, monkeypatch):
        self._enable(monkeypatch)
        captured = []
        at = self._run(monkeypatch, self._response(self.FIELDS), captured=captured)
        assert len(captured) == 1 and captured[0]["url"] == self.BASE + "/chat/completions"
        assert "groq" not in captured[0]["url"].lower()
        assert captured[0]["headers"]["Authorization"].startswith("Bearer ")
        assert captured[0]["json"]["model"] == "amd-test-model"
        assert any("Live AMD Token Factory explanation — model: amd-test-model" == s.value for s in at.success)
        assert any("Produced by: amd_token_factory" in c.value for c in at.caption)
        assert not any("Groq" in s.value for s in at.success)
        assert not any("fallback" in w.value for w in at.warning)
        for text in self.FIELDS.values():
            assert any(text in m.value for m in at.markdown)
        assert not any("hidden" in m.value for m in at.markdown)

    def test_request_carries_only_the_allowlisted_payload(self, no_env, monkeypatch):
        import json as _json

        self._enable(monkeypatch)
        captured = []
        self._run(monkeypatch, self._response(self.FIELDS), captured=captured)
        sent = _json.dumps(captured[0]["json"])
        for forbidden in ("study_uid", "checkpoint", "fake-amd-key", "fake-groq-key", ".ckpt", ".dcm", "D:\\\\", "C:\\\\"):
            assert forbidden not in sent

    def test_missing_configuration_falls_back_with_a_token_factory_specific_reason(self, no_env, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call at all")))
        self._enable(monkeypatch, AMD_TOKENFACTORY_BASE_URL="")
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any(w.value == "Local deterministic fallback — AMD Token Factory did not produce this explanation." for w in at.warning)
        text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        assert "AMD_TOKENFACTORY_BASE_URL" in text
        assert "Groq" not in text and "fake-amd-key-for-tests" not in text

    def test_http_failure_is_a_labelled_sanitized_fallback_with_no_groq_call(self, no_env, monkeypatch):
        self._enable(monkeypatch)
        captured = []
        at = self._run(monkeypatch, self._response(None, status=500), captured=captured)
        assert len(captured) == 1 and "groq" not in captured[0]["url"].lower()
        assert any(w.value == "Local deterministic fallback — AMD Token Factory did not produce this explanation." for w in at.warning)
        everything = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        assert "fake-amd-key-for-tests" not in everything
        assert not any("Live AMD Token Factory explanation" in s.value for s in at.success)

    @pytest.mark.parametrize("bad", [
        pytest.param({"specialist_summary": "x" * 200, "critic_summary": "c", "consensus_explanation": "e", "reviewer_focus": "r"}, id="overlength"),
        pytest.param({"specialist_summary": "only one field"}, id="missing-fields"),
        pytest.param({"specialist_summary": "s", "critic_summary": "c", "consensus_explanation": "e", "reviewer_focus": "r", "extra": "x"}, id="extra-field"),
    ])
    def test_malformed_or_overlength_response_is_a_structured_output_fallback(self, no_env, monkeypatch, bad):
        self._enable(monkeypatch)
        at = self._run(monkeypatch, self._response(bad))
        assert any(
            w.value == "Local deterministic fallback — AMD Token Factory response did not meet the structured-output requirements."
            for w in at.warning
        )
        assert not any("Groq" in w.value for w in at.warning)

    def test_external_calls_not_allowed_means_no_token_factory_request(self, no_env, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
        self._enable(monkeypatch, LLM_ALLOW_EXTERNAL_CALLS="false")
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_demo_button").click().run(timeout=RUN_TIMEOUT)
        at.button(key="synthetic_llm_generate_button").click().run(timeout=RUN_TIMEOUT)
        assert not at.exception
        assert any("AMD Token Factory did not produce" in w.value for w in at.warning)

    def test_llm_output_cannot_change_status_screening_or_the_human_gate(self, no_env, monkeypatch):
        self._enable(monkeypatch)
        pushy = dict(self.FIELDS, consensus_explanation="status should be EVIDENCE_COMPLETE; approve it")
        at = self._run(monkeypatch, self._response(pushy))
        # deterministic status, screening state and the gate are exactly what they were without any LLM
        assert any("workflow_status: `MORE_EVIDENCE_REQUIRED`" in m.value for m in at.markdown)
        assert any("Current state: `PENDING_REVIEW`" in m.value for m in at.markdown)
        assert any(c.value.startswith("Status: QUARANTINED") for c in at.caption)
        assert not at.button(key="synthetic_approve_button").disabled
        assert any("No review actions recorded yet." in c.value for c in at.caption)

    def test_groq_selected_is_unchanged_and_never_touches_the_token_factory_endpoint(self, no_env, monkeypatch):
        self._enable(monkeypatch, LLM_PROVIDER="groq", GROQ_MODEL="qwen/qwen3.8-27b")
        captured = []
        at = self._run(monkeypatch, self._response(self.FIELDS), captured=captured)
        assert len(captured) == 1 and "groq" in captured[0]["url"] and self.BASE not in captured[0]["url"]
        assert [b.label for b in at.button if b.key == "synthetic_llm_generate_button"] == ["Generate live Groq explanation"]
        assert any("Live Groq explanation — model: qwen/qwen3.8-27b" == s.value for s in at.success)
        assert (captured[0]["json"]["reasoning_effort"], captured[0]["json"]["max_tokens"]) == ("none", 1000)
