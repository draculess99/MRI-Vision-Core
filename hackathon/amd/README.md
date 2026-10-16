# AMD / ROCm adapter

Thin adapter over the shared MRI-Core engine (`mri_core.rsna_inference_engine`) for the AMD
Developer Hackathon. All AMD-specific code lives in `hackathon/amd/`; `mri_core/`, `main`, and the
OpenCV/Nebius branches are untouched. Based on `integration/mri-core-shared` @ `1643d45`.

> Research prototype, not a diagnostic device. Probabilities are raw, uncalibrated sigmoid outputs
> of a model trained on at most 58 labeled studies. Nothing here is a clinical finding or a
> confidence score. No training happens in this adapter.

## Modules

| Command | Purpose |
|---|---|
| `python -m hackathon.amd.rocm_env [--require-amd] [--smi]` | Reports torch/ROCm build, visible devices, `rocm-smi`/`amd-smi`. `amd_gpu_available` is true only for a ROCm (HIP) torch build with a visible device; a CUDA or CPU-only build reports `false` with reasons. |
| `python -m hackathon.amd.run_inference --data-root D --checkpoint C.pt [--study-uid U] [--device cuda]` | One real study through the shared engine. Writes `outputs/amd/prediction.json`. Fails (exit 1) rather than inventing output if the checkpoint or study is missing. |
| `python -m hackathon.amd.benchmark --data-root D --checkpoint C.pt [--iterations 50 --warmup 5 --batch-sizes 1 4]` | CPU baseline vs AMD GPU, native `RSNAKneeCNN` path, on the identical checkpoint, preprocessed tensors, and config. Writes `outputs/amd/benchmark.json`. |
| `python -m hackathon.amd.benchmark_kaggle_bridge --data-root D --checkpoint fold0-knee-mil-best.ckpt [--iterations 50 --warmup 5 --batch-sizes 1 4]` | Same CPU-vs-AMD-GPU measurement, for the bridged Kaggle attention-MIL checkpoint (see "Bridged real-checkpoint smoke test" below). Writes `outputs/amd/benchmark_kaggle_bridge.json`. Does not modify or import model logic from `benchmark.py`; only its generic, model-agnostic timing/memory helpers are reused. |

Add `--dicom-root` if DICOMs are not under `<data-root>/raw` (native path only; the bridged path always reads `<data-root>/raw/train_series/...`).

## Output separation (`prediction.json`)

Three top-level sections that never share data:

- `prediction` - the engine's `StudyInferenceResult`, verbatim (probabilities, provenance with
  checkpoint SHA-256, latency).
- `evidence_review` - deterministic evidence/consistency checks from `mri_core.agents`, with
  `quality_gate_status` defaulting to `NOT_ASSESSED`, and `uncertainty.status = "not_estimated"`
  (no uncertainty method exists for these checkpoints). Contains no probabilities.
- `human_review_gate` - simulated review state machine, starts at `PENDING_REVIEW`, bound to the
  evidence fingerprint. No clinician approval is implied.

The OpenCV branch's own ChangeGuard module is not part of the shared commit and was not copied;
the shared evidence layer above is what this adapter surfaces.

## Benchmark methodology

- Study is preprocessed once; the same tensors go to every device (preprocessing time reported
  separately, excluded from device latency).
- Warm-up iterations are discarded; GPU timing is bracketed by `torch.cuda.synchronize()`.
- Reports per batch size: mean/median/p95/min/max/stdev latency and throughput (studies/s).
- GPU resources: peak allocated/reserved bytes, total/free memory, raw `rocm-smi`/`amd-smi` text.
  CPU: torch threads and process RSS (needs `psutil`).
- `max_abs_prob_diff_vs_cpu` is recorded as a numerical sanity check.
- With no AMD GPU, the `gpu` block says `unavailable` with reasons and contains **no numbers**;
  no speedup is computed. `--require-gpu` turns that into exit code 2. `--allow-non-amd` is a
  dev-only escape hatch and labels the result `is_amd_rocm: false`.
- Speed only: no accuracy or clinical claim.

## Running on an AMD cloud GPU

```bash
# ROCm build of PyTorch (match your ROCm version), then:
pip install -r requirements-rsna.txt psutil
python -m hackathon.amd.rocm_env --require-amd --smi --output outputs/amd/env.json
python -m hackathon.amd.run_inference --data-root data/rsna-knee --checkpoint <trained last.pt> --device cuda
python -m hackathon.amd.benchmark     --data-root data/rsna-knee --checkpoint <trained last.pt> --require-gpu
```

ROCm PyTorch exposes the AMD GPU through the `cuda` device name.

## Tests

```bash
python -m pytest tests/test_amd_adapter.py -q
```

Default tests use the shared synthetic checkpoint/DICOM fixtures to check plumbing and output
structure only; their numbers are not real predictions or benchmarks. To run against real assets:
`AMD_CHECKPOINT=<last.pt> AMD_RSNA_ROOT=<data/rsna-knee> pytest tests/test_amd_adapter.py`.

## Bridged real-checkpoint smoke test (`kaggle_mil_bridge.py`)

No checkpoint compatible with `mri_core.rsna_knee_model.RSNAKneeCNN` exists on disk in this
project. The only real *trained* RSNA knee weights found locally are five
`foldN-knee-mil-best.ckpt` files from an external Kaggle notebook (`kneeabdet-baseline-v1`) --
a PyTorch Lightning checkpoint for a `timm efficientnet_b0` backbone with gated attention-MIL
pooling over DICOM slices, not the native MRNet-style encoder. `hackathon/amd/kaggle_mil_bridge.py`
is a minimal, inference-only port of that architecture and its exact preprocessing (24 slices,
224x224, 1st/99th percentile normalization, single-channel ImageNet mean/std) so those real
weights can be run on a real local study. It does not modify `mri_core`'s own model or checkpoint
code, and `run_inference.py` is untouched and still only talks to the native architecture.

Needs `timm` and `pydicom` in addition to `requirements-rsna.txt` (`pip install timm pydicom`).

```bash
python -m hackathon.amd.run_real_checkpoint_smoke \
  --data-root <real RSNA knee data_root with raw/train_series/...> \
  --checkpoint <foldN-knee-mil-best.ckpt> \
  --out outputs/amd/real_checkpoint_prediction.json
```

Output (`real_checkpoint_prediction.json`) has three sections: `prediction` (study/series UID,
plane, 12 sigmoid probabilities, latency), `provenance` (checkpoint path + SHA-256, bridged
architecture description, Lightning epoch/step), and `non_diagnostic_notice`. The probabilities
are real forward-pass outputs of real trained weights on a real DICOM series -- not fabricated --
but this checkpoint has never been evaluated or calibrated by this project, so they carry no
accuracy claim.

Tests: `tests/test_amd_kaggle_bridge.py`, gated by `AMD_KAGGLE_CHECKPOINT` and
`AMD_KAGGLE_RSNA_ROOT` (skips cleanly without them, same convention as `AMD_CHECKPOINT`/
`AMD_RSNA_ROOT` above).

## Bridged benchmark (`benchmark_kaggle_bridge.py`)

Same CPU-vs-AMD-GPU measurement as `benchmark.py`, applied to the bridged checkpoint above
instead of `RSNAKneeCNN`. A separate module so the native path's tests and behavior cannot
regress; it reuses only `benchmark.py`'s generic, model-agnostic timing/memory helpers
(`_sync`, `_summary`, `_process_rss_bytes`), never its model-loading code.

```bash
python -m hackathon.amd.benchmark_kaggle_bridge \
  --data-root <real RSNA knee data_root with raw/train_series/...> \
  --checkpoint <foldN-knee-mil-best.ckpt> \
  --iterations 50 --warmup 5 --batch-sizes 1 4 \
  --output outputs/amd/benchmark_kaggle_bridge.json
```

Same fairness rules as `benchmark.py` (one-time preprocessing, fresh model per device, discarded
warm-up, `torch.cuda.synchronize()`-bracketed GPU timing, `psutil`-based CPU RSS when installed).
`gpu.is_amd_rocm` always reflects the real environment check (`rocm_env.check_environment()`),
never the device string passed in -- a CUDA/MPS accelerator run with `--allow-non-amd` is still
labeled `is_amd_rocm: false`, and a CPU-only machine's `gpu` block is `{"status": "unavailable",
"reasons": [...]}` with no numbers at all, never a fabricated GPU result. `workload` includes the
checkpoint's SHA-256 and the bridged architecture string for provenance.

Tests: `tests/test_amd_benchmark_kaggle_bridge.py` -- always runs (synthetic DICOM fixture +
untrained-but-correctly-shaped Lightning checkpoint, same convention as `test_amd_adapter.py`'s
synthetic tests), including two fully mocked tests of the GPU-available code path (`is_amd_rocm`
true/false wiring) that need no real accelerator.

## Status

- Native-architecture path (`run_inference.py`, `benchmark.py`): the trained fold checkpoints
  (`outputs/rsna-fivefold-benchmark/...`) for `RSNAKneeCNN` are gitignored and were not found
  anywhere on this machine; no real-checkpoint prediction through this path has been produced.
- Bridged path (`kaggle_mil_bridge.py`, above): a real end-to-end run against a real DICOM study
  and a real externally-trained checkpoint has been produced and is reproducible bit-for-bit
  across runs (see commit for the exact command and output).
- Bridged benchmark (`benchmark_kaggle_bridge.py`, above): a real CPU-only run against the same
  real DICOM study and checkpoint has been produced (batch sizes 1 and 4; `outputs/amd/
  benchmark_kaggle_bridge.json`). `gpu.status` correctly reports `unavailable` on this machine.
- No AMD GPU has been available in this environment; `rocm_env.py` has only ever reported
  `amd_gpu_available: false` with reasons. No AMD GPU benchmark (native or bridged) has been
  produced. This is the only remaining requirement for a genuine AMD-hosted demo: an actual ROCm
  environment to run the two commands above (`benchmark.py` once a native checkpoint exists,
  `benchmark_kaggle_bridge.py` today) unchanged.

## Phase 15 research results (reference only — not integrated into this adapter)

Published upstream on `hackathon/opencv-aws` @ `0a69043` (`docs/phase15_training_health_diagnostics.md`).
This is **training-health diagnostics for a third, separate experimental architecture**
("Model C0": ResNet-18 from scratch, 2.5D, attention-MIL, 128px x 9 slices) — not `RSNAKneeCNN`
(the native path above) and not the bridged Kaggle `efficientnet_b0` attention-MIL checkpoint
(the bridged path above). Nothing from Phase 15 has been deployed, trained, or submitted here;
this section only restates what the published diagnostics say, with the distinctions they
themselves insist on:

- **Training-subset ("overfit") AUC — a sanity check, not a performance number.** A 64-study
  subset reached fit macro AUC 1.000 after 60 epochs. This is deliberate memorization of the
  training data used to verify the pipeline is mechanically sound (masks, loss, optimizer, AMP,
  checkpoint/resume); it says nothing about generalization and must never be read as a model
  accuracy figure.
- **Weak-validation AUC — agreement with automated weak labels on a held-out training-only
  split, not clinical accuracy.** Real labels reach final macro AUC 0.615-0.66 across two seeds
  and learning rates, against a shuffled-label negative control at 0.516-0.56 (chance). The gap
  is real but small (about 0.10-0.14 AUC) and was found by looking at the same split that then
  reported it, so the report itself calls the best learning-rate result "a lead, not a confirmed
  effect," pending a pre-registered re-run on fresh splits (proposed, not launched).
- **Clinical performance: not measured, not claimed.** The 58 gold-labeled studies, the Kaggle
  test images, and the leaderboard were explicitly not touched in Phase 15. The source document
  states directly: "No clinical claim is made." This adapter adds nothing to that; it cannot and
  does not claim Phase 15 improves clinical performance.
- **Weights:** three run exports are hash-verified in the source document (`last.pt`/
  `last_best.pt` SHA-256 for `overfit64_resume`, `real_e6_s0`, `real_e12_s0`) but, per the
  project's standing convention, are git-ignored and were not found anywhere on this machine.
  No Phase 15 weight has been deployed into `hackathon/amd/` by this change, and none will be
  without a separate, explicit request.

## Interactive demo (Phase 1): `streamlit_app.py`

A local Streamlit app that lets a user pick a real downloaded study, run the bridged Kaggle
checkpoint on it, and walk the result through the shared evidence-review and human-review
layers -- all on CPU, with no network call and no credential required.

### Launch

```bash
pip install -r requirements-rsna.txt timm pydicom psutil
cp .env.example .env    # optional -- see "Optional LLM reasoning layer" below; safe to skip entirely
set AMD_KAGGLE_RSNA_ROOT=<real rsna-knee data_root with raw/train_series/...>   # optional, prefills the sidebar
set AMD_KAGGLE_CHECKPOINT=<foldN-knee-mil-best.ckpt>                            # optional, prefills the sidebar
streamlit run hackathon/amd/streamlit_app.py
```

(`set` is PowerShell/cmd; use `export` on bash.) Both paths can also be typed directly into the
sidebar at runtime -- nothing is hardcoded to a specific machine. If either field is empty the
app shows an instruction and stops; it never guesses a path or fabricates a result. With no
`.env` at all, the optional LLM section below runs entirely in local deterministic mode.

### Architecture

```
mri_core.rsna_integration        (study discovery, unmodified)
        |
hackathon.amd.kaggle_mil_bridge  (checkpoint + forward pass, unmodified)
        |
hackathon.amd.agent_bridge_adapter   <- new, tested: StudyInferenceResult adapter
        |
mri_core.agents.orchestrator.process_inference_result   (specialist/critic/consensus/review gate, unmodified)
        |
hackathon.amd.review_actions     <- new, tested: approve / request-more-evidence / escalate
        |
hackathon.amd.streamlit_app      <- new: wires the above into a UI, plus:
  - hackathon.amd.providers      <- new, tested: local (default) vs. optional AMD Token Factory
  - outputs/amd/benchmark_kaggle_bridge.json, if present   (CPU benchmark evidence panel)
```

### Why an adapter was needed

`mri_core.agents.orchestrator.process_inference_result()` takes exactly one type:
`mri_core.rsna_inference_engine.StudyInferenceResult`. The bridge's output (see
`run_real_checkpoint_smoke.py`) is a plain dict, because its checkpoint is architecturally
foreign to `mri_core`'s own loaders -- the two are not directly compatible. Rather than modify
`contracts.build_evidence_bundle()` (out of scope: "do not modify the shared mri_core...
implementation"), `agent_bridge_adapter.py` builds a real `StudyInferenceResult` with a real
`ModelProvenance` (both imported unchanged from `mri_core.rsna_inference_engine`), so every
downstream agents-layer module runs completely unmodified on a bridged result. It always
attaches a warning identifying the checkpoint as bridged/external, so the critic layer's
`ENGINE_WARNINGS_PRESENT` finding fires and consensus can never report a bridged run as
evidence-complete. `ModelProvenance.fold` is populated only when the checkpoint filename
matches the real `foldN-...` convention; any other filename gives `fold=None`, not a guess.

### Human review gate: escalate is not a separate shared state

The shared `ReviewGate` only has three states: `PENDING_REVIEW`, `MORE_EVIDENCE_REQUESTED`,
`REVIEWED` -- there is no distinct "escalated" state, and adding one to shared `mri_core` code
is out of scope here. `review_actions.escalate()` therefore transitions to `REVIEWED`, the same
underlying state `approve()` uses, but always with an audit `reason` prefixed `"ESCALATED: "`
so the two are distinguishable in the append-only history -- never silently merged. The gate
never bypasses human approval: every action requires an explicit actor and a non-empty reason,
validated by the shared `ReviewGate.transition()` itself.

### Optional AMD Token Factory provider

`hackathon/amd/providers.py` adds `AMDTokenFactoryProvider`, disabled by default
(`enabled=False`). Disabled, it makes no network call and requires no credentials. Even if a
user checks "Enable AMD Token Factory" in the sidebar and fills in a base URL and API key,
`status()` reports it as configured but unverified -- there is no real HTTP client behind it
yet, and `device()` raises `NotImplementedError` rather than pretending to dispatch work to AMD
hardware. This is the explicit seam for a future real Token Factory integration; nothing here
claims AMD acceleration that was not measured.

### Optional LLM reasoning layer (`llm_config.py`, `llm_providers.py`)

A second, independent optional section in the same app: an untrusted, free-text "reviewer
interpretation" of the deterministic evidence above, routed through the shared, unmodified
`mri_core.agents.llm_adapter.screen_proposal()` (BLOCKED or QUARANTINED -- never "verified")
and never fed back into consensus, the review gate, or any approval decision.

**Configuration is environment/`.env`-driven, not code.** Copy `.env.example` (repo root) to
`.env` and set what you need; see that file for all seven variables. With no `.env` and no
environment variables set, the app runs local-only: no network call, no credential required.

**The hard safety gate** (`llm_config.build_adapter()`): `LLM_ALLOW_EXTERNAL_CALLS` must be
explicitly `true`, *in addition to* `LLM_PROVIDER` naming that provider and a credential being
present, before Groq or AMD Token Factory are ever invoked. **An ambient `GROQ_API_KEY` sitting
in the environment is never, by itself, enough** -- this was a real incident during development
(a test script accidentally made one live, failed call because a real key happened to already be
set), and it is now a permanent regression test
(`test_amd_llm_providers.py::TestAmbientCredentialAloneCannotAuthorize` /
`test_amd_streamlit_app.py::test_ambient_groq_api_key_alone_does_not_authorize_a_call`).

**Verified live Groq connectivity (synthetic data only, one approved call):** on 2026-10-16,
one real request through the unmodified `GroqAdapter.propose()` code path, model
`openai/gpt-oss-20b`, `reasoning_effort=low`, structured `response_format` (`json_schema`,
strict), returned **HTTP 200 in 1.848s with a valid structured JSON response**. The only input
was a synthetic, non-medical placeholder (`study_uid="synthetic-smoke-test-not-a-real-study"`,
placeholder scores, a zeroed fake checkpoint hash) -- no real study, DICOM, patient data, or
project data of any kind was sent. This confirms the Groq *transport and schema* work; it is
**not** a validation of the quality, safety, or usefulness of what Groq says about a real study
-- every real result still only ever reaches `QUARANTINED` or `BLOCKED`, never trusted evidence.

**Mock-tested failure handling** (no live call needed to verify, re-run anytime):
`GroqAdapter` is tested against a missing credential, request timeout, connection error, every
Groq HTTP error status, invalid/non-JSON response bodies, 8 variants of a malformed or
schema-violating structured response, and a reasoning model returning its hidden `reasoning`
field with an empty `content` (the real failure `llama-3.3-70b-versatile` produced before the
model/schema fix below) -- every case cleanly raises `LLMProviderUnavailableError` and falls
back to `LocalDeterministicAdapter` via `propose_with_fallback()`, never crashing the app.

**Why the model changed from `llama-3.3-70b-versatile` to `openai/gpt-oss-20b`:** the former
returned HTTP 404 ("does not exist or you do not have access to it") for the real key tested
here, despite being listed as a current production model in Groq's docs -- availability can be
account/tier-specific. The latter is a reasoning model; without a structured `response_format`
it could spend its whole token budget on hidden reasoning and leave `message.content` empty,
which the adapter correctly rejected rather than ever substituting the `reasoning` field as the
answer. Requesting strict JSON-schema output (`{"summary": str}`) fixed this.

**AMD Token Factory LLM reasoning: not implemented, never claims otherwise.**
`AMDTokenFactoryLLMAdapter.propose()` always raises `NotImplementedError`, regardless of
`AMD_TOKENFACTORY_ENABLED`/configuration -- there is no real Token Factory LLM client in this
project, and no live call has ever been attempted against one.

**No AMD GPU acceleration has been demonstrated anywhere in this adapter** -- not for the LLM
layer (Groq's own inference hardware is irrelevant to this project's AMD claim) and not for
model inference/benchmarking (see "Status" below). All of this remains research-prototype,
non-diagnostic software; nothing in this section produces or implies a clinical finding.

Tests: `tests/test_amd_llm_config.py` (28), `tests/test_amd_llm_providers.py` (47) -- both
always run, fully offline/mocked.

### Dependencies

Same as the bridged smoke test above (`requirements-rsna.txt`, `timm`, `pydicom`), plus
`streamlit` (already in `requirements.txt`), `requests` (already in `requirements.txt`),
optionally `python-dotenv` for `.env` file support (`llm_config.py` degrades gracefully without
it -- real environment variables still work), and optionally `psutil` for CPU memory reporting
in the benchmark panel.

### Tests

```bash
python -m pytest tests/test_amd_agent_bridge_adapter.py tests/test_amd_providers.py \
    tests/test_amd_review_actions.py tests/test_amd_llm_config.py tests/test_amd_llm_providers.py \
    tests/test_amd_streamlit_app.py -v
```

- `test_amd_agent_bridge_adapter.py` -- adapter correctness and full orchestrator compatibility.
- `test_amd_providers.py` -- disabled-by-default, no-network-call, no-fabricated-AMD-claim checks
  (including a socket-construction guard asserting the disabled/unconfigured provider never
  opens a real connection).
- `test_amd_review_actions.py` -- state-transition and audit-trail correctness, independent of
  the UI.
- `test_amd_streamlit_app.py` -- `streamlit.testing.v1.AppTest`-driven: the no-config path always
  runs; the full pipeline (inference -> review actions -> audit trail -> benchmark panel) is
  gated behind `AMD_KAGGLE_CHECKPOINT`/`AMD_KAGGLE_RSNA_ROOT`, same convention as
  `test_amd_kaggle_bridge.py`.

### Known limitations

- The deterministic image-quality gate (`mri_core.decision`) has never been run against the
  bridge's own preprocessing, so `quality_gate_status` is always `NOT_ASSESSED`; consensus
  therefore always reports `MORE_EVIDENCE_REQUIRED` for a bridged result, never
  `EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW`. This is intentional honesty, not a bug to silence.
- "Escalate" and "Approve" share one underlying terminal state in the shared `ReviewGate` (see
  above) -- they are distinguished only in the audit reason/actor, not by a separate state.
- The benchmark panel only displays a file already produced by `benchmark_kaggle_bridge.py`; the
  app itself does not run the benchmark.
- The AMD Token Factory provider is a UI/status seam only; it cannot actually run inference on
  AMD cloud hardware yet.
- `st.cache_resource` caches the loaded model per checkpoint path for the life of the Streamlit
  session; restart the app (not just rerun) after replacing a checkpoint file at the same path.
- Groq connectivity has been verified live exactly once, with synthetic non-medical data only
  (see "Optional LLM reasoning layer" above) -- this confirms the transport and schema work, not
  the quality or safety of what Groq would say about a real study's evidence.
- AMD Token Factory LLM reasoning has never been tested against a live endpoint and has no real
  client; it cannot be enabled into working, only into an honest "not implemented" status.
