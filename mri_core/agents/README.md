# mri_core.agents — Shared Specialist-Critic-Consensus Evidence-Review Layer

**Research prototype. Not a diagnostic device. All statuses below are workflow
classifications about evidence completeness, never clinical findings or a diagnosis.**

This package is shared MRI-Vision-Core infrastructure — it is not specific to the
OpenCV, Nebius, or AMD hackathon branches. It consumes exactly one existing interface,
`mri_core.rsna_inference_engine.StudyInferenceResult` (built in Phase 6), and adds
nothing to how inference itself runs. No model architecture, checkpoint, or training
code is touched by anything in this package.

## Why this exists

Phase 6 gave every branch a single, structured inference result. Phase 8 adds a
second, equally reusable layer on top of that: a deterministic way to (a) group the
twelve raw scores for readability, (b) catch structural problems in the evidence
before anyone acts on it, (c) decide what to do next in plain workflow terms, and (d)
simulate a human-review audit trail — without any branch reimplementing this logic on
its own, and without ever inventing a diagnosis.

## Architecture and module boundaries

```
StudyInferenceResult (Phase 6, mri_core.rsna_inference_engine)
        │
        ▼
contracts.build_evidence_bundle()  →  EvidenceBundle (frozen, fingerprinted)
        │
        ├──► specialist.summarize_by_group()  →  4 SpecialistSummary (grouped scores)
        │
        ├──► critic.review()                  →  tuple[CriticFinding, ...]
        │
        └──► consensus.build_consensus()       →  ConsensusResult (workflow_status)
                     │
                     ▼
            review_gate.ReviewGate              simulated human-review audit trail
                     │
                     ▼ (optional, offline only)
            llm_adapter.LLMAdapter               proposed reviewer brief, strictly validated
```

`orchestrator.process_inference_result()` wires all of the above together in one call
and is the only function most callers need.

| Module | Responsibility |
|---|---|
| `contracts.py` | `EvidenceBundle`, enums (`QualityGateStatus`, `WorkflowStatus`, `ReviewState`, `Severity`), the only constructor (`build_evidence_bundle`) |
| `specialist.py` | Deterministic grouping of the 12 targets into 4 categories; no interpretation |
| `critic.py` | Rule-based structural/consistency checks, returns typed findings |
| `consensus.py` | Documented rule order → one `WorkflowStatus`; never averages into a score |
| `review_gate.py` | Simulated review state machine with an append-only audit trail |
| `llm_adapter.py` | Offline-only adapter interface + deterministic stub + strict output validation |
| `orchestrator.py` | Top-level `process_inference_result()` |
| `evaluation.py` | Synthetic workflow-validation scenarios + 3-level ablation |
| `cli_demo.py` | `python -m mri_core.agents.cli_demo` — runs the scenarios and prints findings |

## Input/output data contracts

**Input — `EvidenceBundle`** (built only via `build_evidence_bundle(result, ...)`, never
hand-assembled):

| Field | Source |
|---|---|
| `study_uid` | `StudyInferenceResult.study_uid` |
| `inference_status` | `StudyInferenceResult.status` |
| `probabilities` | `StudyInferenceResult.probabilities` — exactly the 12 `TARGET_COLUMNS`, in order, or `None` |
| `is_ensemble`, `model_count` | from `StudyInferenceResult.is_ensemble` / `len(model_provenance)` |
| `checkpoint_provenance` | `ModelProvenance.to_dict()` snapshots, order preserved |
| `latency_seconds` | `StudyInferenceResult.latency_seconds` |
| `quality_gate_status` | caller-supplied; **defaults to `NOT_ASSESSED`**, never silently `OK` |
| `warnings` | `StudyInferenceResult.warnings` |
| `evidence_references` | caller-supplied (e.g. checkpoint paths, benchmark report paths) |
| `limitations` | `DEFAULT_LIMITATIONS` + any caller-supplied extras |
| `disclaimer` | `StudyInferenceResult.disclaimer` |
| `fingerprint` | SHA-256 over the above, computed automatically |

**Output — `WorkflowResult`** (from `orchestrator.process_inference_result()`):
`bundle`, `specialist_summaries` (4), `critic_findings` (N), `consensus`
(`ConsensusResult` with `workflow_status`), `review_gate` (fresh `ReviewGate`).

Every dataclass in this package has a `to_dict()` that is plain-JSON-serializable.

### Immutable contracts

A frozen dataclass alone only blocks reassigning its *attributes* — a plain dict or
list stored inside one remains fully mutable in place (`bundle.probabilities["ACL"] =
0.99` would silently succeed on a naive frozen dataclass, leaving the bundle's own
`fingerprint` permanently stale). `EvidenceBundle` is therefore **deep-frozen**:
`probabilities` is a `types.MappingProxyType` (read-only mapping, insertion order
preserved so `TARGET_COLUMNS` order survives untouched) and `checkpoint_provenance` is
a tuple of `MappingProxyType` snapshots, each with its own nested fields (e.g.
`target_columns`) recursively frozen too, via a generic `_deep_freeze()` helper.
Attempting to mutate anything in the tree raises `TypeError` immediately — there is no
path by which the fingerprint can go stale, because mutation itself is rejected, not
merely discouraged.

`build_evidence_bundle()` also always copies the caller's `probabilities` dict before
freezing it, so later mutating the original `StudyInferenceResult.probabilities` the
caller still holds can never retroactively affect an already-built bundle.

**Fingerprint canonicalization.** The fingerprint is computed from a `_deep_jsonable()`
pass over every field (resolving `MappingProxyType`/tuple back to plain dict/list) with
**explicit, non-silent handling of non-finite floats**: `NaN` → the string `"NaN"`,
`+Infinity`/`-Infinity` → `"Infinity"`/`"-Infinity"`. `json.dumps(..., allow_nan=False)`
is then used as a defensive check that nothing non-finite could slip through
unconverted. This matters because Python's `json` module will otherwise silently emit
non-standard literal `NaN`/`Infinity` tokens with no error — this package never relies
on that implicit behavior for hashing or for `to_dict()` output.

## Specialist, critic, and consensus responsibilities

- **Specialist** only groups and restates. `GROUPS` covers the 12 targets exactly once
  (enforced at import time by `_validate_group_coverage()`): Ligaments (ACL, MCL),
  Menisci (Medial/Lateral Meniscus), Osteoarthritis (Medial/Lateral/PF OA), Other
  (Effusion, Synovitis, Baker's, Contusion, Fracture). It never says a score near 0.5
  means an abnormality was detected, and it never issues a positive/negative call.
- **Critic** checks structure and consistency only: label order and count, finite
  values in [0,1], non-empty provenance with a real SHA-256, ensemble/single-model
  counts matching their claim, and whether the quality gate was actually assessed. **A
  successful inference status is never treated as a safety signal** — there is no rule
  anywhere that marks `STATUS_OK` as "safe."
- **Consensus** applies one documented, ordered rule set (see `consensus.py`'s
  docstring) to pick exactly one `WorkflowStatus`. It never averages agent outputs into
  a confidence score — there's no such number anywhere in this package.

## Workflow statuses and deterministic precedence

Exact rule order in `consensus.build_consensus()` (first match wins; each rule is
independently tested in `tests/test_agents.py::TestConsensusPrecedenceEveryCondition`):

| # | Condition | Resulting status |
|---|---|---|
| 1 | Status `OK` with missing predictions | `INCONSISTENT_EVIDENCE` |
| 2 | Non-`OK` status with predictions present | `INCONSISTENT_EVIDENCE` |
| 3 | Invalid values (non-finite/out-of-range), wrong label order, or missing provenance | `INCONSISTENT_EVIDENCE` |
| 4 | Explicit unavailable/error status with no predictions (and none of 1–3 matched) | `INFERENCE_UNAVAILABLE` |
| 5 | Valid inference with `quality_gate_status` `NOT_ASSESSED`, `REVIEW`, or `INVALID` | `MORE_EVIDENCE_REQUIRED` |
| 6 | Valid inference, complete supporting evidence, `quality_gate_status` `OK` | `EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW` |

Rules 1–3 are implemented as one check ("any critic finding whose code is in
`INTERNAL_INCONSISTENCY_CODES`"), checked before rule 4, because a self-contradictory
engine result is a different, more concerning problem than plain unavailability.
`QualityGateStatus.INVALID` is folded into rule 5 (not treated as an inconsistency) on
the reasoning that an invalid-quality image needs a different/better image — that is
"more evidence," not an internal software bug.

## Human-review simulation

`review_gate.ReviewGate` implements `PENDING_REVIEW → {MORE_EVIDENCE_REQUESTED,
REVIEWED}`, `MORE_EVIDENCE_REQUESTED → {PENDING_REVIEW, REVIEWED}`, and `REVIEWED` as
terminal (no reopening in this simulation). Every transition requires the caller to
pass the evidence bundle's exact `fingerprint`; a mismatch raises `StaleEvidenceError`
rather than silently applying a decision to evidence that has since changed. **No
actual clinician approval is implied by any state** — this is stated explicitly in
both the module and class docstrings, and checked by a dedicated test. The audit trail
(`ReviewGate.history`) is append-only — entries are never edited or removed.

**Actor/reason validation.** `transition()` rejects a missing, empty, non-string, or
overlong `actor` (`InvalidActorError`) or `reason` (`InvalidReasonError`) *before*
checking the fingerprint or the transition's validity — a malformed call is rejected
regardless of what else might also be wrong with it. Transitions also never mutate the
original `EvidenceBundle` or `ConsensusResult` (verified directly by
`test_transition_never_mutates_original_evidence_bundle`).

Deterministic reasoning (specialist/critic/consensus) never reads `ReviewGate` state,
so changing review state can never alter a workflow status after the fact — this is
enforced by construction (no code path touches it) and verified by
`test_review_state_change_does_not_affect_deterministic_consensus`.

## Non-diagnostic limitations

- Trained on at most 58 fully labeled RSNA studies (46–48 per cross-validation fold).
- Probabilities are uncalibrated sigmoid outputs, not confidence scores.
- `quality_gate_status` defaults to `NOT_ASSESSED`; callers must pass a real
  `mri_core.decision`-derived status to claim otherwise — this package never infers it.
- No module here makes a positive/negative clinical call, assigns severity, or
  estimates diagnostic uncertainty.

## Integration examples

**OpenCV branch** (already has its own simulated review radio in
`hackathon/opencv_aws/streamlit_app.py`; this package is the shared version of that
same idea, usable without any Streamlit dependency):

```python
from mri_core.rsna_inference_engine import run_single_model_inference
from mri_core.agents.orchestrator import process_inference_result
from mri_core.agents.contracts import QualityGateStatus

result = run_single_model_inference(metadata, checkpoint_path, study_uid)
wr = process_inference_result(result, quality_gate_status=QualityGateStatus.NOT_ASSESSED)
st.write(wr.consensus.workflow_status.value)
for finding in wr.critic_findings:
    st.write(finding.message)
```

**Nebius/NVIDIA branch** — not present on `hackathon/opencv-aws`, so its actual files
were not inspected (switching branches was out of scope for this phase). The expected
integration point is identical to the OpenCV example above: call
`process_inference_result()` with that branch's own `StudyInferenceResult` and render
`wr.consensus`/`wr.critic_findings`/`wr.review_gate`. If a Nemotron-backed reviewer
brief is wanted: implement `LLMAdapter.propose()`, pass its `RawProposal` through
`screen_proposal()`, and render the trusted `render_reviewer_brief()` output and the
screened proposal as two **visually distinct** sections — never merge the quarantined
free text into the verified brief, and never render `BLOCKED`/`QUARANTINED` text as if
it were confirmed evidence.

**AMD branch** — same integration point as above; no AMD-specific code exists in this
package, by design, since it is hardware-agnostic (pure Python, no GPU dependency).

## Evaluation metrics and ablation methodology

`evaluation.run_evaluation()` runs 10 synthetic scenarios (complete-valid inference,
missing study, missing/invalid outputs, quality not performed, critic-detected
contradiction, more-evidence request, invalid reviewer transition, valid review
transition, stale fingerprint, rejected LLM proposal) and reports: scenario count,
pass/fail counts, workflow-status agreement (expected vs. actual, for the scenarios
that assert one), and total/mean processing latency.

**These are synthetic workflow-validation metrics only — not clinical accuracy or
diagnostic performance measurements.**

`evaluation.run_ablation()` compares three levels against the same complete-valid
fixture, measuring only what is actually present on the resulting objects:

| Level | What's exposed | Specialist | Critic | Workflow status | Review gate |
|---|---|---|---|---|---|
| A | Raw `EvidenceBundle` only | ✗ | ✗ | ✗ | ✗ |
| B | + specialist grouping + critic findings | ✓ | ✓ | ✗ | ✗ |
| C | + consensus workflow status + review gate | ✓ | ✓ | ✓ | ✓ |

Evidence-field completeness (fraction of the 9 required `EvidenceBundle` fields that
are non-empty) is identical across all three levels in this fixture, since the
underlying bundle doesn't change between ablation levels — only what's layered on top
of it does. No improvement is claimed beyond what each level structurally exposes.

## Future optional LLM adapter — safe proposal handling

**A keyword/regex screen can never prove a sentence of free text is factually
correct.** It can only catch the most obvious disallowed claim types. This package is
built around that limitation rather than around it, with two separate output types:

- **`ReviewerBrief`** — the only *trusted* output type. Its `summary_text` is produced
  exclusively by `render_reviewer_brief(bundle, consensus)`, a fixed template
  populated only from fields that already exist on the bundle/consensus. **No LLM
  adapter output, free text, or anything else is ever substituted into it.** It
  carries no probabilities/provenance/quality/review-state fields, so there is
  structurally nowhere on the type for an altered value to go.
- **`RawProposal`** — what an `LLMAdapter.propose(bundle, consensus)` actually
  returns: untrusted free text plus claimed citations. It is never treated as
  evidence on its own.

`screen_proposal(proposal, bundle)` runs every automated check this package can
perform — non-empty, length bound, a conservative forbidden-pattern blocklist
(diagnosis language, clinical-certainty words like "confirmed"/"definite", invented
approval phrases like "approved by clinician"), and citation support (every
`cited_evidence` entry must already appear in `bundle.evidence_references`) — and can
only return one of two outcomes, both inside `QuarantinedProposal`:

| `ProposalStatus` | Meaning |
|---|---|
| `BLOCKED` | An automated check failed outright; rejected with a specific reason |
| `QUARANTINED` | Every automated check passed, but the text is **still unverified** and held for explicit human consideration |

**There is no third outcome.** Nothing in this module ever promotes a `QUARANTINED`
proposal to verified evidence, regardless of how plausible it reads — passing the
blocklist is not proof of factual support, only absence of the most obvious violations.
Even `OfflineStubAdapter`'s own deterministic, template-derived proposal is quarantined,
never auto-trusted, to keep this fail-closed by construction rather than by convention.

`LLMAdapter` is a `Protocol` any future Nebius/Nemotron- or AMD-backed implementation
must satisfy: `propose(bundle, consensus) -> RawProposal`. `OfflineStubAdapter` is the
only implementation provided; it makes no network call, no hardcoded endpoint, and
reads no API key.

## No Streamlit integration yet

This package is not wired into `hackathon/opencv_aws/streamlit_app.py` or any other
hackathon branch's UI. That integration is deliberately left for a future phase.
