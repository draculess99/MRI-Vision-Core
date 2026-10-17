# MRI-Vision-Core — AMD AI Review Workflow

A research/demo MRI AI workflow: trained MRI inference on a real DICOM study, deterministic
multi-stage evidence review, an optional LLM-assisted plain-language explanation, reproducible
benchmarking across 22 curated studies, and an explicit human-approval gate that no automated
layer can bypass.

> **Research/demo benchmark only; not clinical accuracy or medical performance.** Every score
> below is a raw, uncalibrated sigmoid output — never a diagnosis or a confidence score.

## System architecture

```mermaid
flowchart TD
    MRI["MRI study<br/>real local DICOM series"] --> PRE
    PRE["Preprocessing + trained MRI inference<br/>(externally-trained checkpoint,<br/>bridged into this app)"] --> SCORES
    SCORES["12 research model scores<br/>raw, uncalibrated sigmoid outputs"] --> BUNDLE
    BUNDLE["Structured evidence<br/>(EvidenceBundle)"] --> AGENTS
    AGENTS["Specialist -> Critic -> Consensus<br/>deterministic evidence review"] --> GATE
    GATE["Human Review Gate<br/>approve / request more evidence / escalate"]

    BUNDLE -.-> LLM
    AGENTS -.-> LLM
    LLM{"Optional LLM provider<br/>AMD Token Factory or Groq<br/>(local fallback if unavailable/disallowed)"}
    LLM -.-> EXPLAIN["Reviewer-facing explanation<br/>(display only)"]
    EXPLAIN -.-> GATE
```

The dotted path is the point of this diagram: the optional LLM layer reads the same structured
evidence a reviewer sees, but it cannot change the 12 model scores, cannot change the
deterministic Consensus result, and cannot move the Human Review Gate — that only happens when
a named human clicks a button with a reason attached.

## How the review agents work together

![AMD MRI-Core review agent flow](../../docs/images/amd_review_agent_flow.png)

All three agents are deterministic, pure functions over one `EvidenceBundle` — they do not call
each other directly; an orchestrator runs them in sequence and passes each stage's output to the
next.

### Specialist
Groups the 12 raw model scores into four fixed categories (Ligaments, Menisci, Osteoarthritis,
Other) and restates each score verbatim. It does not rank, highlight, or interpret any value —
no positive/negative decision is made at this layer.

### Critic
Runs a fixed set of deterministic structural checks against the evidence: label order and value
range, finite probabilities, checkpoint provenance present, model-count/ensemble consistency, and
whether the deterministic image-quality gate has even been run. A passing check is never treated
as a safety signal — it only means the evidence is internally well-formed.

### Consensus
Applies a fixed, ordered rule set over the Critic's findings and the bundle's own inference/
quality status to produce exactly one workflow status (e.g. `MORE_EVIDENCE_REQUIRED`,
`EVIDENCE_COMPLETE_FOR_RESEARCH_REVIEW`, `INCONSISTENT_EVIDENCE`, `INFERENCE_UNAVAILABLE`). The
Specialist's summaries are carried alongside this result for display; the status itself is
derived from the Critic's findings and the bundle's state. Consensus is decision support, not
approval — it always initializes a fresh human-review workflow in a pending state.

### Human Review Gate
There is no agent auto-approval path. Every result, regardless of its workflow status, passes
through the same gate, and the gate only changes state when a named human clicks **Approve**,
**Request more evidence**, or **Escalate** with a required, non-empty reason — through
`mri_core.agents.review_gate.ReviewGate` (shared, unmodified across this project's other
branches). The audit trail is append-only.

### Optional LLM explanation
The Specialist, Critic, and Consensus workflow produces structured review evidence. An optional
LLM provider can turn that evidence into a concise, reviewer-facing explanation. The provider
cannot alter the underlying MRI scores or bypass the human-review gate: only a strict allowlist
of already-deterministic fields is ever sent externally (never the study UID, checkpoint path, or
any local file path), and the provider's output is run through a shared screening gate that can
only return **BLOCKED** or **QUARANTINED** — there is no "verified" outcome, and nothing it
produces is read by the review-gate code at all.

## Reproducible 22-study benchmark

A runner (`scripts/run_amd_benchmark.py`) walks all 22 curated RSNA MRI scenarios, checks each
one's underlying study against the locally configured data root, and runs real inference only
where the study is actually present — never fabricating a result for a missing study.

**Latest measured run, all 22 studies available locally:**

| Metric | Value |
|---|---|
| Curated scenarios | 22 |
| Successfully inferred | 22/22 |
| Inference coverage | 100% |
| Failures | 0 |
| Macro ROC-AUC | 0.599 |
| Macro PR-AUC | 0.485 |
| Mean latency | 0.987 s |
| Median latency | 0.961 s |
| Total inference wall time | 21.723 s |

Per-target ROC-AUC, reported honestly rather than hidden — this checkpoint separates some
findings well and others not at all:

| Stronger | ROC-AUC | Weaker | ROC-AUC |
|---|---|---|---|
| Synovitis | 0.752 | PF OA | 0.324 |
| Medial OA | 0.736 | ACL | 0.419 |
| Fracture | 0.695 | MCL | 0.458 |
| Lateral Meniscus | 0.671 | | |
| Baker's | 0.667 | | |

The benchmark scores against the RSNA competition's own **competition reference labels** — not a
diagnosis — and every probability is a raw, uncalibrated sigmoid output. This is a research
evaluation of one specific bridged checkpoint, not clinical validation, and not a statement about
the underlying clinical task in general.

Full results: [`amd_benchmark_latest.md`](../../benchmarks/amd/results/amd_benchmark_latest.md) ·
[`amd_benchmark_latest.json`](../../benchmarks/amd/results/amd_benchmark_latest.json)

```bash
python scripts/run_amd_benchmark.py --dry-run   # check local availability only, no inference
python scripts/run_amd_benchmark.py             # run real inference where available
```

## LLM provider layer

This is a provider/integration layer, not a second predictive benchmark — the LLM only produces
an optional explanation of evidence the deterministic layers above already computed.

| Provider | Integration | Live validation | Role |
|---|---|---|---|
| **AMD Token Factory** | Real network client (chat-completions, OpenAI-compatible, strict JSON-schema response validation) | A real live call was made during development against this account's endpoint, with synthetic, non-medical data. The regression suite captures an actual oversized response that call returned, and the stricter prompt/length check it drove — repo evidence of a real round trip, not a claim alone. Not independently re-verifiable from this repo without Token Factory access. | Optional reviewer-facing explanation |
| **Groq** | Real network client, same redaction/screening path | A real API key was tested against this endpoint during development (the default model in the adapter was chosen after a different one 404'd for that key). | Optional reviewer-facing explanation |
| **Local (default)** | Deterministic template, no network call, no credential | Always available; covered directly by the test suite | Default, and the automatic fallback whenever an external provider is unavailable or not allowed |

Providers are interchangeable by design: the MRI inference and the deterministic Specialist/
Critic/Consensus/Human-Review-Gate logic never change when the LLM provider changes — only which
text fills the optional explanation panel does.

## Why this architecture matters

This submission keeps four concerns separate: **(1)** trained MRI inference, **(2)** deterministic
evidence/review logic, **(3)** optional generative explanation, **(4)** human authorization. That
separation is what makes the system measurable (the benchmark above) and auditable (the review
gate's append-only history) — it is not a claim about medical relevance.

## Running it locally

```bash
pip install -r requirements-amd.txt
cp .env.example .env    # optional — LLM provider config
streamlit run hackathon/amd/streamlit_app.py
```

Real MRI inference needs a local RSNA knee `data_root` and a bridged checkpoint (`AMD_KAGGLE_RSNA_ROOT`
/ `AMD_KAGGLE_CHECKPOINT`, environment or `.env`). The deployed application does **not** bundle any
raw DICOM data or model weights — without those configured, a synthetic workflow demonstration
panel still exercises the full evidence-review/human-review/LLM pipeline with fixed, clearly
labeled placeholder values.

## Limitations

- Benchmarked against 22 curated studies, not the full competition dataset.
- Model outputs are raw, uncalibrated probabilities — not clinical confidence scores.
- The checkpoint was trained externally and bridged into this app, not trained here.
- The 22-study benchmark is a research evaluation, not clinical validation.
- All measured inference and benchmarking has been CPU-only; no AMD GPU acceleration has been
  exercised in this project.
- The optional LLM explanation is a text-generation convenience, not a diagnostic component, and
  cannot influence any score, workflow status, or approval decision.

## Result

Built a reproducible MRI AI evaluation pipeline across 22 real studies: 22/22 successful
inference runs, 0 failures, macro ROC-AUC 0.599, macro PR-AUC 0.485, and a 0.987-second mean
inference latency — evaluated against competition reference labels, not diagnoses.

*Research/demo benchmark only; not clinical accuracy or medical performance.*
