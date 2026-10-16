# AMD hackathon submission materials

Draft only — review and edit before submitting. Nothing here should be pasted verbatim without
you checking it matches what you actually want to say.

## Submission description (draft)

**Title:** MRI-Vision-Core — AMD-ready research reviewer workflow with safe LLM-assisted interpretation

**Summary (short):**
An interactive Streamlit app that runs a real trained MRI-knee abnormality model on a real
local DICOM study, walks the result through a shared, deterministic evidence-review pipeline
(specialist grouping, structural consistency checks, workflow-status consensus), and gates
every action behind an explicit, audited human-review step. An optional layer adds
LLM-generated plain-language interpretation — via Groq or AMD Token Factory — but that
interpretation is always untrusted, always quarantined, and never auto-approved or allowed to
influence the deterministic evidence or the human reviewer's decision.

**What's real, what's not (say this plainly in the submission — judges will check):**
- Real: the trained checkpoint (bridged Kaggle `efficientnet_b0` attention-MIL), real DICOM
  study inference, real evidence-review logic, real human-review state machine, real Groq and
  AMD Token Factory connectivity (each verified live exactly once, with synthetic non-medical
  data only).
- Not real / not yet done: no AMD GPU acceleration has been demonstrated anywhere in this
  project (CPU-only so far); no checkpoint for the *native* MRNet-style architecture exists on
  this machine; AMD Token Factory's legitimacy as an AMD product is unconfirmed by any official
  AMD documentation found during development — only by the one successful live call.
- Non-diagnostic throughout: every score is a raw, uncalibrated research output; nothing here
  is a clinical finding, and the UI says so repeatedly, not just once in fine print.

**Why this architecture:** the project needed to bridge an externally-trained checkpoint into a
shared evidence-review contract it wasn't built for, without modifying that shared contract —
and then needed an optional LLM layer that couldn't be allowed to leak a single byte of study
identity or override a human's judgment. Both constraints shaped every design decision (see
`hackathon/amd/README.md` for the full technical account, including the real incident where an
ambient API key almost authorized an unintended live call, and the fix that followed).

**Tech stack:** Python, PyTorch (CPU), Streamlit, `timm`/`pydicom` (checkpoint bridge),
`requests` (Groq/AMD HTTP), `python-dotenv` (optional `.env` config).

## Screenshot checklist

Capture these from a real run (real study selected, real checkpoint loaded):

1. **Sidebar, configured** — data_root/checkpoint fields prefilled (or typed), execution
   provider status visible.
2. **Scores panel** — the 12 raw probability scores + non-diagnostic disclaimer banner clearly
   visible in the same frame.
3. **Checkpoint provenance panel** — SHA-256, architecture string, "bridged, not native" label.
4. **Evidence review panel** — specialist groups, critic findings, consensus workflow status
   (ideally showing `MORE_EVIDENCE_REQUIRED`, since that's the honest, expected state here).
5. **Human review gate, before any action** — `PENDING_REVIEW`, all three buttons enabled.
6. **Human review gate, after an action** — audit history table showing an actor/action/reason
   row (approve or escalate), gate now `REVIEWED`.
7. **Optional LLM panel, Local provider** — a QUARANTINED result with "Produced by: local"
   caption, so judges see the default (no-network) path works.
8. **Optional LLM panel, external provider attempted without opt-in** — the caption showing a
   clean fallback-to-local message (e.g. "LLM_ALLOW_EXTERNAL_CALLS is not enabled") — this is
   the screenshot that proves the safety gate, which is worth highlighting explicitly.
9. **CPU benchmark panel** — real CPU latency numbers, and the "AMD GPU benchmark: not
   available" message shown honestly (don't crop this part out).
10. **`.env.example`** (code view) — to show the config surface without exposing `.env` itself.

## Video script (2–3 minutes)

**0:00–0:15 — Hook + framing**
"This is a research prototype for MRI abnormality screening — not a diagnostic tool. What I
want to show in the next two minutes is less about the model and more about the *workflow*
around it: how a real trained checkpoint, a deterministic evidence pipeline, a human-approval
gate, and an optional AI assistant all stay honestly separated from each other."

**0:15–0:45 — Real inference**
Screen: sidebar, study selection, click "Run trained inference." Narrate: "This is a real local
DICOM study, run through a real externally-trained checkpoint — bridged into our shared
evidence pipeline since it wasn't built for this architecture. These are raw, uncalibrated
scores — twelve of them — and that banner isn't boilerplate, it's load-bearing: nothing here is
a clinical finding."

**0:45–1:15 — Evidence review + human gate**
Screen: scroll to specialist/critic/consensus panel, then the review gate. Narrate: "The
workflow status here — more evidence required — is the deterministically correct answer for
this run, not a bug. Before anyone can act on this, there's an audited human-review step:
approve, request more evidence, or escalate — every action needs a named actor and a reason,
and the history is append-only." Click "Approve," show the audit row appear.

**1:15–1:55 — Optional LLM layer, safety-first**
Screen: optional LLM panel. Narrate: "There's an optional layer that can summarize this in plain
language using Groq or AMD Token Factory — both have real, live-verified connectivity. But
notice: by default, nothing leaves this machine." Toggle to show the local-only default, then
(if safe to demo) show the config-gated fallback message when an external provider is selected
without explicit opt-in. "Even when a provider is live, its output is always quarantined —
never treated as verified evidence, never allowed to touch the human-review decision."

**1:55–2:20 — Honesty about limitations**
Screen: benchmark panel showing "AMD GPU benchmark: not available." Narrate: "I want to be
upfront: everything here has run on CPU. No AMD GPU acceleration has been demonstrated yet —
that's the next step, not a claim I'm making today."

**2:20–2:30 — Close**
"Real inference, real evidence pipeline, real human gate, and an AI layer that's genuinely
optional and genuinely contained. Thanks for watching."

## Pre-submission Git review checklist

Run before the release commit (see README/commit for the actual results from this session):

- [ ] `git status` shows only the intended files.
- [ ] `.env` absent from `git status` and `git ls-files` (confirmed gitignored).
- [ ] No checkpoint/dataset binaries (`*.pt`, `*.ckpt`, `*.dcm`, etc.) tracked.
- [ ] `git diff` / `git show --stat` on the release commit reviewed line by line for anything
      that looks like a credential, internal URL, or generated output that shouldn't be there.
- [ ] `outputs/` tracked content is still only `.gitkeep`.
