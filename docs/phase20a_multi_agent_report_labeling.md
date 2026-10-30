# Phase 20A — Multi-Agent MRI Report Labeling Pilot

**Status**: Design phase (Step 1 inspection complete, ready for Step 2 implementation)  
**Effort**: Research pilot, offline only, synthetic data focus  
**Risk**: Data-use constraint requires offline processing; no external APIs permitted  
**Blocking Constraint**: Project self-imposed rule forbids sending RSNA report text to external services

---

## Step 1 — System Inspection Results

### A. Git and Workspace State

| Item | Status | Evidence |
|------|--------|----------|
| Current branch | ✅ CLEAN | `hackathon/opencv-aws @ 53786f2` (Phase 19D final) |
| Uncommitted changes | ✅ NONE | `git status --short` shows 5 Phase 18 untracked files only |
| Phase 18 files | ✅ PRESERVED | 5 files untracked (not modified): `phase18_*.py`, `test_phase18*.py` |
| Working directory | ✅ SAFE | Ready for Phase 20A without any cleanup required |

### B. Existing Report-Label System

#### Report Extraction (Baseline)
- **File**: `mri_core/report_label_rules.py`
- **Method**: Deterministic rule-based extraction (lexical, no parsing)
- **Coverage**: English + Spanish (bilingual support)
- **Version**: `rules-en-es-0.2.2` (Phase 13 audited)
- **Scope**: 12 RSNA targets (but Phase 19D uses only 3)
- **Design**: Conservative — unclear cases become `UNCERTAIN`, not `NEGATIVE`

#### Label Data Contract
- **File**: `mri_core/report_labels.py`
- **States**: `POSITIVE | NEGATIVE | UNCERTAIN | NOT_MENTIONED` (exactly one per target)
- **Evidence**: Short text span (max 160 chars) + source section + reason
- **Versioning**: Language, method, extractor_version tracked per label

#### Training Data (Phase 19D)
- **Partition**: Seed 201 (962 training, 238 validation)
- **Targets**: ACL, Medial Meniscus, Lateral Meniscus (3 targets, NOT all 12)
- **Labels**: Currently rule-based (report_label_rules v0.2.2)
- **Quality**: Weak labels; Phase 19D model achieved 0.788 macro-AUC on validation weak labels (saturation point)

### C. Existing Agents Framework

#### Purpose & Scope
- **Location**: `mri_core/agents/` (9 modules + README)
- **Function**: Process inference results (12-target model predictions) through specialist/critic/consensus workflow
- **NOT designed for**: Report text extraction or label generation

#### Reusable Components
- **contracts.py**: Data structures (frozen dataclasses, versioning, fingerprinting)
- **review_gate.py**: Workflow state machine with audit trail (transferable to label review)
- **llm_adapter.py**: LLM integration pattern (protocol + stub adapter, shows best practices)
- **evaluation.py**: Synthetic scenario testing (methodology applicable to label agents)

#### NOT applicable
- Code assumes model inference scores (probabilities dict), not report text
- Specialist/critic/consensus logic is domain-specific to 12-target model outputs
- Does not extract information from unstructured text

### D. Data-Use Authorization Status

#### Critical Constraint (Project-Imposed)
**NO EXTERNAL API CALLS WITH REAL RSNA REPORTS**

Source: `docs/kaggle_weak_label_experiment.md`
```
Our own constraint (separate from the rules): 
no radiology report text is sent to any external service.
```

Context:
- Kaggle competition rules technically permit external LLM APIs (Rule 2.6.b)
- This project chose stronger self-imposed restriction
- Applies to: Claude API, OpenAI, Anthropic, or any cloud service
- Offline open-source LLMs are not addressed in the constraint (may be permitted)

#### Consequence for Phase 20A
- ✅ CAN use synthetic RSNA-like reports for design and unit tests
- ✅ CAN document agent architecture and comparison methodology
- ❌ CANNOT process real RSNA reports through external APIs
- ❌ CANNOT claim accuracy on real data without explicit authorization review
- ⏸️ Real-data pilot requires separate authorization decision

### E. Protected Data (Must Not Access)

All of the following must be excluded from Phase 20A processing:

| Dataset | Studies | Source | Status |
|---------|---------|--------|--------|
| Gold labels (radiologist read) | 58 | `phase15_output/` | Protected |
| Competition test set | ~1000+ | Official RSNA Kaggle | Protected |
| Phase 16 holdout | 501 | `configs/phase16_registry.json` | Protected |
| Phase 19D validation | 238 | Seed-201 split (protected for evaluation) | Protected |

**Valid only**: Phase 19D training partition (962 studies) — if real-data pilot is authorized.

### F. Available Reference Data

#### Independent Labels
- Phase 19D training labels: rule-based extraction (weak, non-gold)
- Phase 19D validation labels: rule-based extraction (weak, non-gold)
- Gold labels: 58 studies (radiologist read) — available for calibration ONLY if authorized

#### Permitted for Phase 20A
- Synthetic RSNA-like reports (generate from scratch)
- Phase 19D rule-based labels (for rule-vs-agent comparison)
- Phase 19D model predictions (for indirect evaluation)
- Existing extractor code (report_label_rules.py)

---

## Step 2 — Three-Agent Architecture Design

### Agent 1: Report Extractor
**Goal**: Extract abnormalities and anatomical locations from report text  
**Model**: Claude Sonnet 5.5 (local, deterministic prompting)

**Inputs**:
- Raw report text (English or Spanish)
- Target list: [ACL, Medial Meniscus, Lateral Meniscus]

**Output** (JSON):
```json
{
  "study_uid": "1.2.826...",
  "timestamp": "2026-10-29T12:00:00Z",
  "model_id": "claude-sonnet-5-5-20241022",
  "prompt_version": "extractor-0.1",
  "findings": [
    {
      "target": "ACL",
      "state": "positive",
      "confidence": 0.95,
      "evidence_span": "ACL tear noted",
      "evidence_section": "findings",
      "location_specificity": "high",
      "hedging_flags": []
    },
    {
      "target": "Medial Meniscus",
      "state": "uncertain",
      "confidence": 0.60,
      "evidence_span": "possible medial meniscal tear",
      "evidence_section": "impression",
      "location_specificity": "medium",
      "hedging_flags": ["possible"]
    },
    {
      "target": "Lateral Meniscus",
      "state": "negative",
      "confidence": 0.90,
      "evidence_span": "lateral meniscus intact",
      "evidence_section": "findings",
      "location_specificity": "high",
      "hedging_flags": []
    }
  ],
  "language_detected": "en",
  "processing_notes": []
}
```

**Design Constraints**:
- Single pass (no iterations)
- Evidence span extraction mandatory (show your work)
- Conservative bias: unclear → `uncertain`, not `negative`
- No invented findings (must cite report text)
- Location specificity flagged (medial vs lateral confusion is a known error mode)

### Agent 2: Evidence Verifier
**Goal**: Independently validate each extraction claim against source text  
**Model**: Claude Sonnet 5.5 (local)

**Inputs**:
- Report text (original)
- Extraction output (from Agent 1)

**Output** (JSON):
```json
{
  "study_uid": "1.2.826...",
  "verification_timestamp": "2026-10-29T12:00:01Z",
  "verified_by_model": "claude-sonnet-5-5-20241022",
  "verification_findings": [
    {
      "target": "ACL",
      "extractor_state": "positive",
      "verification_status": "SUPPORTED",
      "supporting_text": "ACL tear noted",
      "is_explicit": true,
      "negation_scope": "none",
      "historical_context": false,
      "severity_qualifier": null,
      "confidence": 0.98
    },
    {
      "target": "Medial Meniscus",
      "extractor_state": "uncertain",
      "verification_status": "UNSUPPORTED",
      "supporting_text": null,
      "issue": "claim uses hedging word 'possible' but no definitive mention found elsewhere",
      "alternative_finding": "no other reference to medial meniscus",
      "confidence": 0.92
    },
    {
      "target": "Lateral Meniscus",
      "extractor_state": "negative",
      "verification_status": "SUPPORTED",
      "supporting_text": "lateral meniscus intact",
      "is_explicit": true,
      "negation_scope": "explicit_normal",
      "confidence": 0.97
    }
  ],
  "summary": {
    "supported_count": 2,
    "unsupported_count": 1,
    "consistency_issues": ["medial_meniscus_overextended"]
  }
}
```

**Design Constraints**:
- Independent review (doesn't see Agent 1's confidence scores)
- Return to source text ONLY (no inference)
- Flag negation scope, historical context, hedging
- Do not validate severity — flag for adjudication
- Report absence of supporting text explicitly (not as "negative")

### Agent 3: Adjudicator
**Goal**: Reconcile extraction and verification; escalate unresolved cases  
**Model**: Claude Sonnet 5.5 (deterministic, uses structured rules)

**Inputs**:
- Original report text
- Agent 1 extraction
- Agent 2 verification

**Output** (JSON):
```json
{
  "study_uid": "1.2.826...",
  "adjudication_timestamp": "2026-10-29T12:00:02Z",
  "final_labels": [
    {
      "target": "ACL",
      "final_state": "POSITIVE",
      "consensus": "UNANIMOUS",
      "extractor_state": "positive",
      "verifier_status": "SUPPORTED",
      "evidence_span": "ACL tear noted",
      "confidence": 0.97,
      "human_review_required": false
    },
    {
      "target": "Medial Meniscus",
      "final_state": "UNCERTAIN",
      "consensus": "RESOLVED_VIA_ESCALATION",
      "extractor_state": "uncertain",
      "verifier_status": "UNSUPPORTED",
      "issue": "extractor inferred hedging word; no independent support found",
      "resolution_rule": "verification_overrides_extraction_when_extraction_unsupported",
      "evidence_span": null,
      "confidence": 0.0,
      "human_review_required": true,
      "review_reason": "hedged_claim_without_support"
    },
    {
      "target": "Lateral Meniscus",
      "final_state": "NEGATIVE",
      "consensus": "UNANIMOUS",
      "extractor_state": "negative",
      "verifier_status": "SUPPORTED",
      "evidence_span": "lateral meniscus intact",
      "confidence": 0.97,
      "human_review_required": false
    }
  ],
  "escalated_count": 1,
  "escalation_reasons": ["hedged_claim_without_support"],
  "schema_version": "adjudicator-0.1"
}
```

**Design Constraints**:
- Apply deterministic reconciliation rules (documented, not heuristic)
- NEVER silently convert `UNCERTAIN` to `NEGATIVE`
- NEVER invent evidence when both agents agree it's absent
- Escalate disagreements to human review with specific reason
- Preserve uncertainty rather than guess (fail-safe)

---

## Step 3 — Pilot Definition

### Study Selection (Deterministic)
- **Scope**: Phase 19D training partition only (962 studies)
- **Size**: 30–50 studies for initial pilot
- **Selection method**: Deterministic hash-based sampling from training set
- **Seed**: Fixed (reproducible)
- **Exclusions**: ANY overlap with validation (238), gold (58), test, or holdout → automatic rejection

### Challenge Case Inclusion
Deliberately include:
- Negation ("no tear of the ACL")
- Uncertainty ("possible medial meniscal tear", "cannot exclude")
- Medial/lateral confusion risk (written ambiguously)
- Multilingual sections (English + Spanish)
- Missing findings (report silent on one or more targets)
- Historical context ("prior ACL reconstruction")

### Labeling Comparison
For each of the 30–50 studies:

| Source | Description | Status |
|--------|-------------|--------|
| Rule-based | Output from `report_label_rules.py` v0.2.2 | Baseline |
| Agent-derived | Three-agent consensus output | Experimental |
| Ground truth | Not available for real pilot (synthetic only) | Synthetic only |

### Metrics
- **Agreement rate**: rule-based vs agent-derived
- **Per-state distribution**: (POSITIVE, NEGATIVE, UNCERTAIN, NOT_MENTIONED)
- **Evidence completeness**: rule claims evidence span; agents must too
- **Unsupported assertion rate**: claimed findings with no source span
- **Escalation rate**: percentage flagged for human review
- **Processing time**: wall time per study

---

## Step 4 — Evaluation Plan

### Comparison: Rule-Based vs Agent-Derived

**Metrics** (descriptive, not accuracy):
1. **Label agreement** (exact match on final state per target)
2. **Disagreement patterns** (which states are confused; common errors)
3. **Evidence quality** (span extraction completeness)
4. **Confidence spread** (agents output confidence; rules don't)
5. **Escalation necessity** (how often agents flag for review)

### Accuracy Assessment
**On synthetic data**: Accuracy can be measured against hand-crafted ground truth  
**On real data**: NO accuracy claim without independent gold labels and authorization

---

## Step 5 — Implementation Safeguards

### Code Organization
```
mri_core/report_agents/
├── __init__.py
├── contracts.py              # Frozen data structures, versioning
├── extractor.py              # Agent 1 (extraction)
├── verifier.py               # Agent 2 (verification)
├── adjudicator.py            # Agent 3 (reconciliation)
├── orchestrator.py           # Three-agent pipeline
└── evaluation.py             # Comparison and metrics

tests/
├── test_report_agents.py     # Synthetic test cases
├── test_report_agents_synthetic.py
└── fixtures/
    └── synthetic_reports.py  # Generate synthetic RSNA-like reports
```

### External API Gates (DISABLED BY DEFAULT)

```python
# mri_core/report_agents/config.py

# WARNING: This project has a self-imposed constraint:
# NO REAL RSNA REPORT TEXT SHALL BE SENT TO EXTERNAL SERVICES
# This flag enables Claude API calls only for authorized pilots.

ENABLE_EXTERNAL_LLM_API = False  # MUST BE EXPLICITLY ENABLED

def validate_authorization():
    if ENABLE_EXTERNAL_LLM_API:
        # Check environment variable or authorization flag
        # Raise error if not properly authorized
        pass
```

### No Real-Data Processing by Default
- All import-time code uses synthetic reports
- Real reports only processed when:
  1. `ENABLE_EXTERNAL_LLM_API = True` (explicit)
  2. Environment variable `PHASE20A_AUTHORIZED_REAL_DATA = 1` (set)
  3. Authorization file exists and is current
- Unit tests never touch real RSNA data

### Preserving Phase 19D
- No modifications to checkpoints, predictions, or evaluation files
- No retraining ConvNeXt
- No Kaggle GPU jobs launched
- No git commits or pushes

---

## Step 6 — Deliverables

### Phase 20A Deliverables (This Document)

1. ✅ **Step 1 inspection report** (above)
   - Git state clean
   - Existing systems catalogued
   - Data-use constraint identified
   - Risk assessment: OFFLINE ONLY, no external APIs for real data

2. ✅ **Steps 2–3 design** (three-agent architecture + pilot definition)
   - Extractor, Verifier, Adjudicator agents designed
   - Data structures specified (JSON contracts)
   - 30–50 study pilot scoped
   - Challenge cases identified

3. 🔄 **Step 4–5 implementation** (pending authorization review)
   - Report agent modules (extractor, verifier, adjudicator)
   - Synthetic test suite (no real data)
   - Evaluation procedure (rule-based vs agent comparison)
   - API gates (disabled by default)

4. 📋 **Step 6 blocker list and GO/NO-GO recommendation**
   - See sections below

---

## Blockers and Authorization Requirements

### Data-Use Constraint (Active)
- **Issue**: Project forbids sending RSNA report text to external services
- **Status**: Acknowledged; design accommodates (offline architecture)
- **Solution**: Synthetic data for initial pilot; separate authorization needed for real data

### Missing Component: Real-Data Authorization
- **Requirement**: Explicit approval to process Phase 19D real reports through Claude API
- **Decision maker**: User/host (not determined in this phase)
- **Timeline**: Must be resolved before real-data pilot launches

### Missing Component: Independent Validation Labels
- **Requirement**: Real accuracy assessment needs ground-truth labels
- **Available source**: 58 gold radiologist labels (protected, evaluation-only)
- **Permission**: Not checked; likely requires separate authorization
- **Timeline**: Needed for real-data pilot

---

## GO / NO-GO Recommendation

### GO — Proceed with Synthetic Testing & Design
**Status**: ✅ **PROCEED**

**Justification**:
1. Offline architecture successfully sidesteps real-data constraint
2. Synthetic test suite is fully implementable
3. Agent design is sound and reuses existing contracts framework
4. No changes to Phase 19D or git required
5. Clear path to real-data pilot (requires authorization review)

**Next actions**:
1. Implement report_agents modules (Steps 4–5)
2. Build synthetic test suite with 5–10 realistic scenarios
3. Verify agent output contracts and reconciliation logic
4. Document authorization requirements and blockers
5. Prepare authorization request for real-data pilot

### NO-GO Condition
❌ Would recommend NO-GO only if:
- Real-data pilot were authorized without offline architecture (violates constraint)
- Implementation attempted to send real RSNA reports to external APIs

Neither applies here.

---

## References

- Existing report extraction: `mri_core/report_label_rules.py` (v0.2.2, audited Phase 13)
- Existing agents framework: `mri_core/agents/` (6 modules, agent integration patterns)
- Data constraint: `docs/kaggle_weak_label_experiment.md` (project self-imposed rule)
- Phase 19D results: `docs/phase19d_results/phase19/final_summary.json` (3-target baseline)
- Training partition: `configs/phase17_registry.json` (seed 201, 962/238 split, hashes)

---

**End of Phase 20A Specification**
