# MRI Evidence Copilot — Amazon Alexa+ Video Walkthrough

**Duration:** 2 minutes 30 seconds  
**Target Audience:** Amazon Build, Ship, Shape hackathon judges  
**Format:** Screen recording with narration

---

## Video Sequence & Timing

### [0:00–0:15] — Title Card

**Screen:** White background with title text overlay
```
MRI Evidence Copilot
Amazon Alexa+ (Simulated)
Local-only • Non-diagnostic • Human-approved
```

**Narration:**
"Clinical workflows often need evidence summaries without requiring external APIs or AI inference. The MRI Evidence Copilot is a deterministic, local-only tool layer that surfaces structured evidence to clinicians while maintaining a human-in-the-loop approval safeguard."

---

### [0:15–0:35] — Problem & Safety Boundary

**Screen:** Static slide with bullet points

```
The Problem:
• Voice-first tools need fast, deterministic responses
• Offline environments cannot depend on external APIs
• Approval decisions must never bypass a human reviewer
• Evidence interpretation must not be confused with diagnosis

The Safety Boundary:
✓ Deterministic evidence calculation
✓ Non-diagnostic explanations
✓ Local computation only (no network calls)
✓ Human approval is non-bypassable
```

**Narration:**
"Clinicians need tools that explain evidence without overstepping into diagnosis. The copilot is designed with a clear safety boundary: it surfaces evidence facts and status, but it cannot approve cases. Only a named human reviewer can do that."

---

### [0:35–1:05] — Simulated Alexa+ Interface & Three Tools

**Screen:** Streamlit app showing the three-step interface

**Narration:**
"This is the simulated Alexa+ interface. It exposes three deterministic tools—all running locally with no external API calls. First, analyze MRI evidence loads a demonstration case with synthetic data. Second, explain review status describes the findings in natural language without diagnosing. Third, request human approval records that a human reviewer is required—but the copilot itself cannot approve the case."

**Screen Directions:**
- Show the Streamlit app landing page with the three sections: "Step 1: Analyze MRI Evidence," "Step 2: Explain Review Status," "Step 3: Request Human Approval"
- Hover over or highlight each tool name briefly

---

### [1:05–1:40] — Case 002 Analysis (NEEDS_REVIEW)

**Screen:** Case-002-change selected in the demo

**Narration:**
"Let's walk through Case 002, which shows a large change from baseline. The copilot analyzes the evidence and reports a status of NEEDS_REVIEW. Notice the evidence metrics: the foreground percentage changed by 16.9%, mean intensity by 16.8. These are purely numerical observations—the copilot does not interpret whether this is clinically significant. That interpretation requires a human expert."

**Screen Directions:**
- Open the demo and select "Case 002: Large-change review"
- Show the metrics panel: Case ID, Status (NEEDS_REVIEW), Quality Score, Foreground %, Mean Intensity
- Display the Change Description: "Foreground changed by 16.9%, mean intensity by 16.8. Review needed."

**Insert Screenshot:**
```
📸 docs/images/amazon_alexa_demo/amazon_alexa_step1_needs_review.png
Caption: "Step 1: Analyze MRI Evidence — Case 002 shows deterministic change evidence"
```

---

### [1:40–2:10] — Explain Status & Request Approval

**Screen:** Step 2 and Step 3 of the same case

**Narration:**
"Step 2 explains the review status in natural language. The summary says: 'Current image shows notable differences from baseline.' No diagnosis. No clinical conclusion. Then Step 3 requests approval from a named human reviewer. The approval request is recorded as PENDING_HUMAN_APPROVAL. Notice the critical safeguard: the approval state always remains PENDING on the copilot side. The copilot cannot approve. Only the named human reviewer—in this case, Dr. Smith—can make that decision in a live system."

**Screen Directions:**
- Expand the "Quality Assessment" and "Change Evidence" sections in Step 2
- Show the summary text and quality/change notes
- Scroll to Step 3 and show the reviewer name input field
- Display the success message with "Approval State: PENDING_HUMAN_APPROVAL"

**Insert Screenshot:**
```
📸 docs/images/amazon_alexa_demo/amazon_alexa_step3_pending_human_approval.png
Caption: "Step 3: Request Human Approval — Approval state is non-bypassable PENDING"
```

---

### [2:10–2:30] — Amazon Track & Key Takeaways

**Screen:** Slide with comparison and key points

```
What's New for Amazon Alexa+ Track:
• Deterministic tool layer (not multi-agent consensus)
• Voice-first design (simple, fast responses)
• No LLM inference required
• Local-only (no credentials, no network)
• Synthetic demonstration data only

Human-in-the-Loop Boundary:
✓ Evidence & status: Copilot provides
✓ Approval decision: Human only
✓ This safeguard is non-negotiable
```

**Narration:**
"The Amazon Alexa+ track focuses on simple, deterministic tools that work offline. Unlike other hackathon projects, this doesn't require multi-agent consensus or LLM inference. It's designed for voice interfaces that need fast, reproducible responses. But the core principle is unchanged: human approval is always required and cannot be bypassed by the system."

**Screen Directions:**
- Show the comparison slide between Alexa+ track and other tracks
- Highlight the human approval safeguard as the final point

---

## Recording Checklist

- [ ] Test all tools load correctly before recording
- [ ] Ensure Streamlit app runs without errors: `streamlit run hackathon/amazon_alexa/streamlit_app.py`
- [ ] Select Case 002 for the demo (deterministic, shows clear change evidence)
- [ ] Check that both screenshots are visible and named correctly:
  - [ ] `docs/images/amazon_alexa_demo/amazon_alexa_step1_needs_review.png`
  - [ ] `docs/images/amazon_alexa_demo/amazon_alexa_step3_pending_human_approval.png`
- [ ] Time the narration to fit within 2:30 (adjust pacing if needed)
- [ ] Verify no external network calls occur during recording
- [ ] Confirm all text is readable (use 1.5x–2x zoom if needed)
- [ ] Test narration audio for clarity and background noise

---

## Key Points to Emphasize During Recording

1. **Synthetic Data Only:** "All evidence shown is synthetic demonstration data, not real patient information."

2. **Non-Diagnostic:** "The copilot explains evidence and status. It does not make clinical diagnoses or treatment recommendations."

3. **Local & Offline:** "No external API calls, no LLM inference, no credentials required. Everything runs locally."

4. **Human Approval Safeguard:** "The copilot cannot approve cases. Every case starts in PENDING_HUMAN_APPROVAL. Only a named human reviewer can transition to APPROVED_BY_HUMAN."

5. **Deterministic:** "Same input always produces the same output. Safe for testing, auditing, and compliance review."

---

## Notes for Judges

- This is a **proof-of-concept** for the Amazon hackathon.
- The three tools are fully typed and deterministic.
- The Streamlit interface demonstrates how voice-friendly tools can expose evidence to clinicians.
- All code is in `hackathon/amazon_alexa/` with 30 passing tests.
- The human approval safeguard is non-bypassable by design.
- Ready for extension with live MRI image input and optional Alexa voice integration in future iterations.

---

## Total Word Count (Narration): ~420 words
**Estimated Speaking Time:** 2 minutes 30 seconds (at ~100 words/min pace)
