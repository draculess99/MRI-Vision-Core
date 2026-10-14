# MRI Evidence Copilot — Voice Assistant Video Plan

**Target length:** 2:30 (hard limit 3:00) · **Format:** screen recording with voice-over
**Audience:** Amazon Build, Ship, Shape judges · **Tone:** non-diagnostic; evidence support for a human reviewer

> Say "Alexa-style voice assistant" and "simulation" on camera. Never say this is an Alexa Skill, diagnoses anything, or demonstrates clinical accuracy.

## Setup before recording

1. `cd D:\Work\Springboard\MRI-Vision-Core-Amazon`
2. Start the app **without** cloud keys so the local-first default is what viewers see: `Remove-Item Env:ASSEMBLYAI_API_KEY -ErrorAction SilentlyContinue; streamlit run hackathon/amazon_alexa/streamlit_app.py`
3. In the browser: select **case-002-change**, scroll to **Voice Assistant**, type a name in the **Reviewer Name** field (Step 3), allow the microphone, and test volume once.
4. Keep the evidence screenshots and `amazon_polly_voice_success.mp3` open in tabs for cutaways. Crop out the IDE chat panel and bookmarks bar from any screenshot you show full-screen.
5. Rehearse the phrases. "MRI" can be heard as "MI"; if it happens, correct it in the editable command box on camera and say so.
6. Do not run any live cloud request while recording; the AssemblyAI and Polly proof is shown from the saved evidence.

## Plan

| LEFT — screen / action | RIGHT — narration |
|---|---|
| **0:00–0:20 · Introduction.** Title card: "MRI Evidence Copilot — Alexa-style voice assistant (simulation)". Then the Streamlit page top with the non-diagnostic banner visible. | "Radiology teams need evidence summaries they can ask for out loud, without a tool ever overstepping into diagnosis. This is an Alexa-style voice assistant, a simulation, for MRI change evidence. It answers only from structured data for a synthetic case, and it can never approve anything. A named human reviewer always decides." |
| **0:20–0:50 · Live speech recognition.** Voice Assistant section, case-002-change. Point to the defaults: *Local Whisper (offline, free)* and *Browser speech (local, free)*. Click the mic, say **"Explain the review status."** Stop. Click **Transcribe recording**. Show the green "Heard: … LOCAL · Local Whisper" line and the text in the command box. | "Everything starts local and free. I record a command, push to talk, and local Whisper transcribes it on this laptop. No cloud call, no key. The text lands in an editable box, so I can correct any mishearing before anything runs." |
| **0:50–1:15 · MRI evidence and review-status explanation.** Click **Execute command**. Show Intent `explain_status` and the answer. Click **Speak response** (audio plays). Optionally type "What changed between the images?" and Execute: show the foreground 16.9% and mean-intensity 16.8 line. Cutaway for 3 s: `amazon_alexa_step1_needs_review.png`. | "The router maps my words to the existing evidence tools. Case two is flagged needs review: foreground changed sixteen point nine percent and mean intensity sixteen point eight. Those are measurements, not a clinical interpretation. The answer is read back through the speakers, and every word comes from the tool output." |
| **1:15–1:30 · Human-review request.** Say or type **"Request human review."** Execute. Show `request_human_review` and "approval state is PENDING_HUMAN_APPROVAL". Cutaway: `amazon_human_review_pending.png`. | "I can ask for a human review. It needs a reviewer's name typed into a field, never taken from speech. The case stays pending human approval." |
| **1:30–1:45 · Refused automatic approval.** Say or type **"Approve this MRI."** Execute. Show the amber `approval_refused` message. Cutaway: `amazon_auto_approval_blocked.png`. | "Now I try to approve it by voice. The assistant refuses. Approval cannot be granted by voice, and the case remains pending. There is no code path that approves a case." |
| **1:45–1:55 · Audit trail.** Expand **Audit trail (this session)**; show the command, review requested and approval refused rows. Cutaway: `amazon_approval_safeguard_audit.png`. | "Every command, review request and refusal is written to an audit trail." |
| **1:55–2:10 · Verified AssemblyAI and Amazon Polly.** Cutaway `assemblyai_cloud_success.png` (3 s, point at "Backend used: assemblyai (cloud) · Success True · calls 0 → 1"). Cutaway `amazon_polly_cloud_success.png` (3 s, "amazon_polly (cloud) · neural"). Play 4 s of `amazon_polly_voice_success.mp3`. | "I also validated both cloud services, one request each. AssemblyAI transcribed a recorded clip, and Amazon Polly's neural voice produced this audio." |
| **2:10–2:25 · Local-first cost architecture.** Expand **Usage counters & cost safeguards**. Show the STT and TTS selectors, then the AssemblyAI confirmation checkbox. | "Cloud is opt-in. AssemblyAI needs an explicit single-use confirmation, a recording is never sent twice, and Polly needs a billing acknowledgement. If either fails, it falls back to local Whisper or browser speech. Per-session limits cap usage, and they complement, not replace, provider billing controls." |
| **2:25–2:30 · Closing.** Return to the page top and banner, or a closing card: "Local first · Human approved · Non-diagnostic". | "A voice assistant that explains the evidence, runs free by default, and always leaves the decision to a human." |

## Fallbacks if something goes wrong on camera

- **Mishearing:** edit the command box and press Execute; mention that the text is editable by design.
- **Browser speech silent:** click the in-page **Speak** button once; it needs a user click in some browsers.
- **Microphone problem:** type the command; the typed path is identical after transcription.

## Claims to avoid

- Any diagnostic, clinical-accuracy or "detects disease" claim. The data is synthetic and the metrics demonstrate a workflow only.
- That this is an Alexa Skill, runs on Alexa+, uses MCP, or is deployed on AWS. None of that exists.
- Live cloud use during the recording. The cloud proof is the saved evidence.
