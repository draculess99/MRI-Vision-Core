# Pre-Video Release Checklist — Amazon MRI Voice Assistant

Branch: `hackathon/amazon-alexa-copilot` · Gate run: 2026-10-13 · Nothing has been committed or pushed.

**Overall status: READY FOR VIDEO**, with one missing evidence file (non-blocking for recording; see Missing Artifacts).

Legend: **PASS** verified in this gate (test, command or inspection) · **PASS (evidence)** verified from a saved screenshot of a manual run · **FAIL** · **NOT TESTED**

## Test totals

| Scope | Passed | Failed | Skipped |
|---|---|---|---|
| Full suite (`python -m pytest -q`) | **185** | **0** | **0** |
| Amazon-specific (5 files) | 115 | 0 | 0 |

Per file: approval 9 · tools 21 · voice 48 · voice session 35 · voice UI 2. The 4 warnings are a Pydantic deprecation inside the installed `assemblyai` SDK. All pre-existing tests (100) still pass. No test can reach a paid service: `tests/conftest.py` blocks AssemblyAI and Polly and removes credentials from the environment.

## Capability assessment

| Capability | Result | Basis |
|---|---|---|
| Streamlit starts | PASS | Launched headless on port 8599 with no API key; health endpoint returned `ok`; server then stopped |
| Defaults to free local speech | PASS | Selectors default to Local Whisper and browser speech (tests + screenshots) |
| Microphone recording / session state | PASS | 35 session tests (valid, missing, `DeletedFile`, reruns); recording waveform visible in screenshots |
| Local Whisper transcription | PASS | Decoded and transcribed the Polly MP3 locally; "Heard … LOCAL · Local Whisper" in screenshots |
| AssemblyAI cloud adapter (mocked) | PASS | Mocked success, auth failure, classification, fallback, redaction tests |
| AssemblyAI live (one request) | PASS (evidence) | `assemblyai_cloud_success.png`: backend `assemblyai (cloud)`, success, calls 0→1, 2.0 s |
| Browser speech | PASS (evidence) | Backend selection tested; user-reported manual playback. Screenshot shows a harmless "interrupted" notice. Audio output itself not verified here |
| Amazon Polly adapter (mocked) | PASS | Opt-in, voice/engine, failure fallback, limits, cache tests |
| Amazon Polly live (one request) | PASS (evidence) | `amazon_polly_cloud_success.png`: neural Joanna, success, requests 0→1, 78 chars; MP3 decodes (24 kHz, 5.6 s) |
| Explicit confirmation before cloud requests | PASS | Single-use AssemblyAI confirmation; Polly acknowledgement; tests prove no call without them |
| Duplicate paid-request prevention | PASS | Same recording never re-sent; identical Polly text cached; call/char/second limits |
| Cloud failure fallback | PASS | Local Whisper and browser speech fallbacks tested; UI cannot crash on speech errors |
| Voice command routing | PASS | All six commands plus unknown commands |
| Human approval protection | PASS | Seven approval phrasings refused; no `APPROVED_BY_HUMAN` path; reviewer name never taken from speech |
| Pending approval preserved | PASS | Tests + `amazon_human_review_pending.png` |
| Audit trail recording | PASS | Tests + `amazon_approval_safeguard_audit.png` |
| Three synthetic cases | PASS | Tool, router and report tests parametrised over all three |
| Accurate evidence explanations | PASS | Responses asserted equal to tool outputs; none generated |
| No unsupported clinical claims | PASS | Non-diagnostic notice on every response and in the README; claims reviewed in script and README |
| Credentials not exposed | PASS | Repository scan for the AssemblyAI key and AWS key patterns found nothing; screenshots inspected show only lengths and counters; diagnostics print fingerprints only |
| MCP server / AWS deployment / real Alexa Skill | NOT TESTED | Not implemented; the README says so |
| Real patient data | NOT TESTED | Not used; all data is synthetic |

## Confirmed evidence paths

Images in `docs/images/amazon_alexa_demo/`:
- `assemblyai_cloud_success.png`
- `amazon_polly_cloud_success.png`
- `amazon_human_review_pending.png`
- `amazon_auto_approval_blocked.png`
- `amazon_approval_safeguard_audit.png`
- `amazon_alexa_step1_needs_review.png`, `amazon_alexa_step3_pending_human_approval.png` (original)

Audio: `docs/audio/amazon_alexa_demo/amazon_polly_voice_success.mp3` (33,740 bytes; decodes; not excluded by `.gitignore`). No `.gitignore` rule matches `*.png`, `*.mp3` or `*.wav`.

## Missing artifacts

- **`docs/images/amazon_alexa_demo/amazon_approval_audit_trail.png` does not exist.** No replacement was invented. The audit trail is visible in `amazon_approval_safeguard_audit.png`, which the README and video plan use. If a separate file is required, capture a new screenshot of the audit-trail expander.

## Current limitations

- Alexa-style simulation: no Alexa Skill, device, MCP server or AWS deployment.
- Speech recognition mishears "MRI" as "MI" (both local and AssemblyAI); the command box is editable.
- Screenshots `assemblyai_cloud_success.png` and `amazon_polly_cloud_success.png` show the IDE with a chat side panel and bookmarks bar; no secrets are visible, but crop them if shown full-screen.
- The review screenshots show a reviewer name typed by the user ("wil low"); confirm that is acceptable to publish.
- Stray local artifacts in the repository root: `clip.wav` and `polly_test.mp3` (duplicates of live-test outputs). Not deleted, to preserve evidence; do not commit them.
- Usage counters are per session and do not replace provider billing controls.
- Browser speech audio was not verified in this gate.

## Launch commands (PowerShell)

```powershell
cd D:\Work\Springboard\MRI-Vision-Core-Amazon
python -m pytest -q
streamlit run hackathon/amazon_alexa/streamlit_app.py
```

For the recording, start without cloud keys: `Remove-Item Env:ASSEMBLYAI_API_KEY -ErrorAction SilentlyContinue` first.

## Video preparation status

- `VIDEO_SCRIPT.md`: updated, 2:30 two-column plan covering all nine required segments. The previous local-only script is preserved as `VIDEO_SCRIPT_v1_local_only.md`.
- Not done (by instruction): the video has not been recorded.

## Remaining submission requirements

Not verified here, because hackathon rules were not provided; confirm each against the submission page:
- Record and upload the video.
- Commit and push the branch (needs your explicit approval): add the untracked source, tests, docs and evidence; leave out `clip.wav` and `polly_test.mp3`.
- Decide on the missing `amazon_approval_audit_trail.png`.
- Crop or retake screenshots that show the IDE chat panel.
- Write the submission description using the README's limits and non-diagnostic wording.
