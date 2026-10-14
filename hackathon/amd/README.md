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
