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
| `python -m hackathon.amd.benchmark --data-root D --checkpoint C.pt [--iterations 50 --warmup 5 --batch-sizes 1 4]` | CPU baseline vs AMD GPU on the identical checkpoint, preprocessed tensors, and config. Writes `outputs/amd/benchmark.json`. |

Add `--dicom-root` if DICOMs are not under `<data-root>/raw`.

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

## Status

See the branch hand-off notes: the trained fold checkpoints (`outputs/rsna-fivefold-benchmark/...`)
are gitignored and must be supplied; no real-checkpoint prediction or AMD GPU benchmark has been
produced yet.
