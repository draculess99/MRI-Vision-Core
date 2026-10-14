"""CPU-vs-AMD-GPU benchmark for the bridged, externally-trained Kaggle attention-MIL checkpoint.

A separate, independent CLI mirroring `hackathon/amd/benchmark.py`'s fairness rules and output
schema, applied to the checkpoint bridged in `kaggle_mil_bridge.py` (timm efficientnet_b0 +
gated attention-MIL, not the native `RSNAKneeCNN`). `benchmark.py` is not imported for its model
logic and is not modified by this module -- the native RSNAKneeCNN benchmark path is unaffected.
Only generic, model-agnostic timing/memory helpers (`_sync`, `_summary`, `_process_rss_bytes`)
are reused from it to avoid duplicating that bookkeeping.

Fairness/reproducibility rules (identical in spirit to benchmark.py):
  * The DICOM series is decoded and preprocessed ONCE; the exact same tensor is fed to every
    device and every batch size (repeated along the batch axis). Preprocessing time is reported
    separately and is not part of per-device latency.
  * A fresh model is loaded from the checkpoint for each device (eval mode, inference_mode).
  * Warm-up iterations are discarded; GPU timing is bracketed by torch.cuda.synchronize().
  * Metrics are measured, never estimated. If no AMD GPU is usable the "gpu" block says so and
    contains no numbers; no speedup is computed unless both devices actually ran.
  * A CPU-only run is never reported as an AMD GPU benchmark: the "gpu" block is only populated
    when an accelerator actually ran, and `is_amd_rocm` reflects the real environment check, not
    the device string used.
  * This measures speed and resource use only. It makes no accuracy or clinical claim, and the
    checkpoint itself has not been evaluated or calibrated by this project (see kaggle_mil_bridge.py).

Usage:
  python -m hackathon.amd.benchmark_kaggle_bridge --data-root <real rsna-knee dir> \
      --checkpoint <foldN-knee-mil-best.ckpt> [--study-uid UID] [--iterations 50] \
      [--warmup 5] [--batch-sizes 1 4] [--gpu-device cuda:0] \
      [--output outputs/amd/benchmark_kaggle_bridge.json]
Exit codes: 0 ran (gpu block may be 'unavailable'), 1 error, 2 --require-gpu and no AMD GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from mri_core.rsna_inference_engine import sha256_of_file

from .benchmark import _process_rss_bytes, _summary, _sync
from .kaggle_mil_bridge import load_bridge_checkpoint, read_dicom_volume
from .rocm_env import check_environment, smi_snapshot
from .run_real_checkpoint_smoke import pick_study_and_series

SCHEMA = "amd-benchmark-kaggle-bridge/v1"
NOTE = (
    "Speed/resource measurement only, for the bridged Kaggle attention-MIL checkpoint "
    "(see kaggle_mil_bridge.py), not the native mri_core.RSNAKneeCNN path. Probabilities are "
    "raw, uncalibrated sigmoid outputs of an externally trained, unevaluated research model; "
    "nothing here is an accuracy, confidence, or clinical claim."
)


def bench_device(checkpoint_path: Path, volume: torch.Tensor, device_name: str, *, iterations: int,
                  warmup: int, batch_sizes: list[int]) -> tuple[dict, list[float], dict]:
    """Run the full warmup+timed loop for one device; returns (result, batch0 probs, provenance).

    Loads a fresh model from the checkpoint for this device, matching benchmark.py's rule that
    no state is shared across devices.
    """
    model, model_provenance = load_bridge_checkpoint(checkpoint_path)
    device = torch.device(device_name)
    model.to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = _process_rss_bytes()

    result: dict = {"device": device_name, "batches": {}}
    reference: list[float] = []
    for batch_size in batch_sizes:
        x = volume.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
            _sync(device)
            laps = []
            for _ in range(iterations):
                _sync(device)
                started = time.perf_counter()
                logits = model(x)
                _sync(device)
                laps.append(time.perf_counter() - started)
        total = sum(laps)
        result["batches"][str(batch_size)] = {
            "iterations": iterations,
            "warmup": warmup,
            "latency": _summary(laps),
            "throughput_studies_per_s": batch_size * iterations / total,
        }
        if batch_size == batch_sizes[0]:
            reference = torch.sigmoid(logits[0]).cpu().tolist()

    if device.type == "cuda":
        free, total_mem = torch.cuda.mem_get_info(device)
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": props.name,
            "gcn_arch": getattr(props, "gcnArchName", None),
            "total_memory_bytes": int(total_mem),
            "free_memory_bytes_after": int(free),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "smi_snapshot": smi_snapshot(),
        }
    else:
        rss_after = _process_rss_bytes()
        result["cpu"] = {
            "torch_threads": torch.get_num_threads(),
            "process_rss_bytes_before": rss_before,
            "process_rss_bytes_after": rss_after,
            "note": None if rss_after is not None else "psutil not installed; RSS not recorded",
        }
    return result, reference, model_provenance


def run_benchmark(data_root: Path, checkpoint: Path, study_uid: Optional[str], *, iterations: int,
                   warmup: int, batch_sizes: list[int], gpu_device: str, allow_non_amd: bool = False) -> dict:
    environment = check_environment()

    resolved_study_uid, series_uid, plane = pick_study_and_series(data_root, study_uid)
    series_dir = data_root / "raw" / "train_series" / resolved_study_uid / series_uid

    started = time.perf_counter()
    volume = read_dicom_volume(series_dir)
    preprocessing_seconds = time.perf_counter() - started

    kwargs = dict(iterations=iterations, warmup=warmup, batch_sizes=batch_sizes)
    cpu, cpu_probs, model_provenance = bench_device(checkpoint, volume, "cpu", **kwargs)

    gpu_ok = environment["amd_gpu_available"] or (allow_non_amd and environment["accelerator_available"])
    if gpu_ok:
        gpu, gpu_probs, _ = bench_device(checkpoint, volume, gpu_device, **kwargs)
        gpu["is_amd_rocm"] = environment["amd_gpu_available"]
        gpu["max_abs_prob_diff_vs_cpu"] = max(abs(a - b) for a, b in zip(cpu_probs, gpu_probs))
        b0 = str(batch_sizes[0])
        comparison = {
            "basis": f"batch_size={b0}, median latency",
            "gpu_speedup_vs_cpu": cpu["batches"][b0]["latency"]["median_ms"] / gpu["batches"][b0]["latency"]["median_ms"],
        }
    else:
        gpu = {"status": "unavailable", "reasons": environment["unavailable_reasons"]}
        comparison = {"status": "not_computed", "reason": "no AMD GPU run; no GPU numbers exist"}

    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": NOTE,
        "environment": environment,
        "workload": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_of_file(checkpoint),
            "model_provenance": model_provenance,
            "data_root": str(data_root),
            "study_uid": resolved_study_uid,
            "series_uid": series_uid,
            "plane": plane,
            "input_volume_shape": list(volume.shape),
            "preprocessing_seconds_once": preprocessing_seconds,
            "batch_sizes": batch_sizes,
        },
        "cpu": cpu,
        "gpu": gpu,
        "comparison": comparison,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="RSNA knee data dir (train.csv, raw/train_series/...)")
    parser.add_argument("--checkpoint", required=True, help="Path to a foldN-knee-mil-best.ckpt file")
    parser.add_argument("--study-uid", help="StudyInstanceUID; default: first fully downloaded labeled study")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, help="torch.set_num_threads for the CPU baseline")
    parser.add_argument("--require-gpu", action="store_true", help="exit 2 if no AMD GPU is usable")
    parser.add_argument("--allow-non-amd", action="store_true",
                        help="dev only: allow a non-AMD accelerator; result is labeled is_amd_rocm=false")
    parser.add_argument("--output", default="outputs/amd/benchmark_kaggle_bridge.json")
    args = parser.parse_args(argv)

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    try:
        report = run_benchmark(
            Path(args.data_root), Path(args.checkpoint), args.study_uid, iterations=args.iterations,
            warmup=args.warmup, batch_sizes=args.batch_sizes, gpu_device=args.gpu_device,
            allow_non_amd=args.allow_non_amd,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    gpu_state = report["gpu"].get("status", "ran")
    print(f"wrote {out}  cpu=ran gpu={gpu_state}")
    return 2 if (args.require_gpu and gpu_state == "unavailable") else 0


if __name__ == "__main__":
    sys.exit(main())
