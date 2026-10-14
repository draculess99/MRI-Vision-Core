"""CPU-vs-AMD-GPU benchmark for the identical checkpoint, study input, and preprocessing.

Fairness/reproducibility rules:
  * The study is preprocessed ONCE (shared RSNAKneeDicomDataset path); the exact same
    tensors are fed to every device. Preprocessing time is reported separately and is not
    part of per-device latency.
  * The same weights are loaded into a fresh RSNAKneeCNN per device (eval mode, inference_mode).
  * Warm-up iterations are discarded; GPU timing is bracketed by torch.cuda.synchronize().
  * Metrics are measured, never estimated. If no AMD GPU is usable the "gpu" block says so and
    contains no numbers; no speedup is computed unless both devices actually ran.
  * This measures speed and resource use only. It makes no accuracy or clinical claim.

Usage:
  python -m hackathon.amd.benchmark --data-root data/rsna-knee --checkpoint last.pt \
      [--study-uid UID] [--iterations 50] [--warmup 5] [--batch-sizes 1 4] \
      [--gpu-device cuda:0] [--output outputs/amd/benchmark.json]
Exit codes: 0 ran (GPU block may be 'unavailable'), 1 error, 2 --require-gpu and no AMD GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from mri_core.rsna_inference_engine import InferenceEngineError, sha256_of_file
from mri_core.rsna_knee_dataset import TARGET_COLUMNS, RSNAKneeMetadata
from mri_core.rsna_knee_model import RSNAKneeCNN
from mri_core.rsna_knee_train import RSNAKneeDicomDataset

from ._common import add_data_args, load_metadata_and_study
from .rocm_env import check_environment, smi_snapshot

SCHEMA = "amd-benchmark/v1"
NOTE = ("Speed/resource measurement only. Probabilities are raw uncalibrated sigmoid outputs of a "
        "research model; nothing here is an accuracy, confidence, or clinical claim.")


def load_config_and_weights(checkpoint_path: Path):
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if list(data["target_columns"]) != list(TARGET_COLUMNS):
        raise InferenceEngineError("Checkpoint target_columns do not match TARGET_COLUMNS")
    return data["config"], data["model"]


def prepare_input(metadata: RSNAKneeMetadata, study_uid: str, config: dict):
    rows = metadata.train[metadata.train["StudyInstanceUID"].astype(str) == str(study_uid)]
    if rows.empty:
        raise InferenceEngineError(f"StudyInstanceUID={study_uid} has no labeled row in metadata")
    started = time.perf_counter()
    item = RSNAKneeDicomDataset(metadata, rows, config, split="train")[0]
    seconds = time.perf_counter() - started
    return item["images"].unsqueeze(0), item["mask"].unsqueeze(0), seconds


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(seconds: list[float]) -> dict:
    ordered = sorted(seconds)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "mean_ms": statistics.fmean(seconds) * 1e3,
        "median_ms": statistics.median(seconds) * 1e3,
        "p95_ms": p95 * 1e3,
        "min_ms": ordered[0] * 1e3,
        "max_ms": ordered[-1] * 1e3,
        "stdev_ms": (statistics.stdev(seconds) * 1e3) if len(seconds) > 1 else 0.0,
    }


def _process_rss_bytes() -> Optional[int]:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def bench_device(state_dict, config, images, mask, device_name: str, *, iterations: int,
                 warmup: int, batch_sizes: list[int]) -> tuple[dict, list[float]]:
    device = torch.device(device_name)
    planes = tuple(config.get("planes", ("Axial", "Coronal", "Sagittal")))
    model = RSNAKneeCNN(num_planes=len(planes), dropout=float(config["dropout"]))
    model.load_state_dict(state_dict)
    model.eval().to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rss_before = _process_rss_bytes()

    result: dict = {"device": device_name, "batches": {}}
    reference: list[float] = []
    for batch_size in batch_sizes:
        x = images.repeat(batch_size, *([1] * (images.dim() - 1))).to(device)
        m = mask.repeat(batch_size, *([1] * (mask.dim() - 1))).to(device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(x, m)
            _sync(device)
            laps = []
            for _ in range(iterations):
                _sync(device)
                started = time.perf_counter()
                logits = model(x, m)
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
            reference = logits[0].sigmoid().cpu().tolist()

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
    return result, reference


def run_benchmark(metadata: RSNAKneeMetadata, checkpoint: Path, study_uid: str, *, iterations: int,
                  warmup: int, batch_sizes: list[int], gpu_device: str,
                  allow_non_amd: bool = False) -> dict:
    environment = check_environment()
    config, state_dict = load_config_and_weights(checkpoint)
    images, mask, preprocessing_seconds = prepare_input(metadata, study_uid, config)
    kwargs = dict(iterations=iterations, warmup=warmup, batch_sizes=batch_sizes)

    cpu, cpu_probs = bench_device(state_dict, config, images, mask, "cpu", **kwargs)

    gpu_ok = environment["amd_gpu_available"] or (allow_non_amd and environment["accelerator_available"])
    if gpu_ok:
        gpu, gpu_probs = bench_device(state_dict, config, images, mask, gpu_device, **kwargs)
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
            "study_uid": study_uid,
            "input_images_shape": list(images.shape),
            "input_mask_shape": list(mask.shape),
            "model_config": config,
            "preprocessing_seconds_once": preprocessing_seconds,
            "batch_sizes": batch_sizes,
        },
        "cpu": cpu,
        "gpu": gpu,
        "comparison": comparison,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_data_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, help="torch.set_num_threads for the CPU baseline")
    parser.add_argument("--require-gpu", action="store_true", help="exit 2 if no AMD GPU is usable")
    parser.add_argument("--allow-non-amd", action="store_true",
                        help="dev only: allow a non-AMD accelerator; result is labeled is_amd_rocm=false")
    parser.add_argument("--output", default="outputs/amd/benchmark.json")
    args = parser.parse_args(argv)

    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    try:
        metadata, study_uid = load_metadata_and_study(args.data_root, args.dicom_root, args.study_uid)
        report = run_benchmark(
            metadata, Path(args.checkpoint), study_uid, iterations=args.iterations, warmup=args.warmup,
            batch_sizes=args.batch_sizes, gpu_device=args.gpu_device, allow_non_amd=args.allow_non_amd,
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
