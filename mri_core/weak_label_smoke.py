"""Synthetic GPU/AMP smoke test: needs no competition data, no manifest and no network.

For each arm it trains on random tensors, interrupts mid-epoch, resumes from the checkpoint, verifies the AMP scaler
state round-trips and the durable export matches byte-for-byte, and reports GPU memory, time and throughput. A pass here
says the code path works on this hardware; it says nothing about model quality.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .rsna_knee_dataset import TARGET_COLUMNS
from .weak_label_models import ARMS, build_model
from .weak_label_train import DEFAULT_CONFIG, InMemoryWeakDataset, summarize_runtime, train_weak_label


class _Interrupt(RuntimeError):
    pass


def environment_report() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda, "gpus": [], "bf16_supported": None, "cudnn": None}
    if torch.cuda.is_available():
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        info["cudnn"] = torch.backends.cudnn.version()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": p.name, "memory_gb": round(p.total_memory / 1e9, 1),
                                 "capability": f"{p.major}.{p.minor}"})
        if shutil.which("nvidia-smi"):
            try:
                info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
            except Exception:
                info["nvidia_smi"] = None
    info["note"] = ("Kaggle exposes remaining GPU quota only in the notebook UI (Settings > Accelerator / profile); it cannot "
                    "be read programmatically here. Use runtime_log.jsonl for time actually used.")
    return info


def _synthetic(n: int, planes: int, slices: int, size: int, seed: int, start: int) -> InMemoryWeakDataset:
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, planes, slices, 1, size, size, generator=g)
    mask = torch.ones(n, planes, slices, dtype=torch.bool)
    mask[::4, :, -1] = False
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2, (n, len(TARGET_COLUMNS))).astype(np.float32)
    masks = np.zeros_like(values)
    masks[:, :3] = 1
    values[masks == 0] = np.nan
    return InMemoryWeakDataset([f"synthetic.{start + i}" for i in range(n)], images, mask, values, masks)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_synthetic_smoke(work_dir: Path, *, arms: Sequence[str] = ARMS, device: str = "auto", amp: bool = True,
                        studies: int = 24, image_size: int = 128, slices: int = 9, batch_size: int = 8) -> dict:
    work_dir = Path(work_dir)
    report = {"environment": environment_report(), "arms": {}, "checks_passed": True}
    for arm in arms:
        out, durable = work_dir / f"arm{arm}", work_dir / f"durable{arm}"
        train, val = _synthetic(studies, 3, slices, image_size, 0, 0), _synthetic(max(studies // 3, 4), 3, slices, image_size, 1, 1000)
        config = {**DEFAULT_CONFIG, "arm": arm, "epochs": 2, "batch_size": batch_size, "num_workers": 0, "device": device,
                  "image_size": image_size, "slices_per_series": slices, "amp": amp, "seed": 0}
        steps_per_epoch = -(-len(train) // batch_size)
        checks, entry = {}, {}
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        started = time.perf_counter()

        def interrupt(epoch, step):
            if epoch == 1 and step >= steps_per_epoch + 1:  # inside epoch 2, after the epoch-1 checkpoint is written
                raise _Interrupt

        try:
            train_weak_label(train, val, config, out, fingerprint="smoke", train_ids=train.ids, val_ids=val.ids,
                             batch_hook=interrupt, export_dir=durable)
            checks["interrupt_triggered"] = False
        except _Interrupt:
            checks["interrupt_triggered"] = True
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        checks["checkpoint_after_interrupt_is_epoch_1"] = saved["epoch"] == 1
        checks["full_state_present"] = all(saved.get(k) is not None for k in ("model", "optimizer", "scheduler", "torch_rng", "loader_rng"))
        result = train_weak_label(train, val, config, out, fingerprint="smoke", train_ids=train.ids, val_ids=val.ids,
                                  export_dir=durable)
        final = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        checks["resumed_and_completed"] = result["resumed_from_epoch"] == 1 and result["final_epoch"] == 2
        checks["losses_finite"] = all(np.isfinite([h["train_loss"], h["val_loss"]]).all() for h in result["history"])
        checks["scaler_state_restored"] = (not result["device"].startswith("cuda") or not amp) or bool(final["scaler"])
        checks["durable_export_matches"] = _sha(durable / "last.pt") == _sha(out / "last.pt")
        runtime = summarize_runtime(out)
        seconds = time.perf_counter() - started
        entry.update({
            "params_million": round(sum(p.numel() for p in build_model(arm, config).parameters()) / 1e6, 2),
            "device": result["device"], "amp": amp, "total_seconds": round(seconds, 2),
            "train_studies_per_second": round(len(train) * runtime["epochs"] / max(runtime["epoch_seconds"], 1e-9), 2),
            "peak_cuda_gb": runtime["peak_cuda_gb"] or None, "sessions": runtime["sessions"],
            "gpu_epoch_seconds": round(runtime["gpu_epoch_seconds"], 2), "checks": checks})
        entry["passed"] = all(checks.values())
        report["arms"][arm] = entry
        report["checks_passed"] &= entry["passed"]
    return report
