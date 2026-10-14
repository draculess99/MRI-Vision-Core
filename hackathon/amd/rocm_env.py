"""AMD/ROCm environment check: reports what this machine actually has, nothing assumed.

`amd_gpu_available` is True only when ALL of these hold: the installed PyTorch is a ROCm
(HIP) build, it reports an accelerator, and at least one device is visible. A CUDA (NVIDIA)
build or a CPU-only build is reported as not-AMD, with the reason spelled out.

Usage:  python -m hackathon.amd.rocm_env [--require-amd] [--output env.json]
Exit code 0 = AMD GPU usable, 2 = not usable (always 2 with --require-amd; otherwise 0
so the check can be used purely for reporting).
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import torch

SMI_TOOLS = ("rocm-smi", "amd-smi")


def _run_smi(tool: str, args: list[str], timeout: float = 20.0) -> Optional[str]:
    path = shutil.which(tool)
    if path is None:
        return None
    try:
        proc = subprocess.run([path, *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<{tool} failed: {exc}>"
    return (proc.stdout or proc.stderr).strip()


def smi_snapshot() -> dict:
    """Raw rocm-smi/amd-smi text (verbatim, unparsed) if the tools exist, else None."""
    return {
        "rocm-smi": _run_smi("rocm-smi", ["--showproductname", "--showuse", "--showmeminfo", "vram"]),
        "amd-smi": _run_smi("amd-smi", ["metric"]),
    }


def check_environment() -> dict:
    hip = getattr(torch.version, "hip", None)
    cuda = torch.version.cuda
    accelerator_available = bool(torch.cuda.is_available())
    device_count = torch.cuda.device_count() if accelerator_available else 0

    devices = []
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "gcn_arch": getattr(props, "gcnArchName", None),
        })

    reasons = []
    if hip is None:
        reasons.append(
            "PyTorch is not a ROCm build (torch.version.hip is None)"
            + (f"; it is a CUDA {cuda} build" if cuda else "; it is a CPU-only build")
        )
    if not accelerator_available:
        reasons.append("torch.cuda.is_available() is False (no visible AMD accelerator or driver)")
    elif device_count == 0:
        reasons.append("accelerator reported available but device_count is 0")
    amd_gpu_available = hip is not None and accelerator_available and device_count > 0

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_hip_version": hip,
        "torch_cuda_version": cuda,
        "is_rocm_build": hip is not None,
        "accelerator_available": accelerator_available,
        "device_count": device_count,
        "devices": devices,
        "smi_tools_on_path": {tool: shutil.which(tool) for tool in SMI_TOOLS},
        "amd_gpu_available": amd_gpu_available,
        "unavailable_reasons": [] if amd_gpu_available else reasons,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-amd", action="store_true", help="exit non-zero unless an AMD GPU is usable")
    parser.add_argument("--smi", action="store_true", help="also capture raw rocm-smi/amd-smi output")
    parser.add_argument("--output", help="write the JSON report to this path")
    args = parser.parse_args(argv)

    report = check_environment()
    if args.smi:
        report["smi_snapshot"] = smi_snapshot()
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 2 if (args.require_amd and not report["amd_gpu_available"]) else 0


if __name__ == "__main__":
    sys.exit(main())
