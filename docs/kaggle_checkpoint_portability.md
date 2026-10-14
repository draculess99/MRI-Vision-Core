# Exporting a resumable RSNA checkpoint from Kaggle and resuming on the Dell 3535

This describes how to move a **resumable** checkpoint (as written by
`mri_core.rsna_knee_train_resumable.train_rsna_resumable`, not the older
inference-only format from `rsna_knee_train.train_rsna`) from a Kaggle notebook
back to this machine, verify it wasn't corrupted in transit, and resume training
on CPU.

## 1. On Kaggle: produce the checkpoint

Run training inside the notebook exactly as locally (same `mri_core` source,
uploaded via the existing `mri_core_kaggle*.zip` convention). The resumable
checkpoint (`last.pt`, plus `last_best.pt` on improvement) is written under
whatever output directory you pass as `checkpoint_dir`, e.g.
`/kaggle/working/outputs/rsna-knee-resumable/`.

Record the SHA-256 of the checkpoint file before downloading it:

```python
import hashlib
print(hashlib.sha256(open("/kaggle/working/outputs/rsna-knee-resumable/last.pt", "rb").read()).hexdigest())
```

Download `last.pt` (and `last_best.pt` if present) via the notebook's Output tab,
or `kaggle kernels output <notebook-ref> -p <local-dir>`.

## 2. On the Dell 3535: verify artifact integrity

Before trusting the downloaded file, recompute its hash and compare it against
the one printed on Kaggle:

```powershell
Get-FileHash .\last.pt -Algorithm SHA256
```

If the hashes differ, the file was corrupted or truncated in transit — re-download
rather than attempting to resume from it.

## 3. CPU-safe load

`mri_core.rsna_checkpoint.load_checkpoint` always calls
`torch.load(path, map_location="cpu", weights_only=True)` internally, so a
checkpoint written on a Kaggle GPU session loads safely on this machine's CPU-only
`.venv` with no `map_location` juggling required. `weights_only=True` also means
the file cannot execute arbitrary code during load — only tensors/basic Python
containers are accepted, which matters when importing a file produced on a
different (cloud) machine.

```python
from pathlib import Path
import torch

from mri_core.rsna_knee_model import RSNAKneeCNN
from mri_core.rsna_checkpoint import load_checkpoint, CheckpointIncompatibleError
from mri_core.rsna_knee_dataset import TARGET_COLUMNS

model = RSNAKneeCNN(num_planes=3, dropout=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

try:
    state = load_checkpoint(
        Path("last.pt"), model=model, optimizer=optimizer,
        expected_target_columns=TARGET_COLUMNS,
        expected_dataset_fingerprint=None,  # set once you know the local dataset's fingerprint
        device="cpu",
    )
except CheckpointIncompatibleError as exc:
    raise SystemExit(f"Refusing to resume: {exc}")
```

## 4. Resume locally

Call `train_rsna_resumable(metadata, config, checkpoint_dir, resume=True)` with
the **same** `checkpoint_dir` the downloaded `last.pt` was placed in. The function
re-validates `target_columns` and the local `dataset_fingerprint` (a hash of every
labeled study UID + its 12 labels) before restoring anything — if the local
`data/rsna-knee/train.csv` has drifted from what Kaggle trained against, it raises
`CheckpointIncompatibleError` instead of silently continuing on mismatched data.

## 5. What is *not* portable

- CUDA RNG state restores only if a CUDA device is present and matches in count;
  on this CPU-only machine, `cuda_rng` is saved as `[]` and simply skipped on
  resume — only Python/NumPy/PyTorch CPU RNG state round-trips exactly.
- Exact bitwise numeric reproduction across a GPU (Kaggle) → CPU (Dell 3535)
  resume is **not guaranteed** even with identical seeds — this mirrors the
  existing `determinism_note` already used by `mri_core/cnn_train.py`'s
  provenance records. Only the *data, split, and configuration* are strictly
  verified; floating-point training dynamics may differ slightly across
  hardware.

No dataset or checkpoint is uploaded to any public service by this process —
transfer is a manual download from Kaggle's own Output tab to local disk.
