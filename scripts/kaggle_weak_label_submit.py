"""Offline submission inference: new-model probabilities for the chosen targets, baseline for the rest.

Separate from training. Reads checkpoints from Kaggle input paths only; makes no network calls and submits nothing
by itself -- it only writes submission.csv.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_rsna_runner import INPUT_ROOT, WORKING_ROOT, discover_data, find_dicom_root
from mri_core.rsna_knee_dataset import load_rsna_metadata
from mri_core.weak_label_submit import run_submission

TIER1 = "ACL,Medial Meniscus,Lateral Meniscus"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--dicom-root", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True, help="preserved baseline checkpoint(s)")
    parser.add_argument("--new-checkpoint", type=Path, default=None)
    parser.add_argument("--new-targets", default=TIER1)
    parser.add_argument("--blend-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=WORKING_ROOT / "submission.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.data_dir:
        meta_dir, dicom_root = args.data_dir, args.dicom_root or find_dicom_root(args.data_dir)
    else:
        meta_dir, dicom_root = discover_data(args.input_root)
    import torch
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    result = run_submission(load_rsna_metadata(meta_dir, dicom_root), args.baseline, args.new_checkpoint,
                            [t.strip() for t in args.new_targets.split(",") if t.strip()], args.output,
                            device=device, weight=args.blend_weight, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
