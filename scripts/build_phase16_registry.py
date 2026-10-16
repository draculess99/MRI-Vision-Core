"""Write configs/phase16_registry.json: the Phase 16 design, seeds, partition hashes and decision rules, fixed BEFORE any Phase 16 run.

Inputs are the Phase 14/15 selection of 1,200 studies (selected_studies.json from the Phase 15 kernel output) and the v0.1.0 manifest.
Partition hashes are SHA-256 of the sorted validation study IDs of each seed; the Kaggle driver refuses to continue if a run's
validation set differs from the registered one.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_core.weak_label_data import hash_split, load_weak_manifest  # noqa: E402

SEEDS = [101, 102, 103, 104, 105]  # never used before; Phase 13-15 used seeds 0 and 1
PREVIOUS_SEEDS = [0, 1]


def digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("configs/phase16_registry.json"))
    args = ap.parse_args(argv)
    manifest = load_weak_manifest(args.manifest)
    selected = set(json.loads(args.selected.read_text()))
    order = [u for u in manifest.ids if u in selected]
    pool = [u for u in manifest.ids if u not in selected]

    def val_ids(seed):
        return {order[i] for i in hash_split(order, 0.2, seed)[1]}

    earlier = {s: val_ids(s) for s in PREVIOUS_SEEDS}
    registry = {
        "registry_version": 1,
        "question": "Does C0 (128x9, lr 1e-4) learn weak-label signal beyond what a shuffled-label control shows, and does gradient clipping at 5.0 differ from 1.0?",
        "manifest_sha256": manifest.sha256, "training_pool": {"studies": len(order), "sha256": digest(order)},
        "model": {"arm": "C", "image_size": 128, "slices_per_series": 9, "amp": True, "batch_size": 8, "weight_decay": 1e-4,
                  "warmup_epochs": 1, "val_fraction": 0.2, "early_stopping_patience": 0, "labels": "rules 0.1.0"},
        "learning_rate": 1e-4, "epochs": 12, "grad_clip_arms": [1.0, 5.0], "seeds": SEEDS, "shuffle_seed_offset": 1000,
        "selection": {"studies": 1200, "min_per_class": 60, "selection_seed": 0},
        "partitions": {str(s): {"validation_studies": len(val_ids(s)), "validation_sha256": digest(val_ids(s)),
                                "overlap_with_seed0_validation": len(val_ids(s) & earlier[0]),
                                "overlap_with_seed1_validation": len(val_ids(s) & earlier[1])} for s in SEEDS},
        "holdout_pool": {"description": "v0.1.0 manifest studies NOT in the 1,200 selection; never trained or validated on in Phases 14-15 "
                                          "(they appeared in the larger Phase 9-13 runs, so this is not independent data)",
                         "studies_before_series_and_readability_filters": len(pool), "sha256": digest(pool)},
        "per_run": {"epochs_fixed": 12, "report_epoch": "final (fixed in advance; no selection on validation or hold-out)",
                    "secondary_report": "best-validation-loss epoch (descriptive only)"},
        "criteria": {
            "C1_validation": "real final macro AUC minus shuffled twin >= 0.10",
            "C2_holdout": "real hold-out macro AUC minus shuffled twin hold-out macro AUC >= 0.05 (final-epoch checkpoint)",
            "C3_loss": "real final validation loss < constant-prediction floor (p=0.5) of that split",
            "C4_control_sanity": "mean over seeds of shuffled validation macro AUC and of shuffled hold-out macro AUC both within [0.45, 0.55]; otherwise verdict is invalid_control and the run is investigated, not interpreted",
            "supported": "C1, C2 and C3 each hold in 5 of 5 seeds (arm evaluated separately)",
            "promising": "C1, C2 and C3 each hold in at least 4 of 5 seeds",
            "not_supported": "otherwise",
            "incomplete": "fewer than 5 complete seed blocks: report descriptively, no verdict",
            "clip_comparison": "different only if the sign of (clip 5.0 minus clip 1.0 real final validation AUC) is the same in 5 of 5 seeds and the mean magnitude >= 0.03"},
        "stopping_rules": {"non_finite_loss": "abort the run and stop the experiment", "two_consecutive_run_failures": "stop",
                           "partition_mismatch": "stop", "leakage_check_failure": "stop",
                           "time": "no seed block is started if it cannot finish before 6900 s of kernel time; kernel limit 7200 s",
                           "control_alarm": "if a shuffled run reaches validation AND hold-out macro AUC >= 0.60, stop and investigate leakage"},
        "forbidden": ["gold studies", "public leaderboard feedback", "hidden test images", "any choice of epoch, seed or arm made after seeing results"],
        "validity_note": "Alternative splits of the same 1,200 studies are not an independent external validation set; the hold-out pool is "
                         "disjoint from this experiment's training/validation studies but comes from the same source, labeler and earlier phases."}
    args.output.write_text(json.dumps(registry, indent=2))
    print(json.dumps({k: registry[k] for k in ("partitions", "holdout_pool")}, indent=2))


if __name__ == "__main__":
    main()
