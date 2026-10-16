"""Freeze the Phase 17A experiment definition BEFORE any run:
  configs/phase17_candidate_v1.json  rules, counts and fingerprints of the candidate (C1) mask - no study IDs
  configs/phase17_registry.json      seeds, registered validation-partition hashes, arms, recipe, evaluation policy
  <out>/candidate_mask_v1.json       UID-level override file (git-ignored; ships in the private Kaggle inputs dataset)
  <out>/PROVENANCE.json              hashes of every file in the inputs package
Original manifests are only read. The candidate mask is recomputed from the manifests here and must reproduce the audited fingerprint.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mri_core import weak_label_mask_override as mo  # noqa: E402
from mri_core import weak_label_quality as wq  # noqa: E402
from mri_core.rsna_knee_dataset import TARGET_COLUMNS  # noqa: E402
from mri_core.weak_label_data import hash_split  # noqa: E402

TIER1 = ("ACL", "Medial Meniscus", "Lateral Meniscus")
AUDITED_FINGERPRINT = "e3cc0bd9f0806fc0fad4e3bd589cabce00a40937bdc9d92efed0e50e5912b4b9"
SEEDS = [201, 202, 203, 204, 205]  # never used before (Phases 13-16X used 0, 1, 101-105)
EARLIER_SEEDS = [0, 1, 101, 102, 103, 104, 105]


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, default=Path("outputs/weak-labels"))
    ap.add_argument("--header-features", type=Path, required=True)
    ap.add_argument("--phase16-registry", type=Path, default=Path("configs/phase16_registry.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/kaggle_smoke/p17a_inputs"))
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    args = ap.parse_args(argv)
    profiles = {p: wq.load_manifest_rows(args.manifest_dir / f"weak_manifest_tier1_v{p}.jsonl") for p in wq.PROFILES}
    base = profiles["0.1.0"]
    selection = json.loads(args.selection.read_text())
    ids = [u for u in base if u in set(selection)]
    manifest_sha = sha_file(args.manifest_dir / "weak_manifest_tier1_v0.1.0.jsonl")
    res = wq.candidate_labels(profiles, ids, tier_targets=TIER1)
    fp = wq.candidate_fingerprint(ids, res)
    if fp != AUDITED_FINGERPRINT:
        raise SystemExit(f"candidate fingerprint {fp} differs from the audited {AUDITED_FINGERPRINT}")
    override = mo.build_override(res["candidate"], res["values"], ids, manifest_sha256=manifest_sha, selection_sha256=sha_ids(selection), targets=TIER1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidate_mask_v1.json").write_text(json.dumps(override))
    shutil.copyfile(args.selection, args.out / "selected_studies.json")
    shutil.copyfile(args.header_features, args.out / "header_features.csv")
    counts = {t: {"studies": len(override["targets"][t]["studies"]), "positives": int(sum(override["targets"][t]["values"]))} for t in TIER1}
    p16 = json.loads(args.phase16_registry.read_text())
    candidate_cfg = {
        "schema": "phase17-candidate-definition/1", "name": "phase17_candidate_v1", "frozen": True,
        "source_manifest_sha256": manifest_sha, "profile_manifest_sha256": {p: sha_file(args.manifest_dir / f"weak_manifest_tier1_v{p}.jsonl") for p in wq.PROFILES},
        "selection": {"studies": len(ids), "sha256": sha_ids(selection)}, "candidate_fingerprint_sha256": fp, "override_file": "candidate_mask_v1.json",
        "override_file_sha256": sha_file(args.out / "candidate_mask_v1.json"), "override_fingerprint_sha256": override["fingerprint"], "counts_per_target": counts,
        "rules": ["decided (positive/negative) in the original manifest and Tier-1 trainable",
                  "confidence high: positive without modifier, or negative from negated_lesion / normal_structure / normal_cartilage",
                  "evidence recorded and not shared by >= 4 targets of the study",
                  "same decided state in profiles 0.1.0, 0.2.0, 0.2.2 (studies dropped by a later profile count as unstable)"],
        "guarantees": ["original manifest and label values are never modified", "the mask can only remove trainable entries",
                       "missing, uncertain and masked labels are never converted into negatives", "audit code: scripts/audit_weak_label_quality.py, mri_core/weak_label_quality.py"],
        "not_used": ["Phase 16 hold-out pool", "gold studies", "competition test labels", "model predictions or performance", "leaderboard"]}
    (args.configs / "phase17_candidate_v1.json").write_text(json.dumps(candidate_cfg, indent=2))

    order = [u for u in base if u in set(selection)]
    earlier = {s: {order[i] for i in hash_split(order, 0.2, s)[1]} for s in EARLIER_SEEDS}
    partitions = {}
    for s in SEEDS:
        tr, va = hash_split(order, 0.2, s)
        val = {order[i] for i in va}
        partitions[str(s)] = {"train_studies": len(tr), "validation_studies": len(va), "validation_sha256": sha_ids(val),
                              "max_overlap_with_earlier_validation_sets": max(len(val & e) for e in earlier.values())}
    registry = {
        "schema": "phase17a-registry/1", "frozen": True, "primary_target": "Medial Meniscus", "secondary_targets": ["ACL", "Lateral Meniscus"],
        "candidate_definition": "configs/phase17_candidate_v1.json", "candidate_fingerprint_sha256": fp, "override_fingerprint_sha256": override["fingerprint"], "manifest_sha256": manifest_sha,
        "training_pool": {"studies": len(ids), "sha256": sha_ids(selection)}, "seeds": SEEDS, "shuffle_seed_offset": 2000, "untrained_seeds": list(range(9001, 9011)),
        "model": {"arm": "C", "image_size": 128, "slices_per_series": 9, "amp": True, "batch_size": 8, "learning_rate": 1e-4, "epochs": 12, "grad_clip": 1.0,
                  "weight_decay": 1e-4, "warmup_epochs": 1, "val_fraction": 0.2, "early_stopping_patience": 0,
                  "note": "the Phase 16 recipe, carried over unchanged; no new tuning on any validation or hold-out set"},
        "arms": {"B0": "C0 trained with the original weak-label mask", "C1": "C0 trained with the candidate mask (same seed, split, initialisation as B0)",
                 "M0": "metadata-only logistic regression and boosted tree fitted on the same training studies (header features)",
                 "S": "C0 trained on the candidate mask with labels shuffled across training studies (paired with C1)", "N": "untrained C0, seeds 9001-9010, forward pass only"},
        "partitions": partitions,
        "protected_sets": {"phase16_holdout_pool": p16["holdout_pool"], "gold_studies": 58, "kaggle_test": "excluded by the manifest and re-checked per run"},
        "evaluation_policy": {
            "weak_label_validation": "descriptive only: the registered validation partitions scored with the manifest's own labels; never a basis for a claim",
            "phase16_holdout": "already inspected in Phases 16 and 16X; not independent confirmation; excluded from training/validation; may be reported descriptively only",
            "independent_evidence": "requires a human-adjudicated reference disjoint from the training pool, all validation partitions, the Phase 16 hold-out, the gold and test sets; "
                                    "if it does not exist, the result is reported with that limitation and no claim is made",
            "claim_allowed_without_adjudicated_reference": False},
        "forbidden": ["gold studies", "leaderboard feedback", "hidden test images", "choice of arm, seed or epoch after seeing results"]}
    (args.configs / "phase17_registry.json").write_text(json.dumps(registry, indent=2))
    prov = {"kind": "phase17a_inputs", "files": {n: {"sha256": sha_file(args.out / n), "bytes": (args.out / n).stat().st_size}
                                                 for n in ("candidate_mask_v1.json", "selected_studies.json", "header_features.csv")},
            "candidate_fingerprint_sha256": fp, "contains_images": False, "contains_reports": False, "contains_gold_labels": False,
            "note": "UID-level candidate labels derived from reports; keep private"}
    (args.out / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({"fingerprint": fp, "counts": counts, "partitions": {k: v["validation_sha256"][:12] for k, v in partitions.items()}}, indent=1))


if __name__ == "__main__":
    main()
