"""Kaggle-native weak-label experiment: preflight / train / evaluate-gold / export (offline, GPU optional).

Training and submission are separate workflows (see kaggle_weak_label_submit.py). Nothing here uses the network:
the competition data, the weak-label manifest, the code, and any pretrained weights are all Kaggle inputs.
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_rsna_runner import INPUT_ROOT, WORKING_ROOT, device_info, discover_data, find_dicom_root
from mri_core.rsna_checkpoint import studies_fingerprint
from mri_core.rsna_knee_dataset import TARGET_COLUMNS, load_rsna_metadata
from mri_core.weak_label_data import (WeakLabelDicomDataset, assert_no_leakage, hash_split, load_weak_manifest,
                                      preflight_series, select_smoke_subset)
from mri_core.weak_label_train import (DEFAULT_CONFIG, IDENTITY_KEYS, GoldAlreadyScoredError, evaluate_gold, sha256_file,
                                       summarize_runtime, train_weak_label)

EXPORT_NAMES = ("last.pt", "last_best.pt", "config.json", "runtime_log.jsonl", "gold_eval.json", "preflight.json",
                "invalid_studies.json")


def _paths(args):
    if args.data_dir is not None:
        return args.data_dir, args.dicom_root or find_dicom_root(args.data_dir)
    return discover_data(args.input_root)


def _config(args) -> dict:
    config = {**DEFAULT_CONFIG, **(json.loads(args.config.read_text()) if args.config else {})}
    if args.arm:
        config["arm"] = args.arm
    return config


def _load(args):
    meta_dir, dicom_root = _paths(args)
    metadata = load_rsna_metadata(meta_dir, dicom_root)
    manifest = load_weak_manifest(args.manifest)
    gold = metadata.complete_label_train
    test_ids = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
    assert_no_leakage(manifest.ids, [], gold_ids=gold["StudyInstanceUID"], test_ids=test_ids)
    return metadata, manifest, gold


def cmd_preflight(args) -> int:
    metadata, manifest, gold = _load(args)
    config = _config(args)
    problems = preflight_series(metadata, manifest.ids, config["planes"])
    counts = {t: int(m) for t, m in zip(TARGET_COLUMNS, manifest.masks.sum(axis=0)) if m > 0}
    report = {"manifest_sha256": manifest.sha256, "studies": len(manifest.ids), "active_targets": list(counts),
              "trainable_counts": counts, "gold_studies": int(len(gold)), "gold_overlap": 0,
              "studies_with_series_problems": len(problems),
              "problem_examples": [{"study": "..." + u[-6:], "reasons": r} for u, r in list(problems.items())[:5]],
              "device": device_info()}
    print(json.dumps(report, indent=2))
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        (args.work_dir / "preflight.json").write_text(json.dumps(report, indent=2))
    return 0 if not problems else 1


def _select(args, metadata, manifest, config):
    """Studies that will actually be used: series present, then an optional deterministic subset. Pure and repeatable."""
    args.work_dir.mkdir(parents=True, exist_ok=True)
    bad = preflight_series(metadata, manifest.ids, config["planes"])
    if bad:
        (args.work_dir / "preflight_excluded.json").write_text(json.dumps(bad, indent=2))
        print(f"excluding {len(bad)} studies with missing series (IDs and reasons in preflight_excluded.json)")
    keep = [i for i, uid in enumerate(manifest.ids) if uid not in bad]
    selection = None
    if getattr(args, "smoke_subset", 0):  # stratified: both classes of each Tier 1 target where feasible
        selection = select_smoke_subset([manifest.ids[i] for i in keep], manifest.values[keep], manifest.masks[keep],
                                        args.smoke_subset, seed=int(config["seed"]),
                                        min_per_class=getattr(args, "min_per_class", 8))
        keep = [keep[i] for i in selection["indices"]]
        (args.work_dir / "smoke_selection.json").write_text(json.dumps(
            {"studies": [manifest.ids[i] for i in keep], **{k: v for k, v in selection.items() if k != "indices"}}, indent=2))
        print(json.dumps({"smoke_subset": len(keep), "candidates": selection["candidates"],
                          "class_balance": selection["class_balance"], "unmet_quota": selection["unmet_quota"],
                          "excluded_missing_series": len(bad)}))
    elif getattr(args, "max_studies", 0):  # deterministic subset by UID hash
        keep = sorted(sorted(keep, key=lambda i: hashlib.sha256(f"subset:{manifest.ids[i]}".encode()).hexdigest())[:args.max_studies])
    ids = [manifest.ids[i] for i in keep]
    return ids, manifest.values[keep], manifest.masks[keep], selection, bad


def cmd_train(args) -> int:
    metadata, manifest, _ = _load(args)
    config = _config(args)
    if args.epochs_override:
        config["epochs"] = args.epochs_override
    ids, values, masks, _, _ = _select(args, metadata, manifest, config)
    train_idx, val_idx = hash_split(ids, float(config["val_fraction"]), int(config["seed"]))
    fingerprint = hashlib.sha256(f"{manifest.fingerprint()}:{studies_fingerprint(ids)}".encode()).hexdigest()

    def make(idx):
        return WeakLabelDicomDataset(metadata, [ids[i] for i in idx], values[idx], masks[idx], config, on_error="skip",
                                     cache_dir=config.get("cache_dir"))

    (args.work_dir / "config.json").write_text(json.dumps(config, indent=2))
    result = train_weak_label(make(train_idx), make(val_idx), config, args.work_dir, fingerprint=fingerprint,
                              train_ids=[ids[i] for i in train_idx], val_ids=[ids[i] for i in val_idx], resume=True,
                              max_seconds=args.max_seconds, export_dir=args.export_dir,
                              stop_after_epochs=args.stop_after_epochs or None)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2, default=float))
    return 0


def cmd_verify_checkpoint(args) -> int:
    """Read-only restore check: is this checkpoint compatible with the current manifest, selection and split?"""
    import torch
    metadata, manifest, _ = _load(args)
    config = _config(args)
    ids, _, _, _, _ = _select(args, metadata, manifest, config)
    train_idx, val_idx = hash_split(ids, float(config["val_fraction"]), int(config["seed"]))
    fingerprint = hashlib.sha256(f"{manifest.fingerprint()}:{studies_fingerprint(ids)}".encode()).hexdigest()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    report = {
        "checkpoint_sha256": sha256_file(args.checkpoint), "epoch": int(state["epoch"]), "global_step": int(state["global_step"]),
        "arm": state["config"].get("arm"),
        "state_present": {k: state.get(k) is not None for k in ("model", "optimizer", "scheduler", "scaler", "torch_rng",
                                                                  "python_rng", "numpy_rng", "cuda_rng", "loader_rng")},
        "fingerprint_matches": state["dataset_fingerprint"] == fingerprint,
        "train_split_matches": sorted(state["train_study_ids"]) == sorted(ids[i] for i in train_idx),
        "validation_split_matches": sorted(state["valid_study_ids"]) == sorted(ids[i] for i in val_idx),
        "target_columns_match": list(state["target_columns"]) == list(TARGET_COLUMNS),
        "scheduler_last_epoch": (state["scheduler"] or {}).get("last_epoch"),
        "scaler_scale": (state["scaler"] or {}).get("scale") if state.get("scaler") else None,
        "config_matches": all(state["config"].get(k) == config.get(k) for k in IDENTITY_KEYS)}
    report["compatible"] = all(report[k] for k in ("fingerprint_matches", "train_split_matches", "validation_split_matches",
                                                    "target_columns_match", "config_matches"))
    print(json.dumps(report, indent=2))
    return 0 if report["compatible"] else 1


def cmd_real_preflight(args) -> int:
    """Read-only: decode the selected studies once and validate tensors, masks, latency, cache size, GPU and disk."""
    import shutil
    import time
    import numpy as np
    import torch
    metadata, manifest, gold = _load(args)
    config = _config(args)
    ids, values, masks, selection, excluded = _select(args, metadata, manifest, config)
    train_idx, val_idx = hash_split(ids, float(config["val_fraction"]), int(config["seed"]))
    test_ids = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
    assert_no_leakage([ids[i] for i in train_idx], [ids[i] for i in val_idx], gold_ids=gold["StudyInstanceUID"], test_ids=test_ids)
    dataset = WeakLabelDicomDataset(metadata, ids, values, masks, config, on_error="raise", cache_dir=config.get("cache_dir"))
    latencies, problems, slices = [], [], []
    expected = (len(config["planes"]), int(config["slices_per_series"]), 1, int(config["image_size"]), int(config["image_size"]))
    for i in range(len(ids)):
        started = time.perf_counter()
        try:
            sample = dataset[i]
        except Exception as exc:  # report and keep going so every problem is visible
            problems.append({"study": "..." + ids[i][-6:], "error": type(exc).__name__ + ": " + str(exc)[:160]})
            continue
        latencies.append(time.perf_counter() - started)
        images, mask = sample["images"], sample["mask"]
        error = None
        if tuple(images.shape) != expected:
            error = f"shape {tuple(images.shape)} != {expected}"
        elif not bool(torch.isfinite(images).all()):
            error = "non-finite pixel values"
        elif not bool(mask.any(dim=1).all()):
            error = "a plane has no valid slice"
        elif float(images.min()) < -1e-6 or float(images.max()) > 1 + 1e-6:
            error = "pixel range outside [0, 1]"
        if error:
            problems.append({"study": "..." + ids[i][-6:], "error": error})
        slices.append(int(mask.sum()))
    cache_dir = config.get("cache_dir")
    cache_bytes = sum(f.stat().st_size for f in Path(cache_dir).glob("*.pt")) if cache_dir and Path(cache_dir).is_dir() else 0
    report = {
        "manifest_sha256": manifest.sha256, "studies_selected": len(ids), "train": len(train_idx), "validation": len(val_idx),
        "excluded_missing_series": len(excluded), "decoded_ok": len(latencies), "problems": problems, "gold_or_test_overlap": 0,
        "decode_seconds_per_study": {"mean": float(np.mean(latencies)) if latencies else None,
                                     "p95": float(np.percentile(latencies, 95)) if latencies else None,
                                     "total": float(np.sum(latencies)) if latencies else None},
        "valid_slices_per_study_mean": float(np.mean(slices)) if slices else None, "tensor_shape": list(expected),
        "cache_bytes": cache_bytes, "free_disk_gb": round(shutil.disk_usage(args.work_dir).free / 1e9, 1),
        "class_balance": selection["class_balance"] if selection else None,
        "unmet_quota": selection["unmet_quota"] if selection else None, "device": device_info()}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["gpu_memory_free_gb"], report["gpu_memory_total_gb"] = round(free / 1e9, 2), round(total / 1e9, 2)
    report["passed"] = bool(not problems and len(latencies) == len(ids) and len(ids) > 0)
    (args.work_dir / "real_preflight.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "device"}, indent=2, default=float))
    return 0 if report["passed"] else 1


def cmd_evaluate_gold(args) -> int:
    metadata, _, gold = _load(args)
    try:
        report = evaluate_gold(args.checkpoint, metadata, gold.reset_index(drop=True), ledger_path=args.ledger,
                               force=args.force, rounds=args.rounds)
    except GoldAlreadyScoredError as exc:
        print(f"REFUSED: {exc}")
        return 2
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "gold_eval.json").write_text(json.dumps(report, indent=2, default=float))
    keys = ("arm", "epoch", "macro_auc", "macro_ci95", "headline_macro_auc", "headline_ci95")
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    return 0


def cmd_export(args) -> int:
    """Copy run artifacts to the export dir, re-hash every copy, and write provenance + a SHA-256 manifest + a zip."""
    args.export_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in EXPORT_NAMES:
        source = args.work_dir / name
        if source.is_file():
            target = args.export_dir / name
            target.write_bytes(source.read_bytes())
            digest = sha256_file(source)
            if sha256_file(target) != digest:
                raise OSError(f"export copy of {name} does not match its source")
            files[name] = {"sha256": digest, "bytes": source.stat().st_size}
    if "last.pt" not in files:
        raise FileNotFoundError(f"no last.pt in {args.work_dir}")
    import torch
    state = torch.load(args.export_dir / "last.pt", map_location="cpu", weights_only=True)
    provenance = {"arm": state["config"].get("arm"), "epoch": int(state["epoch"]), "global_step": int(state["global_step"]),
                  "dataset_fingerprint": state["dataset_fingerprint"], "config": state["config"],
                  "n_train_studies": len(state["train_study_ids"]), "n_valid_studies": len(state["valid_study_ids"]),
                  "target_columns": state["target_columns"], "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpus": device_info().get("gpus", []),
                  "note": "weak-label validation only was used for checkpoint selection"}
    (args.export_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    files["provenance.json"] = {"sha256": sha256_file(args.export_dir / "provenance.json"),
                                "bytes": (args.export_dir / "provenance.json").stat().st_size}
    (args.export_dir / "MANIFEST.json").write_text(json.dumps({**files, "runtime": summarize_runtime(args.work_dir)}, indent=2))
    archive = args.export_dir / "weak_label_run_export.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in [*files, "MANIFEST.json"]:
            bundle.write(args.export_dir / name, name)
    print(json.dumps({"archive": str(archive), "files": files}, indent=2))
    return 0


def cmd_smoke(args) -> int:
    from mri_core.weak_label_smoke import run_synthetic_smoke
    report = run_synthetic_smoke(args.work_dir, arms=args.arms, device=args.device, amp=not args.no_amp,
                                 studies=args.studies, image_size=args.image_size, slices=args.slices,
                                 batch_size=args.batch_size)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["checks_passed"] else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "real-preflight", "verify-checkpoint", "train", "evaluate-gold", "export"):
        p = sub.add_parser(name)
        p.add_argument("--input-root", type=Path, default=INPUT_ROOT)
        p.add_argument("--data-dir", type=Path, default=None)
        p.add_argument("--dicom-root", type=Path, default=None)
        p.add_argument("--manifest", type=Path, default=None, help="weak-label manifest .jsonl (a Kaggle input)")
        p.add_argument("--work-dir", type=Path, default=WORKING_ROOT / "outputs" / "weak-label")
        p.add_argument("--config", type=Path, default=None)
        p.add_argument("--arm", choices=("A", "B", "C"), default=None)
    sub.choices["train"].add_argument("--max-seconds", type=float, default=None,
                                      help="stop cleanly after the epoch that would exceed this")
    sub.choices["train"].add_argument("--export-dir", type=Path, default=None,
                                      help="copy checkpoints here after every epoch")
    sub.choices["train"].add_argument("--epochs-override", type=int, default=0)
    sub.choices["train"].add_argument("--stop-after-epochs", type=int, default=0,
                                      help="end this session cleanly after N epochs (the run can be resumed later)")
    sub.choices["verify-checkpoint"].add_argument("--checkpoint", type=Path, required=True)
    for name in ("train", "real-preflight", "verify-checkpoint"):
        sub.choices[name].add_argument("--min-per-class", type=int, default=8,
                                       help="quota of positive and negative studies per active target in the subset")
        sub.choices[name].add_argument("--smoke-subset", type=int, default=0,
                                       help="stratified deterministic subset of this many studies (real-data smoke test)")
        sub.choices[name].add_argument("--max-studies", type=int, default=0,
                                       help="deterministic UID-hash subset of this many studies")
    sub.choices["evaluate-gold"].add_argument("--checkpoint", type=Path, required=True)
    sub.choices["evaluate-gold"].add_argument("--ledger", type=Path, required=True)
    sub.choices["evaluate-gold"].add_argument("--force", action="store_true")
    sub.choices["evaluate-gold"].add_argument("--rounds", type=int, default=1000)
    sub.choices["export"].add_argument("--export-dir", type=Path, default=WORKING_ROOT / "export")
    smoke = sub.add_parser("smoke", help="synthetic GPU/AMP smoke test; needs no data, manifest or network")
    smoke.add_argument("--work-dir", type=Path, default=WORKING_ROOT / "outputs" / "smoke")
    smoke.add_argument("--arms", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--no-amp", action="store_true")
    smoke.add_argument("--studies", type=int, default=24)
    smoke.add_argument("--image-size", type=int, default=128)
    smoke.add_argument("--slices", type=int, default=9)
    smoke.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command not in ("export", "smoke") and args.manifest is None:
        parser.error("--manifest is required")
    handlers = {"preflight": cmd_preflight, "real-preflight": cmd_real_preflight, "verify-checkpoint": cmd_verify_checkpoint, "train": cmd_train, "evaluate-gold": cmd_evaluate_gold, "export": cmd_export,
                "smoke": cmd_smoke}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
