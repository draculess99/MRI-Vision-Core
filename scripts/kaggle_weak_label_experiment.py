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
    excluded_file = getattr(args, "exclude_studies", None)
    excluded_extra = set(json.loads(Path(excluded_file).read_text())) if excluded_file else set()
    keep = [i for i, uid in enumerate(manifest.ids) if uid not in bad and uid not in excluded_extra]
    include_file = getattr(args, "include_studies", None)
    if include_file:  # restrict to an earlier selection (e.g. the studies whose cache variants were built)
        allowed = set(json.loads(Path(include_file).read_text()))
        keep = [i for i in keep if manifest.ids[i] in allowed]
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
    from mri_core.weak_label_diagnostics import constant_baselines, leakage_report, shuffle_labels
    metadata, manifest, gold = _load(args)
    config = _config(args)
    if args.epochs_override:
        config["epochs"] = args.epochs_override
    ids, values, masks, _, _ = _select(args, metadata, manifest, config)
    train_idx, val_idx = hash_split(ids, float(config["val_fraction"]), int(config["seed"]))
    fingerprint_text = f"{manifest.fingerprint()}:{studies_fingerprint(ids)}"
    train_values, train_masks = values[train_idx], masks[train_idx]
    if args.val_equals_train:  # diagnostic only (overfit test): the validation metrics then describe fit to the training set
        val_idx = train_idx
        fingerprint_text += ":val_equals_train"
    if args.shuffle_train_labels is not None:  # negative control: labels are permuted across TRAINING studies only
        train_values, train_masks, _ = shuffle_labels(train_values, train_masks, args.shuffle_train_labels)
        fingerprint_text += f":shuffled_train_labels:{args.shuffle_train_labels}"
    fingerprint = hashlib.sha256(fingerprint_text.encode()).hexdigest()
    test_ids = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
    leak = leakage_report([ids[i] for i in train_idx], [ids[i] for i in val_idx], gold["StudyInstanceUID"], test_ids,
                          allow_val_equals_train=args.val_equals_train)
    if not leak["clean"]:
        raise SystemExit(f"leakage check failed: {leak}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "leakage.json").write_text(json.dumps(leak, indent=2))
    (args.work_dir / "baselines.json").write_text(json.dumps(constant_baselines(
        values[train_idx], masks[train_idx], values[val_idx], masks[val_idx], float(config["max_pos_weight"])), indent=2))

    def make(idx, labels=None):
        v, m = (values[idx], masks[idx]) if labels is None else labels
        return WeakLabelDicomDataset(metadata, [ids[i] for i in idx], v, m, config, on_error="skip",
                                     cache_dir=config.get("cache_dir"))

    (args.work_dir / "config.json").write_text(json.dumps(config, indent=2))

    def split_counts(idx):
        counts = {}
        for j, t in enumerate(TARGET_COLUMNS):
            on = masks[idx, j] == 1
            if on.any():
                counts[t] = {"positive": int(((values[idx, j] == 1) & on).sum()), "negative": int(((values[idx, j] == 0) & on).sum())}
        return counts

    (args.work_dir / "split_summary.json").write_text(json.dumps(
        {"included_studies": len(ids), "train_studies": len(train_idx), "validation_studies": len(val_idx),
         "train_targets": split_counts(train_idx), "validation_targets": split_counts(val_idx),
         "shuffle_train_labels": args.shuffle_train_labels, "val_equals_train": bool(args.val_equals_train),
         "manifest_sha256": manifest.sha256, "dataset_fingerprint": fingerprint}, indent=2))
    result = train_weak_label(make(train_idx, (train_values, train_masks)), make(val_idx), config, args.work_dir, fingerprint=fingerprint,
                              train_ids=[ids[i] for i in train_idx], val_ids=[ids[i] for i in val_idx], resume=True,
                              max_seconds=args.max_seconds, export_dir=args.export_dir,
                              stop_after_epochs=args.stop_after_epochs or None)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2, default=float))
    return 0


def cmd_cache_warm_multi(args) -> int:
    """Decode each selected study ONCE and write every requested (size x slices) cache variant; report time and size per variant."""
    import time
    from torch.utils.data import DataLoader
    from mri_core.weak_label_multicache import MultiVariantWarmDataset, parse_variants
    metadata, manifest, _ = _load(args)
    config = _config(args)
    variants = parse_variants(args.variants)
    ids, _, _, selection, _ = _select(args, metadata, manifest, config)
    cache_dir = Path(config["cache_dir"])
    dataset = MultiVariantWarmDataset(metadata, ids, variants, config, cache_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=int(config["num_workers"]))
    started, bad, decoded, reused = time.perf_counter(), [], 0, 0
    for batch in loader:
        bad.extend(u for u, ok in zip(batch["study_uid"], batch["ok"].tolist()) if not ok)
        decoded += int(batch["decoded"].sum())
        reused += int((~batch["decoded"]).sum())
    seconds = time.perf_counter() - started
    per_variant = {}
    bad_set = set(bad)
    for (size, slices), ds in dataset.variant_datasets.items():
        paths = [ds._cache_path(u) for u in ids if u not in bad_set]
        per_variant[f"{size}x{slices}"] = {"entries": sum(p.is_file() for p in paths), "bytes": sum(p.stat().st_size for p in paths if p.is_file())}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "selected_studies.json").write_text(json.dumps(sorted(set(ids) - bad_set)))
    (args.work_dir / "excluded_unreadable.json").write_text(json.dumps(sorted(bad)))
    report = {"studies": len(ids), "unreadable": len(bad), "decoded": decoded, "already_cached": reused, "seconds": round(seconds, 1),
              "seconds_per_study": round(seconds / max(len(ids), 1), 3), "workers": int(config["num_workers"]), "variants": per_variant,
              "class_balance": selection["class_balance"] if selection else None}
    (args.work_dir / "cache_warm_multi.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def cmd_cache_warm(args) -> int:
    """Decode every selected study once with worker processes: fills the cache, times it, and lists unreadable studies."""
    import time
    import torch
    from torch.utils.data import DataLoader
    metadata, manifest, gold = _load(args)
    config = _config(args)
    ids, values, masks, _, excluded = _select(args, metadata, manifest, config)
    cache_dir = Path(config["cache_dir"])
    before = len(list(cache_dir.glob("*.pt"))) if cache_dir.is_dir() else 0
    dataset = WeakLabelDicomDataset(metadata, ids, values, masks, config, on_error="skip", cache_dir=cache_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=int(config["num_workers"]))
    bad, started, done = [], time.perf_counter(), 0
    for batch in loader:
        done += len(batch["valid"])
        bad.extend(u for u, v in zip(batch["study_uid"], batch["valid"].tolist()) if not v)
    seconds = time.perf_counter() - started
    after = len(list(cache_dir.glob("*.pt")))
    report = {"studies": len(ids), "unreadable": len(bad), "cache_entries_before": before, "cache_entries_after": after,
              "cache_bytes": sum(f.stat().st_size for f in cache_dir.glob("*.pt")), "seconds": round(seconds, 1),
              "seconds_per_study": round(seconds / max(len(ids), 1), 3), "workers": int(config["num_workers"]),
              "excluded_missing_series": len(excluded)}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "excluded_unreadable.json").write_text(json.dumps(sorted(bad)))
    (args.work_dir / "cache_warm.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def cmd_common_eval(args) -> int:
    """Score a checkpoint on the common weak-validation yardstick shared by several manifests (never touches gold studies)."""
    import numpy as np
    import torch
    from mri_core.weak_label_models import build_model
    from mri_core.weak_label_train import predict_probabilities
    from mri_core.weak_val_common import TIER1, common_yardstick, score_on_yardstick
    meta_dir, dicom_root = _paths(args)
    metadata = load_rsna_metadata(meta_dir, dicom_root)
    gold = set(metadata.complete_label_train["StudyInstanceUID"].astype(str))
    manifests = {}
    for item in args.manifests:
        name, path = item.split("=", 1)
        manifests[name] = load_weak_manifest(Path(path))
        assert_no_leakage(manifests[name].ids, [], gold_ids=gold)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = {**_config(args), **state["config"]}
    yard = common_yardstick(manifests, val_fraction=float(config["val_fraction"]), seed=int(config["seed"]))
    excluded = set(json.loads(Path(args.exclude_studies).read_text())) if args.exclude_studies else set()
    missing = preflight_series(metadata, [u for u in yard["ids"] if u not in excluded], config["planes"])  # same filter as training
    uids = [u for u in yard["ids"] if u not in excluded and u not in missing]
    # a study may only be scored if it was NOT in this checkpoint's own training split
    overlap = set(uids) & set(state["train_study_ids"])
    if overlap:
        raise SystemExit(f"{len(overlap)} yardstick studies were in this checkpoint's training split; refusing")
    dataset = WeakLabelDicomDataset(metadata, uids, np.zeros((len(uids), 12), dtype=np.float32), np.zeros((len(uids), 12), dtype=np.float32),
                                    config, on_error="raise", cache_dir=config.get("cache_dir"))
    model = build_model(config["arm"], {**config, "pretrained_weights": None})
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    got_uids, probs = predict_probabilities(model.to(device), dataset, device, batch_size=8, num_workers=0)
    per_target = {t: probs[:, TARGET_COLUMNS.index(t)] for t in TIER1}
    report = score_on_yardstick(per_target, got_uids, yard, rounds=args.rounds, seed=0)
    report.update({"checkpoint_sha256": sha256_file(args.checkpoint), "arm": config["arm"], "epoch": int(state["epoch"]),
                   "manifests": {n: m.sha256 for n, m in manifests.items()}, "scored_studies": len(got_uids),
                   "dropped_missing_series": len(missing)})
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "common_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
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


def cmd_holdout_eval(args) -> int:
    """Score one checkpoint on studies it never saw (not in its train/validation sets, not gold, not test). Weak labels only."""
    import numpy as np
    import torch
    from mri_core.weak_label_models import build_model
    from mri_core.weak_label_train import masked_auc, predict_probabilities
    metadata, manifest, gold = _load(args)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = {**_config(args), **{k: state["config"][k] for k in IDENTITY_KEYS if k in state["config"]}}
    ids, values, masks, _, _ = _select(args, metadata, manifest, config)
    test_ids = set(metadata.test["StudyInstanceUID"].astype(str)) | set(metadata.sample_submission["StudyInstanceUID"].astype(str))
    seen = set(state["train_study_ids"]) | set(state["valid_study_ids"])
    if seen & set(ids):
        raise SystemExit(f"refusing: {len(seen & set(ids))} hold-out studies were in the checkpoint's training/validation sets")
    assert_no_leakage(ids, [], gold_ids=gold["StudyInstanceUID"], test_ids=test_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["arm"], config)
    model.load_state_dict(state["model"])
    model.to(device)
    dataset = WeakLabelDicomDataset(metadata, ids, values, masks, config, on_error="raise", cache_dir=config.get("cache_dir"))
    uids, probs = predict_probabilities(model, dataset, device, num_workers=int(config["num_workers"]))
    assert list(uids) == list(ids)
    labels = np.nan_to_num(values, nan=0.0)
    auc = masked_auc(labels, probs, masks)
    defined = [a for a in auc.values() if a is not None]
    active = [j for j in range(len(TARGET_COLUMNS)) if (masks[:, j] == 1).any()]
    report = {"checkpoint_sha256": sha256_file(args.checkpoint), "epoch": int(state["epoch"]), "studies": len(ids),
              "per_target_auc": auc, "macro_auc": float(np.mean(defined)) if defined else None,
              "prob_std": {TARGET_COLUMNS[j]: float(probs[masks[:, j] == 1, j].std()) for j in active},
              "prob_mean": {TARGET_COLUMNS[j]: float(probs[masks[:, j] == 1, j].mean()) for j in active},
              "class_counts": {TARGET_COLUMNS[j]: {"positive": int(((labels[:, j] == 1) & (masks[:, j] == 1)).sum()),
                                                   "negative": int(((labels[:, j] == 0) & (masks[:, j] == 1)).sum())} for j in active},
              "disjoint_from_train_and_validation": True, "gold_overlap": 0, "test_overlap": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("studies", "macro_auc", "epoch")}))
    return 0


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
        "cache_bytes": cache_bytes, "cache_hits": dataset.cache_hits, "cache_misses": dataset.cache_misses, "free_disk_gb": round(shutil.disk_usage(args.work_dir).free / 1e9, 1),
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
    for name in ("preflight", "real-preflight", "verify-checkpoint", "cache-warm", "cache-warm-multi", "common-eval", "train",
                 "evaluate-gold", "export", "holdout-eval"):
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
    sub.choices["train"].add_argument("--shuffle-train-labels", type=int, default=None, metavar="SEED",
                                      help="negative control: permute labels across training studies (validation labels stay true)")
    sub.choices["train"].add_argument("--val-equals-train", action="store_true",
                                      help="overfit diagnostic: validate on the training studies (never for model selection)")
    sub.choices["train"].add_argument("--stop-after-epochs", type=int, default=0,
                                      help="end this session cleanly after N epochs (the run can be resumed later)")
    sub.choices["holdout-eval"].add_argument("--checkpoint", type=Path, required=True)
    sub.choices["holdout-eval"].add_argument("--output", type=Path, required=True)
    sub.choices["verify-checkpoint"].add_argument("--checkpoint", type=Path, required=True)
    sub.choices["common-eval"].add_argument("--checkpoint", type=Path, required=True)
    sub.choices["common-eval"].add_argument("--manifests", nargs="+", required=True, metavar="NAME=PATH")
    sub.choices["common-eval"].add_argument("--exclude-studies", type=Path, default=None)
    sub.choices["common-eval"].add_argument("--rounds", type=int, default=1000)
    sub.choices["cache-warm-multi"].add_argument("--variants", default="128x9,192x9,128x15,192x15", help="comma-separated SIZExSLICES list")
    for name in ("train", "real-preflight", "verify-checkpoint", "cache-warm", "cache-warm-multi", "holdout-eval"):
        sub.choices[name].add_argument("--include-studies", type=Path, default=None,
                                       help="JSON list of study IDs; only these (minus exclusions) are used")
    for name in ("train", "real-preflight", "verify-checkpoint", "cache-warm", "cache-warm-multi", "holdout-eval"):
        sub.choices[name].add_argument("--exclude-studies", type=Path, default=None,
                                       help="JSON list of study IDs to leave out (e.g. unreadable series found by cache-warm)")
    for name in ("train", "real-preflight", "verify-checkpoint", "cache-warm", "cache-warm-multi"):
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
    if args.command not in ("export", "smoke", "common-eval") and args.manifest is None:
        parser.error("--manifest is required")
    handlers = {"preflight": cmd_preflight, "real-preflight": cmd_real_preflight, "verify-checkpoint": cmd_verify_checkpoint, "cache-warm": cmd_cache_warm, "cache-warm-multi": cmd_cache_warm_multi, "holdout-eval": cmd_holdout_eval, "common-eval": cmd_common_eval, "train": cmd_train, "evaluate-gold": cmd_evaluate_gold, "export": cmd_export,
                "smoke": cmd_smoke}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
