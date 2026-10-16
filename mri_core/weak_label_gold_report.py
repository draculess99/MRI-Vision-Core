"""One-look, paired gold-label evaluation of frozen weak-label checkpoints (trained targets only).

Rules enforced here:
  * a checkpoint whose training/validation study IDs include any gold study is refused;
  * each checkpoint file hash may be scored once (ledger); a forced repeat is recorded as such;
  * thresholds are fixed in advance by the caller; nothing is tuned on gold;
  * every checkpoint is scored on the SAME eligible studies (a study unreadable for any model is dropped for all and reported);
  * undefined metrics are None, never 0;
  * raw probabilities are returned so any later analysis never needs another look.

Only the targets a model was trained on are evaluated; the untrained, masked output heads are ignored.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch

from .rsna_knee_dataset import TARGET_COLUMNS
from .rsna_knee_train import RSNAKneeDicomDataset
from .weak_label_data import assert_no_leakage
from .weak_label_models import build_model
from .weak_label_train import GoldAlreadyScoredError, sha256_file

TIER1 = ("ACL", "Medial Meniscus", "Lateral Meniscus")
DEFAULT_THRESHOLDS = (0.5,)


def auc_or_none(y: np.ndarray, score: np.ndarray, min_class: int = 3) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    pos = int(y.sum())
    if pos < min_class or len(y) - pos < min_class:
        return None
    return float(roc_auc_score(y, score))


def wilson(k: int, n: int, z: float = 1.96) -> Optional[list]:
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(centre - half) / d, (centre + half) / d]


def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    tn, fn = int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    ratio = lambda a, b: (a / b) if b else None
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": ratio(tp, tp + fn), "sensitivity_ci95": wilson(tp, tp + fn),
            "specificity": ratio(tn, tn + fp), "specificity_ci95": wilson(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn), "predicted_positive_fraction": float(pred.mean())}


def bootstrap_auc(y: np.ndarray, probs: Mapping[str, np.ndarray], rounds: int, seed: int) -> dict:
    """Paired percentile bootstrap over studies for every model's AUC and for every pairwise difference."""
    rng = np.random.default_rng(seed)
    names = list(probs)
    draws = {n: [] for n in names}
    diffs = {(a, b): [] for i, a in enumerate(names) for b in names[i + 1:]}
    skipped = 0
    for _ in range(rounds):
        idx = rng.integers(0, len(y), len(y))
        aucs = {n: auc_or_none(y[idx], probs[n][idx], min_class=1) for n in names}
        if any(v is None for v in aucs.values()):
            skipped += 1
            continue
        for n in names:
            draws[n].append(aucs[n])
        for (a, b) in diffs:
            diffs[(a, b)].append(aucs[a] - aucs[b])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 50 else None
    return {"auc_ci95": {n: ci(draws[n]) for n in names}, "difference_ci95": {f"{a}-{b}": ci(v) for (a, b), v in diffs.items()},
            "difference_draws": {f"{a}-{b}": v for (a, b), v in diffs.items()}, "skipped_draws": skipped}


def predict_gold(model: torch.nn.Module, samples: Sequence[dict], device: torch.device, batch: int = 8) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            images = torch.stack([s["images"] for s in chunk]).to(device)
            mask = torch.stack([s["mask"] for s in chunk]).to(device)
            out.append(model(images, mask).float().sigmoid().cpu().numpy())
    return np.vstack(out)


def load_samples(dataset: RSNAKneeDicomDataset) -> tuple[list, list]:
    """Decode every gold study once; unreadable studies are returned separately instead of aborting."""
    samples, failed = [], []
    for i in range(len(dataset)):
        uid = str(dataset.studies.iloc[i]["StudyInstanceUID"])
        try:
            samples.append({**dataset[i], "uid": uid})
        except Exception as exc:
            failed.append({"study": "..." + uid[-6:], "error": type(exc).__name__ + ": " + str(exc)[:140]})
    return samples, failed


def score_models_on_gold(checkpoints: Mapping[str, Path], gold_frame, metadata, *, ledger_path: Path, force: bool = False,
                         targets: Sequence[str] = TIER1, thresholds: Sequence[float] = DEFAULT_THRESHOLDS, rounds: int = 2000,
                         seed: int = 0, device: str = "cpu", dataset_factory: Optional[Callable] = None) -> dict:
    gold_ids = list(gold_frame["StudyInstanceUID"].astype(str))
    states, digests = {}, {}
    for name, path in checkpoints.items():
        states[name] = torch.load(path, map_location="cpu", weights_only=True)
        assert_no_leakage(states[name]["train_study_ids"], states[name]["valid_study_ids"], gold_ids=gold_ids)
        digests[name] = sha256_file(Path(path))
    ledger = Path(ledger_path)
    seen = set()
    if ledger.is_file():
        seen = {json.loads(l)["checkpoint_sha256"] for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()}
    repeated = [n for n, d in digests.items() if d in seen]
    if repeated and not force:
        raise GoldAlreadyScoredError(f"already scored on gold: {repeated}")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as handle:
        for name, digest in digests.items():
            handle.write(json.dumps({"checkpoint_sha256": digest, "name": name, "forced": digest in seen, "n_gold": len(gold_ids)}) + "\n")

    configs = {n: s["config"] for n, s in states.items()}
    sizes = {(c["image_size"], c["slices_per_series"], tuple(c["planes"])) for c in configs.values()}
    if len(sizes) != 1:
        raise ValueError("models must share preprocessing settings to be compared on identical tensors")
    config = next(iter(configs.values()))
    dataset = (dataset_factory or (lambda: RSNAKneeDicomDataset(metadata, gold_frame, config, split="train")))()
    samples, failed = load_samples(dataset)
    labels = gold_frame.set_index(gold_frame["StudyInstanceUID"].astype(str)).loc[[s["uid"] for s in samples], list(TARGET_COLUMNS)]
    cols = [TARGET_COLUMNS.index(t) for t in targets]
    y_all = labels.to_numpy(dtype=np.float32)[:, cols].astype(int)

    dev = torch.device(device)
    probs = {}
    for name, state in states.items():
        model = build_model(state["config"]["arm"], {**state["config"], "pretrained_weights": None}).to(dev)
        model.load_state_dict(state["model"])
        probs[name] = predict_gold(model, samples, dev)[:, cols]

    report = {"n_gold_total": len(gold_ids), "n_evaluated": len(samples), "invalid_or_missing": failed, "targets": list(targets),
              "thresholds": list(thresholds), "bootstrap_rounds": rounds, "seed": seed, "checkpoints": digests,
              "models": {n: {"arm": s["config"]["arm"], "epoch": int(s["epoch"]), "dataset_fingerprint": s["dataset_fingerprint"],
                             "n_train_studies": len(s["train_study_ids"]), "n_valid_studies": len(s["valid_study_ids"]),
                             "gold_overlap_with_training": 0} for n, s in states.items()},
              "per_target": {}, "macro": {}}
    for j, target in enumerate(targets):
        y = y_all[:, j]
        pos = int(y.sum())
        entry = {"positives": pos, "negatives": int(len(y) - pos), "prevalence": float(y.mean()), "models": {}}
        boot = bootstrap_auc(y, {n: p[:, j] for n, p in probs.items()}, rounds, seed + j)
        for n, p in probs.items():
            entry["models"][n] = {"auc": auc_or_none(y, p[:, j]), "auc_ci95": boot["auc_ci95"][n],
                                  "confusion": [confusion_at(y, p[:, j], t) for t in thresholds],
                                  "mean_probability": float(p[:, j].mean())}
        entry["paired_auc_difference"] = {k: {"observed": (entry["models"][k.split("-")[0]]["auc"] - entry["models"][k.split("-")[1]]["auc"])
                                               if None not in (entry["models"][k.split("-")[0]]["auc"], entry["models"][k.split("-")[1]]["auc"]) else None,
                                               "ci95": boot["difference_ci95"][k]} for k in boot["difference_ci95"]}
        entry["skipped_bootstrap_draws"] = boot["skipped_draws"]
        report["per_target"][target] = entry

    # macro over the trained targets: same resampled studies across targets
    rng = np.random.default_rng(seed + 1000)
    macro_draws = {n: [] for n in probs}
    macro_diff = {k: [] for k in report["per_target"][targets[0]]["paired_auc_difference"]}
    for _ in range(rounds):
        idx = rng.integers(0, len(y_all), len(y_all))
        per = {n: [auc_or_none(y_all[idx, j], probs[n][idx, j], 1) for j in range(len(targets))] for n in probs}
        if any(v is None for vals in per.values() for v in vals):
            continue
        macros = {n: float(np.mean(v)) for n, v in per.items()}
        for n in probs:
            macro_draws[n].append(macros[n])
        for k in macro_diff:
            a, b = k.split("-")
            macro_diff[k].append(macros[a] - macros[b])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 50 else None
    for n in probs:
        aucs = [report["per_target"][t]["models"][n]["auc"] for t in targets]
        report["macro"][n] = {"auc": None if None in aucs else float(np.mean(aucs)), "auc_ci95": ci(macro_draws[n])}
    diff = lambda k: None if None in (report["macro"][k.split("-")[0]]["auc"], report["macro"][k.split("-")[1]]["auc"]) else         report["macro"][k.split("-")[0]]["auc"] - report["macro"][k.split("-")[1]]["auc"]
    report["macro_paired_difference"] = {k: {"observed": diff(k),
                                             "ci95": ci(v), "bootstrap_draws": len(v)} for k, v in macro_diff.items()}
    report["_arrays"] = {"uids": [s["uid"] for s in samples], "labels": y_all, "probabilities": probs}
    return report
