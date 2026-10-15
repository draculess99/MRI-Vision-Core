"""Repeatable evaluation of a report-label extractor against the 58 gold-labeled RSNA knee studies.

Discipline enforced in code (not just documented):
  * The gold studies are split deterministically into a DEV half and an EVAL half by hashing the study UID
    with a fixed salt. Rules / prompts / thresholds may be inspected and changed using DEV only.
  * The EVAL half can be scored once per extractor version. A ledger file records every EVAL run; asking for
    the same extractor_version again raises EvalAlreadySpentError unless the caller explicitly forces it, and a
    forced re-run is recorded as such.
  * Gold studies are never eligible for weak-label training (see weak_label_manifest).

Metrics, per target and per language group (en / es / unsupported):
  positive precision & recall, negative reliability (precision of `negative`) and negative recall,
  coverage (fraction decided), 4-state x gold confusion counts, AUROC (only when both classes exist and each
  has >= MIN_CLASS_COUNT members), and percentile-bootstrap 95% CIs (resampling studies).

The first prototype (scratchpad, 58/58 studies, lexicon written without gold but evaluated on all of it) is
exploratory only. Numbers from this harness are the ones to cite, and only the EVAL half is untouched
by subsequent rule work -- and even that half was seen once by the prototype evaluation, so treat it as
"lightly contaminated", not as an independent test set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .report_labels import LabelState, ReportLabelExtractor, ReportLabels
from .rsna_knee_dataset import TARGET_COLUMNS

SPLIT_SALT = "rsna-knee-weak-label-gold-split-v1"
DEV_FRACTION = 0.5
MIN_CLASS_COUNT = 3
BOOTSTRAP_ROUNDS = 1000
STATES = tuple(state.value for state in LabelState)
AUROC_SCORE = {LabelState.POSITIVE: 1.0, LabelState.NEGATIVE: 0.0, LabelState.UNCERTAIN: 0.5, LabelState.NOT_MENTIONED: 0.5}


class EvalAlreadySpentError(RuntimeError):
    pass


def gold_studies(train: pd.DataFrame) -> pd.DataFrame:
    """Studies with all 12 structured targets present (the 58 gold studies)."""
    return train.dropna(subset=list(TARGET_COLUMNS)).reset_index(drop=True)


def split_of(study_uid: str, salt: str = SPLIT_SALT, dev_fraction: float = DEV_FRACTION) -> str:
    bucket = int(hashlib.sha256(f"{salt}:{study_uid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if bucket < dev_fraction else "eval"


def assign_splits(gold: pd.DataFrame, **kwargs) -> pd.Series:
    return gold["StudyInstanceUID"].astype(str).map(lambda uid: split_of(uid, **kwargs))


def language_group(labels: ReportLabels) -> str:
    return labels.language if labels.supported else "unsupported"


def extract_all(extractor: ReportLabelExtractor, frame: pd.DataFrame) -> list[ReportLabels]:
    return [extractor.extract(str(row.Report), str(row.StudyInstanceUID)) for row in frame.itertuples()]


def _confusion(states: Sequence[LabelState], truth: np.ndarray) -> dict:
    return {state: {"gold_positive": int(sum(1 for s, t in zip(states, truth) if s.value == state and t == 1)),
                    "gold_negative": int(sum(1 for s, t in zip(states, truth) if s.value == state and t == 0))}
            for state in STATES}


def _metrics(states: Sequence[LabelState], truth: np.ndarray) -> dict:
    n = len(truth)
    predicted_pos = np.array([s is LabelState.POSITIVE for s in states])
    predicted_neg = np.array([s is LabelState.NEGATIVE for s in states])
    gold_pos, gold_neg = truth == 1, truth == 0

    def ratio(num, den):
        return float(num / den) if den else None

    scores = np.array([AUROC_SCORE[s] for s in states])
    auroc = None
    if gold_pos.sum() >= MIN_CLASS_COUNT and gold_neg.sum() >= MIN_CLASS_COUNT:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(truth, scores))
    return {
        "n": int(n), "gold_positive": int(gold_pos.sum()), "gold_negative": int(gold_neg.sum()),
        "positive_precision": ratio((predicted_pos & gold_pos).sum(), predicted_pos.sum()),
        "positive_recall": ratio((predicted_pos & gold_pos).sum(), gold_pos.sum()),
        "negative_reliability": ratio((predicted_neg & gold_neg).sum(), predicted_neg.sum()),
        "negative_recall": ratio((predicted_neg & gold_neg).sum(), gold_neg.sum()),
        "coverage": ratio((predicted_pos | predicted_neg).sum(), n),
        "auroc": auroc,
        "confusion": _confusion(states, truth),
    }


def _bootstrap(per_study_states: list, truth: np.ndarray, rounds: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    keys = ("positive_precision", "positive_recall", "negative_reliability", "negative_recall", "coverage", "auroc")
    draws = {k: [] for k in keys}
    n = len(truth)
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        metrics = _metrics([per_study_states[i] for i in idx], truth[idx])
        for k in keys:
            if metrics[k] is not None:
                draws[k].append(metrics[k])
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None)
            for k, v in draws.items()}


def evaluate(extracted: Sequence[ReportLabels], gold: pd.DataFrame, *, rounds: int = BOOTSTRAP_ROUNDS,
             seed: int = 0) -> dict:
    """Metrics for already-extracted labels aligned row-for-row with `gold`."""
    if len(extracted) != len(gold):
        raise ValueError("extracted labels and gold rows must align")
    groups = np.array([language_group(r) for r in extracted])
    report = {"n_studies": len(gold), "language_groups": {str(g): int((groups == g).sum()) for g in sorted(set(groups))},
              "bootstrap_rounds": rounds, "targets": {}}
    for target in TARGET_COLUMNS:
        truth_all = gold[target].to_numpy(dtype=float).astype(int)
        states_all = [r.state(target) for r in extracted]
        entry = {"overall": _metrics(states_all, truth_all)}
        entry["overall"]["ci95"] = _bootstrap(states_all, truth_all, rounds, seed)
        for group in map(str, sorted(set(groups))):
            mask = groups == group
            states = [s for s, m in zip(states_all, mask) if m]
            entry[group] = _metrics(states, truth_all[mask])
            if mask.sum() >= 10:
                entry[group]["ci95"] = _bootstrap(states, truth_all[mask], rounds, seed)
        report["targets"][target] = entry
    aurocs = [v["overall"]["auroc"] for v in report["targets"].values() if v["overall"]["auroc"] is not None]
    report["macro_auroc_defined_targets"] = float(np.mean(aurocs)) if aurocs else None
    report["targets_with_auroc"] = len(aurocs)
    report["mean_coverage"] = float(np.mean([v["overall"]["coverage"] for v in report["targets"].values()]))
    return report


def run_gold_evaluation(extractor: ReportLabelExtractor, train: pd.DataFrame, *, split: str,
                        ledger_path: Optional[Path] = None, force: bool = False,
                        rounds: int = BOOTSTRAP_ROUNDS, seed: int = 0) -> dict:
    """Evaluate on the DEV half, the EVAL half, or both ("all", read-only exploratory use).

    EVAL scoring is recorded in `ledger_path` and may happen once per extractor.version.
    """
    if split not in ("dev", "eval", "all"):
        raise ValueError("split must be dev, eval or all")
    gold = gold_studies(train)
    parts = assign_splits(gold)
    ledger = Path(ledger_path) if ledger_path else None
    if split in ("eval", "all"):
        spent = ledger is not None and ledger.is_file() and any(
            json.loads(line)["extractor_version"] == extractor.version
            for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())
        if spent and not force:
            raise EvalAlreadySpentError(
                f"The EVAL split was already scored for extractor {extractor.version}. Bump the extractor version "
                f"after changing rules, or pass force=True to re-score (recorded in the ledger).")
        if ledger is not None:
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(ledger, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"extractor_version": extractor.version, "split": split,
                                         "forced": bool(spent and force), "n": int(len(gold))}) + "\n")
    selected = gold if split == "all" else gold[parts == split].reset_index(drop=True)
    report = evaluate(extract_all(extractor, selected), selected, rounds=rounds, seed=seed)
    report.update({"split": split, "extractor_version": extractor.version, "split_salt": SPLIT_SALT,
                   "split_sizes": {k: int(v) for k, v in parts.value_counts().items()}})
    return report


def format_summary(report: dict) -> str:
    lines = [f"split={report.get('split')} extractor={report.get('extractor_version')} n={report['n_studies']} "
             f"groups={report['language_groups']}",
             f"mean coverage={report['mean_coverage']:.3f}  macro AUROC (targets with both classes)="
             f"{report['macro_auroc_defined_targets']}  [{report['targets_with_auroc']} targets]",
             f"{'target':18s} {'pos P':>6s} {'pos R':>6s} {'neg rel':>7s} {'cover':>6s} {'AUROC':>6s}"]
    fmt = lambda v: "   n/a" if v is None else f"{v:6.2f}"
    for target, entry in report["targets"].items():
        o = entry["overall"]
        lines.append(f"{target:18s} {fmt(o['positive_precision'])} {fmt(o['positive_recall'])} "
                     f"{o['negative_reliability'] if o['negative_reliability'] is None else round(o['negative_reliability'], 2)!s:>7s} "
                     f"{fmt(o['coverage'])} {fmt(o['auroc'])}")
    return "\n".join(lines)
