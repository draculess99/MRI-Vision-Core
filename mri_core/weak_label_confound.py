"""Exploratory (post-hoc) acquisition-confounding diagnostics for the weak-label models.

Question: does an image model rank studies by weak label better than simple acquisition characteristics (series counts, slice counts,
pixel size, spacing, thickness, completeness, scanner/protocol fields) read from DICOM headers - no pixel data, no patient identifiers?

Everything here is descriptive: nothing is used to select a model, and the hold-out pool is only scored, never tuned on.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .rsna_knee_dataset import PLANE_NAMES, TARGET_COLUMNS, select_series

HEADER_TAGS = ["Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "ImagePositionPatient",
               "ImageOrientationPatient", "InstanceNumber", "EchoTime", "RepetitionTime", "MagneticFieldStrength", "Manufacturer",
               "ManufacturerModelName"]  # technical acquisition fields only; no patient, date or institution fields are read
ACTIVE = ("ACL", "Medial Meniscus", "Lateral Meniscus")
ACTIVE_IDX = [TARGET_COLUMNS.index(t) for t in ACTIVE]


def _num(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def series_header_features(directory) -> dict:
    """Header-only (stop_before_pixels) summary of one series directory. Raises if nothing readable."""
    import pydicom
    rows = []
    for path in sorted(p for p in directory.iterdir() if p.is_file()):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, specific_tags=HEADER_TAGS)
        except Exception:
            continue
        rows.append(ds)
    if not rows:
        raise ValueError(f"no readable DICOM headers in {directory}")
    first = rows[0]
    positions, instances, normal = [], [], None
    orientation = getattr(first, "ImageOrientationPatient", None)
    if orientation is not None and len(orientation) == 6:
        r, c = np.array(orientation[:3], float), np.array(orientation[3:], float)
        normal = np.cross(r, c)
    for ds in rows:
        pos = getattr(ds, "ImagePositionPatient", None)
        if pos is not None and normal is not None and len(pos) == 3:
            positions.append(float(np.dot(np.array(pos, float), normal)))
        if getattr(ds, "InstanceNumber", None) is not None:
            instances.append(int(ds.InstanceNumber))
    diffs = np.diff(np.sort(positions)) if len(positions) > 1 else np.array([])
    spacing = float(np.median(np.abs(diffs))) if len(diffs) else _num(getattr(first, "SpacingBetweenSlices", None))  # positions beat the tag
    pixel = getattr(first, "PixelSpacing", None) or [np.nan, np.nan]
    inst_span = (max(instances) - min(instances) + 1) if instances else 0
    return {"n_files": len(rows), "rows": _num(getattr(first, "Rows", None)), "cols": _num(getattr(first, "Columns", None)),
            "pixel_spacing": _num(pixel[0]), "slice_thickness": _num(getattr(first, "SliceThickness", None)), "slice_spacing": spacing,
            "span_mm": float(np.ptp(positions)) if len(positions) > 1 else np.nan,
            "instance_gap_fraction": float(1 - len(set(instances)) / inst_span) if inst_span else np.nan,
            "spacing_cv": float(np.std(diffs) / max(np.mean(np.abs(diffs)), 1e-9)) if len(diffs) > 1 else np.nan,
            "te": _num(getattr(first, "EchoTime", None)), "tr": _num(getattr(first, "RepetitionTime", None)),
            "field": _num(getattr(first, "MagneticFieldStrength", None)),
            "manufacturer": str(getattr(first, "Manufacturer", "") or "unknown").strip().lower()[:24]}


def study_header_features(metadata, uid: str, planes: Sequence[str] = PLANE_NAMES, split: str = "train") -> dict:
    series = metadata.series_for(uid, split)
    out = {"n_series": float(len(series)), "n_fluid": float((series["Fluid_Sensitive"] == 1).sum()),
           "n_fatsup": float((series["Fat_Suppression"] == 1).sum())}
    manufacturer = "unknown"
    for plane in planes:
        candidates = series[series["Anatomical_Plane"].astype(str).str.lower() == plane.lower()]
        out[f"{plane}_n_series"] = float(len(candidates))
        if candidates.empty:
            continue
        chosen = select_series(candidates)  # the same series the image model reads
        feats = series_header_features(metadata.dicom_series_dir(uid, chosen, split))
        found = feats.pop("manufacturer")
        if manufacturer == "unknown":
            manufacturer = found
        out.update({f"{plane}_{k}": v for k, v in feats.items()})
    out["manufacturer"] = manufacturer
    return out


def build_feature_table(metadata, ids: Sequence[str], planes: Sequence[str] = PLANE_NAMES, workers: int = 8,
                        split: str = "train") -> tuple[pd.DataFrame, dict]:
    """Rows = studies (index = UID). Unreadable studies are reported, not raised."""
    failures = {}

    def one(uid):
        try:
            return uid, study_header_features(metadata, uid, planes, split)
        except Exception as error:  # report the class only; IDs stay in the output files
            failures[uid] = type(error).__name__
            return uid, None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, ids))
    table = pd.DataFrame({uid: row for uid, row in results if row is not None}).T
    numeric = [c for c in table.columns if c != "manufacturer"]
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce")
    return table, failures


# ------------------------------------------------------------------------------------------------ metadata-only baselines
def design_matrix(table: pd.DataFrame, ids: Sequence[str], fit: Optional[dict] = None, top_manufacturers: int = 6):
    """Numeric header features + one-hot manufacturer. Imputation/scaling/categories are learned from the TRAINING rows only (fit=None)."""
    numeric = [c for c in table.columns if c != "manufacturer"]
    X = table.loc[list(ids), numeric].astype(float)
    if fit is None:
        keep = [c for c in numeric if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
        fit = {"cols": keep, "median": X[keep].median(), "mean": None, "std": None,
               "manufacturers": list(table.loc[list(ids), "manufacturer"].value_counts().index[:top_manufacturers])}
        filled = X[keep].fillna(fit["median"])
        fit["mean"], fit["std"] = filled.mean(), filled.std().fillna(1.0).replace(0, 1.0)  # a single training row has no spread
    filled = X[fit["cols"]].fillna(fit["median"])
    Z = ((filled - fit["mean"]) / fit["std"]).values
    onehot = np.stack([(table.loc[list(ids), "manufacturer"] == m).astype(float).values for m in fit["manufacturers"]], axis=1) \
        if fit["manufacturers"] else np.zeros((len(ids), 0))
    return np.hstack([Z, onehot]), fit


def fit_predict_baselines(table, train_ids, train_values, train_masks, eval_sets: Mapping[str, Sequence[str]], *, c: float = 0.5) -> dict:
    """Fixed, untuned baselines per active target: L2 logistic regression and a small boosted tree (HistGradientBoosting, depth 3).
    Returns {model: {set_name: array [n, 3]}} with NaN where a target has a single class in training."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    Xtr, fit = design_matrix(table, train_ids)
    designs = {name: design_matrix(table, ids, fit)[0] for name, ids in eval_sets.items()}
    out = {"logreg": {n: np.full((len(ids), len(ACTIVE)), np.nan) for n, ids in eval_sets.items()},
           "boosted": {n: np.full((len(ids), len(ACTIVE)), np.nan) for n, ids in eval_sets.items()}}
    for k, j in enumerate(ACTIVE_IDX):
        use = train_masks[:, j] == 1
        y = np.nan_to_num(train_values[use, j]).astype(int)
        if len(set(y)) < 2:
            continue
        lr = LogisticRegression(C=c, max_iter=3000).fit(Xtr[use], y)
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05, random_state=0).fit(
            table.loc[list(train_ids)].drop(columns=["manufacturer"]).astype(float).values[use], y)
        for name, ids in eval_sets.items():
            out["logreg"][name][:, k] = lr.predict_proba(designs[name])[:, 1]
            out["boosted"][name][:, k] = gb.predict_proba(
                table.loc[list(ids)].drop(columns=["manufacturer"]).astype(float).values)[:, 1]
    return out


# ------------------------------------------------------------------------------------------------ metrics and uncertainty
def target_auc(labels: np.ndarray, masks: np.ndarray, scores: np.ndarray, k: int) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    on = masks[:, ACTIVE_IDX[k]] == 1
    y = np.nan_to_num(labels[on, ACTIVE_IDX[k]]).astype(int)
    s = scores[on, k]
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if len(y) and 0 < y.sum() < len(y) else None


def macro_target_auc(labels, masks, scores, idx=None) -> Optional[float]:
    idx = np.arange(len(labels)) if idx is None else idx
    vals = [target_auc(labels[idx], masks[idx], scores[idx], k) for k in range(len(ACTIVE))]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def stratified_auc(y: np.ndarray, score: np.ndarray, strata: np.ndarray) -> tuple[Optional[float], float]:
    """Within-stratum AUC averaged with weights n_pos*n_neg (only comparisons between studies of the same stratum);
    returns (value, share of all positive-negative pairs that lie inside a stratum)."""
    from sklearn.metrics import roc_auc_score
    total = float(y.sum() * (len(y) - y.sum()))
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        pos = int(y[m].sum())
        neg = int(m.sum()) - pos
        if pos and neg:
            w = pos * neg
            num += w * roc_auc_score(y[m], score[m])
            den += w
    return (num / den if den else None), (den / total if total else 0.0)


def strata_from_cuts(values: np.ndarray, cuts: Sequence[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(sorted(set(cuts))), values, side="right")


def macro_stratified_auc(labels, masks, scores, strata_by_target: np.ndarray, idx=None) -> Optional[float]:
    """strata_by_target: [n, 3] integer strata (may differ by target, e.g. baseline-score quintiles)."""
    idx = np.arange(len(labels)) if idx is None else idx
    vals = []
    for k, j in enumerate(ACTIVE_IDX):
        on = masks[idx, j] == 1
        y = np.nan_to_num(labels[idx][on, j]).astype(int)
        s = scores[idx][on, k]
        st = strata_by_target[idx][on, k]
        ok = ~np.isnan(s)
        value, _ = stratified_auc(y[ok], s[ok], st[ok])
        if value is not None:
            vals.append(value)
    return float(np.mean(vals)) if vals else None


def bootstrap(metrics: Mapping[str, Callable[[np.ndarray], Optional[float]]], n: int, *, rounds: int = 1000, seed: int = 0,
              differences: Sequence[tuple[str, str]] = ()) -> dict:
    """Study-level percentile bootstrap. Every metric is evaluated on the SAME resample, so differences are paired."""
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in metrics}
    paired = {f"{a}-{b}": [] for a, b in differences}
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        row = {k: f(idx) for k, f in metrics.items()}
        for k, v in row.items():
            if v is not None:
                draws[k].append(v)
        for a, b in differences:
            if row[a] is not None and row[b] is not None:
                paired[f"{a}-{b}"].append(row[a] - row[b])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None
    point = {k: f(np.arange(n)) for k, f in metrics.items()}
    return {"point": point, "ci95": {k: ci(v) for k, v in draws.items()},
            "paired_difference": {k: {"point": (point[k.split("-")[0]] - point[k.split("-")[1]]) if None not in (point[k.split("-")[0]], point[k.split("-")[1]]) else None,
                                      "ci95": ci(v)} for k, v in paired.items()}, "rounds": rounds}


def spearman_by_target(a: np.ndarray, b: np.ndarray) -> list:
    from scipy.stats import spearmanr
    out = []
    for k in range(a.shape[1]):
        ok = ~(np.isnan(a[:, k]) | np.isnan(b[:, k]))
        out.append(float(spearmanr(a[ok, k], b[ok, k]).correlation) if ok.sum() > 2 else None)
    return out
