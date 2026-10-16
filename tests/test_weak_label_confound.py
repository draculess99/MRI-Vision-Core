"""Exploratory acquisition-confound diagnostics: header features, metadata baselines, stratified AUC, paired bootstrap."""

import numpy as np
import pandas as pd
import pytest

from mri_core.weak_label_confound import (ACTIVE_IDX, bootstrap, build_feature_table, fit_predict_baselines, macro_stratified_auc,
                                          macro_target_auc, series_header_features, spearman_by_target, stratified_auc,
                                          strata_from_cuts, study_header_features)
from .test_weak_label_resolution import SLICES_IN_VOLUME, world  # noqa: F401  (fixture)

pytestmark = pytest.mark.rsna


def test_header_features_come_from_headers_only_and_describe_the_series(world):  # noqa: F811
    uid = world.ids[0]
    row = world.metadata.series_for(uid).iloc[0]
    feats = series_header_features(world.metadata.dicom_series_dir(uid, row["SeriesInstanceUID"]))
    assert feats["n_files"] == SLICES_IN_VOLUME[row["Anatomical_Plane"]] and feats["rows"] == 10 and feats["cols"] == 12
    assert feats["slice_spacing"] == pytest.approx(3.0, abs=1e-6) and feats["instance_gap_fraction"] == pytest.approx(0.0)
    assert feats["spacing_cv"] == pytest.approx(0.0, abs=1e-6)


def test_study_features_cover_all_planes_and_report_unreadable_studies(world):  # noqa: F811
    study = study_header_features(world.metadata, world.ids[0])
    assert study["n_series"] == 3 and study["Axial_n_files"] == 24 and study["Coronal_n_files"] == 12 and study["Sagittal_n_files"] == 17
    victim = world.ids[1]
    for f in (world.raw / "train_series" / victim / f"{victim}.s.Axial").iterdir():
        f.write_bytes(b"corrupt")
    table, failures = build_feature_table(world.metadata, world.ids, workers=2)
    assert set(failures) == {victim} and victim not in table.index and len(table) == len(world.ids) - 1


def make_table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    n_series = rng.integers(3, 12, n).astype(float)
    table = pd.DataFrame({"n_series": n_series, "noise": rng.normal(size=n), "manufacturer": rng.choice(["ge", "siemens"], n)}, index=ids)
    y = (rng.random(n) < 1 / (1 + np.exp(-(n_series - 7) / 1.5))).astype(float)
    values = np.full((n, 12), np.nan)
    masks = np.zeros((n, 12))
    for j in ACTIVE_IDX:
        values[:, j], masks[:, j] = y, 1
    return ids, table, values, masks


def test_metadata_baselines_learn_an_acquisition_signal_and_use_only_training_rows():
    ids, table, values, masks = make_table()
    tr, ev = ids[:200], {"val": ids[200:]}
    out = fit_predict_baselines(table, tr, values[:200], masks[:200], ev)
    for model in ("logreg", "boosted"):
        auc = macro_target_auc(values[200:], masks[200:], out[model]["val"])
        assert auc > 0.75
    # an evaluation-only change cannot alter predictions of the training fit: refit with different eval rows, same training rows
    again = fit_predict_baselines(table, tr, values[:200], masks[:200], {"val": ids[250:]})
    assert np.allclose(out["logreg"]["val"][50:], again["logreg"]["val"])


def test_stratified_auc_removes_a_confounder_that_explains_the_whole_signal():
    rng = np.random.default_rng(1)
    strata = rng.integers(0, 3, 600)
    y = (rng.random(600) < np.array([0.1, 0.4, 0.7])[strata]).astype(int)
    score = strata + rng.normal(scale=0.01, size=600)  # score = the confounder
    pooled = macro = None
    from sklearn.metrics import roc_auc_score
    pooled = roc_auc_score(y, score)
    within, coverage = stratified_auc(y, score, strata)
    assert pooled > 0.65 and within == pytest.approx(0.5, abs=0.08) and 0 < coverage < 1


def test_strata_cuts_and_paired_bootstrap():
    assert list(strata_from_cuts(np.array([1, 5, 9]), [4, 8])) == [0, 1, 2]
    ids, table, values, masks = make_table(400, seed=2)
    good = np.tile(table["n_series"].values[:, None], (1, 3))
    rng = np.random.default_rng(0)
    bad = rng.normal(size=good.shape)
    res = bootstrap({"good": lambda i: macro_target_auc(values, masks, good, i), "bad": lambda i: macro_target_auc(values, masks, bad, i)},
                    400, rounds=200, seed=0, differences=[("good", "bad")])
    d = res["paired_difference"]["good-bad"]
    assert res["point"]["good"] > 0.75 and abs(res["point"]["bad"] - 0.5) < 0.1 and d["ci95"][0] > 0 and res["ci95"]["good"] is not None
    strata = np.tile(table["n_series"].values.astype(int)[:, None], (1, 3))  # one stratum per distinct series count
    assert macro_stratified_auc(values, masks, good, strata) == pytest.approx(0.5, abs=0.1)


def test_spearman_by_target():
    a = np.arange(30, dtype=float).reshape(10, 3)
    assert spearman_by_target(a, a) == pytest.approx([1.0, 1.0, 1.0])
    b = a.copy()
    b[:, 0] = np.nan
    assert spearman_by_target(a, b)[0] is None
