"""Common weak-validation yardstick for models trained on different label versions."""

import numpy as np
import pytest

from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.weak_label_data import WeakLabelManifest, hash_split
from mri_core.weak_val_common import common_yardstick, score_on_yardstick

T = len(TARGET_COLUMNS)


def manifest(ids, values_by_target):
    n = len(ids)
    values = np.full((n, T), np.nan, dtype=np.float32)
    masks = np.zeros((n, T), dtype=np.float32)
    for target, col in values_by_target.items():
        j = TARGET_COLUMNS.index(target)
        for i, v in enumerate(col):
            if v is not None:
                values[i, j], masks[i, j] = v, 1
    return WeakLabelManifest(tuple(ids), values, masks, tuple("en" for _ in ids), {"manifest_sha256": "x"})


def test_only_shared_validation_studies_with_agreeing_decided_labels_count():
    ids = [f"s{i}" for i in range(400)]
    rng = np.random.default_rng(0)
    base = [float(rng.integers(0, 2)) for _ in ids]
    v1 = manifest(ids, {"ACL": base})
    changed = list(base)
    changed[3], changed[7] = 1 - changed[3], 1 - changed[7]
    undecided = list(changed)
    undecided[11] = None
    v2 = manifest(ids, {"ACL": undecided})
    yard = common_yardstick({"v1": v1, "v2": v2}, val_fraction=0.25, seed=2, targets=("ACL",))
    _, val_idx = hash_split(ids, 0.25, 2)
    val = {ids[i] for i in val_idx}
    assert set(yard["ids"]) == val and yard["n_shared_validation_studies"] == len(val)
    agreed = yard["labels"]["ACL"]
    for uid in ("s3", "s7", "s11"):
        assert uid not in agreed  # disagreement or undecided in one manifest removes the study from the yardstick
    assert all(agreed[u] == int(base[int(u[1:])]) for u in agreed)


def test_studies_missing_from_one_manifest_are_excluded():
    ids = [f"s{i}" for i in range(300)]
    full = manifest(ids, {"ACL": [1.0 if i % 3 == 0 else 0.0 for i in range(300)]})
    sub_ids = ids[:200]
    small = manifest(sub_ids, {"ACL": [1.0 if i % 3 == 0 else 0.0 for i in range(200)]})
    yard = common_yardstick({"a": full, "b": small}, val_fraction=0.3, seed=1, targets=("ACL",))
    assert set(yard["ids"]) <= set(sub_ids)


def test_scoring_handles_perfect_random_and_undefined_targets():
    ids = [f"s{i}" for i in range(400)]
    labels = {t: {u: (i % 2) for i, u in enumerate(ids)} for t in ("ACL", "Medial Meniscus")}
    labels["Lateral Meniscus"] = {u: 0 for u in ids}  # one class only -> undefined, never 0
    yard = {"labels": labels, "n_shared_validation_studies": 400, "ids": ids}
    perfect = {t: np.array([float(i % 2) for i in range(400)]) for t in labels}
    out = score_on_yardstick(perfect, ids, yard, rounds=100)
    assert out["per_target"]["ACL"]["auc"] == 1.0 and out["per_target"]["Lateral Meniscus"]["auc"] is None
    assert out["macro_auc"] is None and out["per_target"]["ACL"]["auc_ci95"] == [1.0, 1.0]
    noise = {t: np.random.default_rng(0).random(400) for t in labels}
    assert 0.4 < score_on_yardstick(noise, ids, yard, rounds=50)["per_target"]["ACL"]["auc"] < 0.6


def test_common_eval_cli_scores_a_checkpoint_on_the_shared_yardstick(tmp_path, capsys):
    """End to end on synthetic DICOMs: two label versions, one checkpoint, never gold, never its own training studies."""
    import json
    import pandas as pd
    from scripts.kaggle_weak_label_experiment import main as cli
    from mri_core.report_label_rules import RuleBasedExtractor
    from mri_core.weak_label_manifest import build_manifest, write_manifest
    from .test_weak_label_dicom_and_submission import CFG as DICOM_CFG
    from .test_weak_label_dicom_and_submission import write_study, SERIES_COLUMNS
    from mri_core.rsna_knee_dataset import load_rsna_metadata

    meta, raw = tmp_path / "meta", tmp_path / "raw"
    meta.mkdir()
    ids = [f"1.2.826.{i}" for i in range(1, 25)]
    series_rows = []
    for uid in ids:
        write_study(raw, "train", uid, series_rows)
    en = "Findings: The study of the knee was performed and the following is noted in this report.\n"
    pos = "Complete tear of the anterior cruciate ligament. Tear of the medial meniscus. The lateral meniscus is intact."
    neg = "No tear of the anterior cruciate ligament. The medial meniscus is intact. Tear of the lateral meniscus."
    train = pd.DataFrame({"StudyInstanceUID": ids, "Report": [en + (pos if i % 2 == 0 else neg) for i in range(len(ids))],
                          **{t: np.nan for t in TARGET_COLUMNS}})
    train.to_csv(meta / "train.csv", index=False)
    pd.DataFrame(series_rows, columns=SERIES_COLUMNS).to_csv(meta / "train_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": []}).to_csv(meta / "test.csv", index=False)
    pd.DataFrame(columns=SERIES_COLUMNS).to_csv(meta / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": [], **{t: [] for t in TARGET_COLUMNS}}).to_csv(meta / "sample_submission.csv", index=False)
    paths = {}
    for profile in ("0.1.0", "0.2.2"):
        rows, summary = build_manifest(train, RuleBasedExtractor(profile), active_targets=["ACL", "Medial Meniscus", "Lateral Meniscus"])
        paths[profile] = write_manifest(rows, summary, tmp_path / f"m_{profile}.jsonl")[0]
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({**DICOM_CFG, "epochs": 1, "val_fraction": 0.4, "cache_dir": str(tmp_path / "cache"), "seed": 0}))
    common = ["--data-dir", str(meta), "--dicom-root", str(raw), "--config", str(cfg)]
    assert cli(["train", *common, "--manifest", str(paths["0.1.0"]), "--arm", "C", "--work-dir", str(tmp_path / "run")]) == 0
    capsys.readouterr()
    ckpt = tmp_path / "run" / "last_best.pt"
    # a model may be scored only on studies outside its own training split
    code = cli(["common-eval", *common, "--checkpoint", str(ckpt), "--manifests", f"v1={paths['0.1.0']}", f"v2={paths['0.2.2']}",
                "--work-dir", str(tmp_path / "ev"), "--rounds", "50"])
    out = json.loads((tmp_path / "ev" / "common_eval.json").read_text())
    assert code == 0 and out["arm"] == "C" and set(out["manifests"]) == {"v1", "v2"}
    assert set(out["per_target"]) == {"ACL", "Medial Meniscus", "Lateral Meniscus"} and out["scored_studies"] > 0
