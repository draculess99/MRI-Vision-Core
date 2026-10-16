"""Read-only MRNet evaluation: preprocessing parity with the RSNA pipeline, label alignment, meniscus combination, controls, read-only guarantees."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mri_core import mrnet_eval as me
from mri_core.rsna_knee_train import _normalize
from mri_core.weak_label_models import build_model
from scripts.evaluate_mrnet_readonly import main as evaluate_cli, tree_signature

from .test_weak_label_experiment import CFG

CONFIG = {**CFG, "arm": "C", "image_size": 16, "slices_per_series": 3}


def make_mrnet(root: Path, n_train=30, n_valid=10, seed=0):
    rng = np.random.default_rng(seed)
    (root / "labels").mkdir(parents=True)
    images = root / "MRNet_files"
    for p in me.PLANES:
        (images / p).mkdir(parents=True)
    ids = [f"{i:04d}" for i in range(n_train + n_valid)]
    labels = {}
    for target in ("abnormal", "acl", "meniscus"):
        y = (rng.random(len(ids)) < 0.4).astype(int)
        labels[target] = y
        for split, sl in (("train", slice(0, n_train)), ("valid", slice(n_train, None))):
            (root / "labels" / f"{split}_{target}.csv").write_text("\n".join(f"{i},{v}" for i, v in zip(ids[sl], y[sl])) + "\n")
    for exam in ids:
        for p in me.PLANES:
            slices = int(rng.integers(5, 12))
            np.save(images / p / f"{exam}.npy", rng.integers(0, 255, (slices, 24, 24), dtype=np.uint8))
    return ids, labels


def make_checkpoint(path: Path, seed: int):
    torch.manual_seed(seed)
    model = build_model("C", CONFIG)
    torch.save({"model": model.state_dict(), "config": CONFIG}, path)
    return model


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_tensor_matches_an_independent_recomputation_and_pads():
    rng = np.random.default_rng(0)
    vols = {"axial": rng.integers(0, 255, (7, 20, 20), dtype=np.uint8), "coronal": rng.integers(0, 255, (2, 20, 20), dtype=np.uint8),
            "sagittal": rng.integers(0, 255, (5, 20, 20), dtype=np.uint8)}
    images, mask = me.exam_tensor(vols, 16, 4)
    assert images.shape == (3, 4, 1, 16, 16) and mask.sum(dim=1).tolist() == [4, 2, 4]
    for p, plane in enumerate(me.PLANES):
        chosen = np.linspace(0, vols[plane].shape[0] - 1, min(4, vols[plane].shape[0]), dtype=int)
        for k, s in enumerate(chosen):
            ref = F.interpolate(torch.from_numpy(_normalize(vols[plane][int(s)])).unsqueeze(0).unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False)[0]
            assert torch.equal(images[p, k], ref.float())
        assert float(images[p, int(mask[p].sum()):].abs().sum()) == 0.0  # padding is exactly zero
        assert chosen[0] == 0 and chosen[-1] == vols[plane].shape[0] - 1  # first and last slice included


def test_labels_are_aligned_to_exams_and_missing_images_are_refused(tmp_path):
    ids, labels = make_mrnet(tmp_path / "m")
    rows, prov = me.load_reference(tmp_path / "m")
    by = {r["exam_id"]: r for r in rows}
    assert set(by) == set(ids)
    for i, exam in enumerate(ids):
        assert int(by[exam]["acl"]) == labels["acl"][i] and int(by[exam]["meniscus"]) == labels["meniscus"][i]
        assert by[exam]["split"] == ("train" if i < 30 else "valid")
    assert prov["exams"] == 40 and set(prov["label_sha256"]) >= {"train_acl.csv", "valid_meniscus.csv"}
    (tmp_path / "m" / "MRNet_files" / "axial" / "0003.npy").unlink()
    with pytest.raises(ValueError, match="lack an image plane"):
        me.load_reference(tmp_path / "m")


def test_meniscus_score_is_the_larger_of_medial_and_lateral():
    probs = np.zeros((3, 12))
    probs[:, me.J_ACL] = [0.1, 0.2, 0.3]
    probs[:, me.J_MED] = [0.9, 0.1, 0.5]
    probs[:, me.J_LAT] = [0.2, 0.7, 0.4]
    s = me.endpoint_scores(probs)
    assert s[:, 0].tolist() == [0.1, 0.2, 0.3] and s[:, 1].tolist() == [0.9, 0.7, 0.5]


def test_metadata_control_detects_acquisition_leakage_and_is_chance_otherwise():
    rng = np.random.default_rng(0)
    n = 400
    slices = rng.integers(20, 40, n).astype(float)
    meta = np.stack([slices, slices, slices, rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)], axis=1)
    confounded = (slices + rng.normal(scale=3, size=n) > 30).astype(int)
    clean = rng.integers(0, 2, n)
    assert me.auc(confounded, me.metadata_oof_scores(meta, confounded)[:, 0]) > 0.8
    assert abs(me.auc(clean, me.metadata_oof_scores(meta, clean)[:, 0]) - 0.5) < 0.12


def test_evaluate_reports_correct_aucs_and_paired_differences():
    rng = np.random.default_rng(1)
    n = 200
    labels = {e: rng.integers(0, 2, n) for e in me.ENDPOINTS}
    perfect = np.stack([labels[e].astype(float) for e in me.ENDPOINTS], axis=1)
    noise = rng.random((n, 2))
    meta = {e: np.stack([noise[:, k], noise[:, k]], axis=1) for k, e in enumerate(me.ENDPOINTS)}
    untrained = {9001: rng.random((n, 2)), 9002: rng.random((n, 2))}
    res = me.evaluate(labels, {"GOOD": perfect}, untrained, meta, [("GOOD", "META_boosted"), ("GOOD", "UNTRAINED")], rounds=60)
    assert res["auc"]["GOOD|acl"] == 1.0 and res["auc"]["GOOD|mean"] == 1.0 and abs(res["auc"]["META_boosted|acl"] - 0.5) < 0.15
    d = res["paired_difference"]["GOOD|mean-META_boosted|mean"]
    assert d["point"] > 0.3 and d["ci95"][0] > 0.2 and set(res["untrained_per_seed"]) == {9001, 9002}


def test_full_run_is_read_only_and_produces_hashed_outputs(tmp_path):
    root = tmp_path / "m"
    make_mrnet(root)
    ck = tmp_path / "ck"
    ck.mkdir()
    make_checkpoint(ck / "b0.pt", 1)
    make_checkpoint(ck / "c1.pt", 2)
    sig, hashes = tree_signature(root), {p.name: sha(p) for p in ck.iterdir()}
    out = tmp_path / "out"
    assert evaluate_cli(["--mrnet-root", str(root), "--checkpoint", f"B0={ck / 'b0.pt'}", "--checkpoint", f"C1={ck / 'c1.pt'}", "--out", str(out), "--device", "cpu",
                         "--untrained-seeds", "2", "--rounds", "30"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["read_only_verified"] is True and summary["hashes_before"] == summary["hashes_after"]
    assert tree_signature(root) == sig and {p.name: sha(p) for p in ck.iterdir()} == hashes  # nothing was written or modified
    assert summary["predictions_sha256"] == sha(out / "predictions.csv") and len(summary["primary_all_exams"]["untrained_per_seed"]) == 2
    assert len((out / "predictions.csv").read_text().splitlines()) == 41 and summary["secondary_valid_split"]["exams"] == 10
    assert "not clinical accuracy" in summary["claim_scope"]
    assert "C1|meniscus" in summary["primary_all_exams"]["auc"] and "C1|mean-B0|mean" in summary["primary_all_exams"]["paired_difference"]


def test_preflight_never_computes_metrics(tmp_path):
    root = tmp_path / "m"
    make_mrnet(root)
    ck = tmp_path / "ck.pt"
    make_checkpoint(ck, 1)
    out = tmp_path / "out"
    assert evaluate_cli(["--mrnet-root", str(root), "--checkpoint", f"B0={ck}", "--out", str(out), "--device", "cpu", "--preflight", "6"]) == 0
    pre = json.loads((out / "preflight.json").read_text())
    assert pre["preflight"] is True and pre["exams"] == 6 and pre["tensor"]["images_per_exam"] == [3, 3, 1, 16, 16]
    assert not any(k in pre for k in ("primary_all_exams", "secondary_valid_split")) and not (out / "summary.json").exists() and not (out / "predictions.csv").exists()
    assert pre["throughput"]["B0"]["images_per_second_forward"] > 0
