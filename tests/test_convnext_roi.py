"""ConvNeXt-T plain and ROI-box-pooling models: architecture, masking, ROI semantics, provenance guards and pipeline integration (CPU, tiny configs)."""

import json

import numpy as np
import pytest
import torch

from mri_core.convnext_roi import (FULL_BOX, TRUNK_PARAMETERS, ConvNeXtMIL, ConvNeXtTrunk, RoiDataset, RoiError, box_pool_grid, build_convnext_model,
                                   load_roi_table, load_trunk_weights, roi_tensors, validate_boxes)
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.weak_label_models import build_model
from mri_core.weak_label_train import InMemoryWeakDataset, predict_probabilities, train_weak_label

from .test_weak_label_experiment import CFG

SMALL = {"convnext_depths": (1, 1, 1, 1), "convnext_dims": (8, 16, 32, 64)}
ACL, MED = TARGET_COLUMNS.index("ACL"), TARGET_COLUMNS.index("Medial Meniscus")


def model(arm="D", **kw):
    torch.manual_seed(0)
    return build_convnext_model(arm, {**SMALL, "dropout": 0.0, "drop_path": 0.0, **kw}).eval()


def batch(b=2, p=3, s=4, size=64, valid=(4, 3, 2), seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(b, p, s, 1, size, size, generator=g)
    mask = torch.zeros(b, p, s, dtype=torch.bool)
    for j, n in enumerate(valid):
        mask[:, j, :n] = True
    images[~mask] = 0
    return images, mask


def test_convnext_tiny_matches_the_published_architecture():
    trunk = ConvNeXtTrunk()
    assert sum(p.numel() for p in trunk.parameters()) == TRUNK_PARAMETERS == 27_820_128  # torchvision convnext_tiny minus its 1000-class head
    keys = set(trunk.state_dict())
    for k in ("features.0.0.weight", "features.0.1.weight", "features.1.0.layer_scale", "features.1.0.block.0.weight", "features.1.0.block.2.weight", "features.1.0.block.3.weight",
              "features.1.0.block.5.bias", "features.2.0.weight", "features.2.1.weight", "features.7.2.block.5.weight", "out_norm.weight"):
        assert k in keys, k
    assert [len(trunk.features[i]) for i in (1, 3, 5, 7)] == [3, 3, 9, 3]
    with torch.no_grad():
        assert trunk.eval()(torch.zeros(1, 3, 256, 256)).shape == (1, 768, 8, 8)  # stride 32


def test_local_pretrained_weights_load_and_mismatches_are_refused(tmp_path):
    a, b = ConvNeXtTrunk(**{"depths": (1, 1, 1, 1), "dims": (8, 16, 32, 64)}).eval(), ConvNeXtTrunk(**{"depths": (1, 1, 1, 1), "dims": (8, 16, 32, 64)}).eval()
    state = {("classifier.0." + k[len("out_norm."):] if k.startswith("out_norm.") else k): v for k, v in a.state_dict().items()}
    state["classifier.2.weight"] = torch.zeros(1000, 64)  # the ImageNet head is ignored
    path = tmp_path / "w.pt"
    torch.save(state, path)
    assert load_trunk_weights(b, path)["loaded_tensors"] == len(a.state_dict())
    x = torch.rand(1, 3, 64, 64)
    assert torch.equal(a(x), b(x))
    torch.save({k: v for k, v in state.items() if not k.startswith("features.7")}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="does not match"):
        load_trunk_weights(ConvNeXtTrunk(depths=(1, 1, 1, 1), dims=(8, 16, 32, 64)), tmp_path / "bad.pt")


def test_forward_shapes_and_padded_slices_do_not_influence_the_output():
    images, mask = batch()
    for arm in ("D", "E"):
        m = model(arm)
        with torch.no_grad():
            out = m(images, mask)
            assert out.shape == (2, 12) and torch.isfinite(out).all()
            noisy = images.clone()
            noisy[~mask] = torch.rand_like(noisy[~mask])
            assert torch.allclose(m(noisy, mask), out, atol=1e-6)


def test_box_pooling_samples_inside_the_box_and_rejects_invalid_boxes():
    fmap = torch.zeros(2, 1, 8, 8)
    fmap[0, 0, :, 4:] = 1.0  # right half hot
    fmap[1, 0, :, :4] = 1.0  # left half hot
    left = torch.tensor([[0.0, 0.0, 0.5, 1.0]] * 2)
    right = torch.tensor([[0.5, 0.0, 1.0, 1.0]] * 2)
    assert box_pool_grid(fmap, left).mean(dim=(1, 2, 3)).tolist() == pytest.approx([0.0, 1.0], abs=1e-5)
    assert box_pool_grid(fmap, right).mean(dim=(1, 2, 3)).tolist() == pytest.approx([1.0, 0.0], abs=1e-5)
    validate_boxes(torch.tensor([FULL_BOX]))
    for bad in ([0.5, 0.0, 0.4, 1.0, 0.0, 1.0], [0, 0, 1, 1, 0.8, 0.2], [-0.1, 0, 1, 1, 0, 1], [0, 0, 1, 1.2, 0, 1]):
        with pytest.raises(RoiError):
            validate_boxes(torch.tensor([bad], dtype=torch.float32))


def test_roi_targets_use_only_slices_inside_the_z_range_and_other_targets_ignore_the_boxes():
    m = model("E")
    images, mask = batch(b=1, p=3, s=6, valid=(6, 6, 6))
    rois = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.4]).expand(1, 3, 2, 6)  # first slices only (positions 0, 0.2, 0.4) for both ROI targets
    changed = images.clone()
    changed[0, :, 5] = torch.rand(3, 1, 64, 64)  # the last slice of every series, far outside the z-range
    with torch.no_grad():
        a, b = m(images, mask, rois), m(changed, mask, rois)
    assert torch.allclose(a[0, [ACL, MED]], b[0, [ACL, MED]], atol=1e-6)  # ROI targets cannot see the out-of-range slice
    other = [t for t in range(12) if t not in (ACL, MED)]
    assert not torch.allclose(a[0, other], b[0, other], atol=1e-6)  # the other targets still use every slice


def test_missing_boxes_fall_back_to_the_whole_image_and_strict_mode_refuses_them():
    m = model("E")
    images, mask = batch()
    full = torch.tensor(FULL_BOX).expand(2, 3, 2, 6)
    with torch.no_grad():
        no_rois = m(images, mask)
        flagged_invalid = m(images, mask, torch.rand(2, 3, 2, 6).clamp(0.1, 0.9).sort(dim=-1).values, torch.zeros(2, 3, 2, dtype=torch.bool))
        explicit_full = m(images, mask, full, torch.ones(2, 3, 2, dtype=torch.bool))
    assert torch.allclose(no_rois, explicit_full, atol=1e-6) and torch.allclose(flagged_invalid, explicit_full, atol=1e-6)
    with pytest.raises(RoiError):
        model("E", strict_rois=True)(images, mask)


def test_roi_changes_the_roi_target_outputs_only():
    m = model("E")
    images, mask = batch(seed=3)
    small = torch.tensor([0.1, 0.1, 0.4, 0.4, 0.0, 1.0]).expand(2, 3, 2, 6)
    with torch.no_grad():
        full, boxed = m(images, mask), m(images, mask, small)
    assert not torch.allclose(full[:, [ACL, MED]], boxed[:, [ACL, MED]], atol=1e-6)
    others = [t for t in range(12) if t not in (ACL, MED)]
    assert torch.allclose(full[:, others], boxed[:, others], atol=1e-6)


def test_gradients_reach_the_backbone_the_in_box_attention_and_the_box_independent_heads():
    m = build_convnext_model("E", {**SMALL, "dropout": 0.0, "drop_path": 0.0}).train()
    images, mask = batch()
    m(images, mask, torch.tensor([0.2, 0.2, 0.8, 0.8, 0.0, 1.0]).expand(2, 3, 2, 6)).sum().backward()
    assert m.spatial_attention.weight.grad.abs().sum() > 0 and m.trunk.features[0][0].weight.grad.abs().sum() > 0 and m.heads[ACL].weight.grad.abs().sum() > 0


def test_roi_tables_require_independent_provenance_and_valid_boxes(tmp_path):
    def table(source, boxes=None):
        path = tmp_path / f"{source}.json"
        path.write_text(json.dumps({"schema": "roi-table/1", "provenance": {"source": source, "description": "test"},
                                    "boxes": boxes or {"s1": {"Sagittal": {"ACL": [0.2, 0.2, 0.8, 0.8, 0.1, 0.9]}}}}))
        return path
    boxes, prov = load_roi_table(table("detector"))
    assert prov.independent and prov.source == "detector"
    with pytest.raises(RoiError, match="not independent"):
        load_roi_table(table("ground_truth"))
    assert not load_roi_table(table("oracle"), allow_oracle=True)[1].independent  # an explicit oracle upper bound is flagged as such
    with pytest.raises(RoiError):
        load_roi_table(table("detector", {"s1": {"Sagittal": {"ACL": [0.9, 0.2, 0.1, 0.8, 0.1, 0.9]}}}))
    with pytest.raises(RoiError, match="unknown target"):
        load_roi_table(table("detector", {"s1": {"Sagittal": {"Nope": [0, 0, 1, 1, 0, 1]}}}))
    bad = tmp_path / "noprov.json"
    bad.write_text(json.dumps({"schema": "roi-table/1", "provenance": {}, "boxes": {}}))
    with pytest.raises(RoiError, match="no provenance"):
        load_roi_table(bad)
    rois, valid = roi_tensors(boxes, ["s1", "s2"], ["Axial", "Coronal", "Sagittal"], ["ACL", "Medial Meniscus"])
    assert valid.tolist() == [[[False, False], [False, False], [True, False]], [[False, False]] * 3] and rois[0, 2, 0].tolist() == pytest.approx([0.2, 0.2, 0.8, 0.8, 0.1, 0.9])
    assert rois[1].tolist() == [[list(FULL_BOX)] * 2] * 3


def toy_data(n=8, size=32):
    g = torch.Generator().manual_seed(1)
    images = torch.rand(n, 3, 3, 1, size, size, generator=g)
    values = np.full((n, 12), np.nan, dtype=np.float32)
    masks = np.zeros((n, 12), dtype=np.float32)
    for j in (ACL, MED):
        masks[:, j] = 1
        values[:, j] = (np.arange(n) % 2).astype(np.float32)
    return InMemoryWeakDataset([f"s{i}" for i in range(n)], images, torch.ones(n, 3, 3, dtype=torch.bool), values, masks)


def run_cfg(arm):
    return {**CFG, **SMALL, "arm": arm, "epochs": 2, "image_size": 32, "slices_per_series": 3, "num_workers": 0, "batch_size": 4, "device": "cpu", "dropout": 0.0,
            "learning_rate": 1e-3}


def test_arms_d_and_e_train_through_the_existing_pipeline_with_a_full_checkpoint(tmp_path):
    assert isinstance(build_model("D", run_cfg("D")), ConvNeXtMIL)  # reachable from the standard factory; ARMS is untouched
    ds = toy_data()
    for arm in ("D",):
        out = train_weak_label(ds, ds, run_cfg(arm), tmp_path / arm, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids)
        state = torch.load(tmp_path / arm / "last.pt", map_location="cpu", weights_only=True)
        assert out["final_epoch"] == 2 and all(state.get(k) is not None for k in ("model", "optimizer", "scheduler", "torch_rng", "python_rng", "numpy_rng", "config", "metrics"))
        again = train_weak_label(ds, ds, run_cfg(arm), tmp_path / arm, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids)
        assert again["resumed_from_epoch"] == 2  # finished runs resume at their last epoch


def test_arm_e_receives_boxes_from_the_roi_dataset_during_training_and_prediction(tmp_path):
    ds = toy_data()
    boxes = {f"s{i}": {p: {"ACL": [0.1, 0.1, 0.9, 0.9, 0.0, 1.0], "Medial Meniscus": [0.2, 0.2, 0.7, 0.7, 0.0, 1.0]} for p in ("Axial", "Coronal", "Sagittal")} for i in range(8)}
    wrapped = RoiDataset(ds, boxes, ["Axial", "Coronal", "Sagittal"], ["ACL", "Medial Meniscus"])
    item = wrapped[0]
    assert item["rois"].shape == (3, 2, 6) and item["roi_valid"].all() and wrapped.values is ds.values
    cfg = {**run_cfg("E"), "strict_rois": True}  # strict: training would raise if boxes did not reach the model
    out = train_weak_label(wrapped, wrapped, cfg, tmp_path, fingerprint="f", train_ids=ds.ids, val_ids=ds.ids)
    assert out["final_epoch"] == 2
    state = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=True)
    m = build_model("E", cfg)
    m.load_state_dict(state["model"])
    uids, probs = predict_probabilities(m, wrapped, torch.device("cpu"), batch_size=4)
    assert probs.shape == (8, 12) and np.isfinite(probs).all()
    with pytest.raises(RoiError):  # without boxes a strict ROI model refuses to run
        predict_probabilities(m, ds, torch.device("cpu"), batch_size=4)
