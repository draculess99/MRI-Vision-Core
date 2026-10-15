"""Weak-label experiment: masked loss, leakage guards, model arms, resumable training and determinism.

Everything here is synthetic (tiny random tensors); no Kaggle, GPU, DICOM or real data is used.
"""

import json
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from mri_core.rsna_checkpoint import CheckpointIncompatibleError
from mri_core.rsna_knee_dataset import TARGET_COLUMNS
from mri_core.weak_label_data import assert_no_leakage, hash_split, load_weak_manifest
from mri_core.weak_label_manifest import LeakageError, build_manifest, write_manifest
from mri_core.report_label_rules import RuleBasedExtractor
from mri_core.weak_label_models import (ARMS, GatedAttention, ResNet18Backbone, RSNAKneeAttnCNN, RSNAKneePretrainedMIL,
                                        build_model, load_pretrained_backbone)
from mri_core.weak_label_train import (DEFAULT_CONFIG, GoldAlreadyScoredError, InMemoryWeakDataset, auc_report,
                                       compute_pos_weight, evaluate_gold, masked_auc, masked_bce, predict_probabilities,
                                       reliability_vector, summarize_runtime, train_weak_label)

T = len(TARGET_COLUMNS)
PLANES, SLICES, SIZE = 3, 3, 16
CFG = {**DEFAULT_CONFIG, "epochs": 4, "batch_size": 4, "num_workers": 0, "device": "cpu", "image_size": SIZE,
       "slices_per_series": SLICES, "warmup_epochs": 1, "learning_rate": 5e-3, "seed": 3}


def make_dataset(n=24, seed=0, start=0, active=(0, 1, 2)):
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, PLANES, SLICES, 1, SIZE, SIZE, generator=g)
    mask = torch.ones(n, PLANES, SLICES, dtype=torch.bool)
    mask[::3, :, -1] = False  # some padded slices
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2, (n, T)).astype(np.float32)
    masks = np.zeros((n, T), dtype=np.float32)
    masks[:, list(active)] = 1
    masks[::5, 0] = 0  # a few entries masked individually
    values[masks == 0] = np.nan
    return InMemoryWeakDataset([f"1.2.826.{start + i}" for i in range(n)], images, mask, values, masks)


# ---------------------------------------------------------------------------------------------- masked loss
class TestMaskedLoss:
    def setup_method(self):
        torch.manual_seed(0)
        self.logits = torch.randn(6, T, requires_grad=True)
        self.labels = torch.randint(0, 2, (6, T)).float()
        self.mask = torch.zeros(6, T)
        self.mask[:, :3] = 1
        self.pw, self.rel = torch.ones(T), torch.ones(T)

    def test_masked_entries_have_no_effect_and_no_gradient(self):
        base = masked_bce(self.logits, self.labels, self.mask, self.pw, self.rel)
        flipped = self.labels.clone()
        flipped[:, 5:] = 1 - flipped[:, 5:]  # change only masked columns
        assert torch.equal(base, masked_bce(self.logits, flipped, self.mask, self.pw, self.rel))
        base.backward()
        assert torch.all(self.logits.grad[:, 3:] == 0) and torch.any(self.logits.grad[:, :3] != 0)

    def test_nothing_trainable_gives_zero_loss_with_a_gradient_path(self):
        loss = masked_bce(self.logits, self.labels, torch.zeros(6, T), self.pw, self.rel)
        assert float(loss.detach()) == 0.0
        loss.backward()  # must not raise

    def test_reliability_weights_rescale_targets(self):
        rel = torch.zeros(T)
        rel[0] = 1.0
        only_first = masked_bce(self.logits, self.labels, self.mask, self.pw, rel)
        direct = torch.nn.functional.binary_cross_entropy_with_logits(self.logits[:, 0], self.labels[:, 0])
        assert torch.isclose(only_first, direct)
        assert float(reliability_vector({"reliability_weights": {"ACL": 0.5}})[0]) == 0.5
        with pytest.raises(ValueError):
            reliability_vector({"reliability_weights": {"Nonsense": 1}})
        with pytest.raises(ValueError):
            reliability_vector({"reliability_weights": {"ACL": -1}})

    def test_pos_weight_uses_only_trainable_entries_and_is_clipped(self):
        values = np.zeros((100, T), dtype=np.float32)
        masks = np.ones((100, T), dtype=np.float32)
        values[:2, 0] = 1  # 2 positives vs 98 negatives -> 49, clipped to 10
        masks[50:, 1] = 0
        values[50:, 1] = 1  # masked positives must not count
        weight = compute_pos_weight(values, masks, 10.0)
        assert weight[0] == 10.0 and weight[1] == 1.0  # target 1 has no trainable positives -> neutral
        assert (weight >= 1.0).all()

    def test_masked_auc_ignores_masked_entries_and_single_class_targets(self):
        labels = np.array([[0], [1], [0], [1], [1]] + [[0]] * 0, dtype=np.float32)
        probs = np.array([[0.1], [0.9], [0.2], [0.8], [0.0]], dtype=np.float32)
        masks = np.array([[1], [1], [1], [1], [0]], dtype=np.float32)
        full = np.zeros((5, T), dtype=np.float32)
        full[:, 0], pr, mk = labels[:, 0], np.zeros((5, T), np.float32), np.zeros((5, T), np.float32)
        pr[:, 0], mk[:, 0] = probs[:, 0], masks[:, 0]
        out = masked_auc(full, pr, mk)
        assert out["ACL"] == 1.0 and out["MCL"] is None


# ---------------------------------------------------------------------------------------------- leakage
class TestLeakage:
    def test_split_is_disjoint_deterministic_and_order_independent(self):
        ids = [f"s{i}" for i in range(200)]
        a = hash_split(ids, 0.2, 1)
        b = hash_split(ids[::-1], 0.2, 1)
        assert set(a[0]).isdisjoint(a[1])
        assert {ids[i] for i in a[1]} == {ids[::-1][i] for i in b[1]}
        assert {ids[i] for i in hash_split(ids, 0.2, 2)[1]} != {ids[i] for i in a[1]}

    def test_gold_test_and_cross_split_overlap_are_rejected(self):
        with pytest.raises(LeakageError):
            assert_no_leakage(["a"], ["b"], gold_ids=["a"])
        with pytest.raises(LeakageError):
            assert_no_leakage(["a"], ["b"], gold_ids=[], test_ids=["b"])
        with pytest.raises(LeakageError):
            assert_no_leakage(["a"], ["a"], gold_ids=[])
        assert_no_leakage(["a"], ["b"], gold_ids=["g"], test_ids=["t"])

    def test_tier_masking_keeps_inactive_targets_out_of_training(self, tmp_path):
        en = "Findings: The study of the knee was performed and the following is noted in this report.\n"
        rows = [{"StudyInstanceUID": f"1.2.826.{i}",
                 "Report": en + "Complete tear of the anterior cruciate ligament. No joint effusion.",
                 **{t: np.nan for t in TARGET_COLUMNS}} for i in range(3)]
        rows.append({"StudyInstanceUID": "1.2.826.gold", "Report": en + "No tear.", **{t: 1.0 for t in TARGET_COLUMNS}})
        built, summary = build_manifest(pd.DataFrame(rows), RuleBasedExtractor(), active_targets=["ACL"])
        out, _ = write_manifest(built, summary, tmp_path / "m.jsonl")
        manifest = load_weak_manifest(out)
        assert "1.2.826.gold" not in manifest.ids
        assert manifest.masks[:, TARGET_COLUMNS.index("ACL")].sum() == 3
        assert manifest.masks[:, 1:].sum() == 0  # effusion was extracted as negative but stays tier-masked
        assert built[0]["targets"]["Effusion"]["state"] == "negative" and built[0]["targets"]["Effusion"]["tier_masked"]
        assert summary["config"]["active_targets"] == ["ACL"] and set(summary["trainable"]) == {"ACL"}
        assert manifest.fingerprint() == load_weak_manifest(out).fingerprint()
        with pytest.raises(Exception):
            build_manifest(pd.DataFrame(rows), RuleBasedExtractor(), active_targets=["Not a target"])


# ---------------------------------------------------------------------------------------------- models
class TestModels:
    @pytest.mark.parametrize("arm", ARMS)
    def test_output_shape_and_finite(self, arm):
        ds = make_dataset(4)
        model = build_model(arm, CFG).eval()
        out = model(ds.images, ds.mask)
        assert out.shape == (4, T) and torch.isfinite(out).all()

    @pytest.mark.parametrize("arm", ARMS)
    def test_padded_slices_do_not_change_the_output(self, arm):
        ds = make_dataset(4)
        model = build_model(arm, CFG).eval()
        a = model(ds.images, ds.mask)
        noisy = ds.images.clone()
        noisy[~ds.mask] = 99.0
        assert torch.allclose(a, model(noisy, ds.mask), atol=1e-5)

    @pytest.mark.parametrize("arm", ARMS)
    def test_every_plane_needs_a_valid_slice(self, arm):
        ds = make_dataset(2)
        bad = ds.mask.clone()
        bad[0, 1] = False
        with pytest.raises(ValueError):
            build_model(arm, CFG)(ds.images, bad)

    def test_attention_weights_sum_to_one_over_valid_instances(self):
        attn = GatedAttention(8)
        emb = torch.randn(2, 5, 8)
        valid = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        pooled, w = attn(emb, valid)
        assert torch.allclose(w.sum(-1), torch.ones(2)) and (w[~valid] == 0).all() and pooled.shape == (2, 8)

    def test_context_stack_uses_neighbouring_valid_slices_and_repeats_edges(self):
        images = torch.arange(4.0).view(1, 1, 4, 1, 1, 1).expand(1, 1, 4, 1, 2, 2).clone()
        mask = torch.tensor([[[True, True, True, False]]])
        stacked = RSNAKneePretrainedMIL.context_stack(images, mask)
        assert stacked.shape == (1, 1, 4, 3, 2, 2)
        assert stacked[0, 0, 0, :, 0, 0].tolist() == [0, 0, 1]  # first slice repeats itself as "previous"
        assert stacked[0, 0, 2, :, 0, 0].tolist() == [1, 2, 2]  # last valid slice repeats itself as "next"

    def test_pretrained_weights_load_from_a_local_torchvision_style_file(self, tmp_path):
        source = ResNet18Backbone()
        torch.save(source.state_dict(), tmp_path / "resnet18.pth")
        model = RSNAKneePretrainedMIL(pretrained_path=tmp_path / "resnet18.pth")
        assert model.pretrained_info["loaded"] > 0
        for (k, a), (_, b) in zip(source.state_dict().items(), model.backbone.state_dict().items()):
            if not k.startswith("fc."):
                assert torch.equal(a, b)
        with pytest.raises(FileNotFoundError):
            RSNAKneePretrainedMIL(pretrained_path=tmp_path / "missing.pth")
        torch.save({"conv1.weight": torch.zeros(1)}, tmp_path / "bad.pth")
        with pytest.raises(ValueError):
            RSNAKneePretrainedMIL(pretrained_path=tmp_path / "bad.pth")

    def test_frozen_batchnorm_stays_in_eval_mode_during_training(self):
        model = RSNAKneePretrainedMIL().train()
        assert not any(m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
        assert model.attention.training

    def test_unknown_arm_is_rejected(self):
        with pytest.raises(ValueError):
            build_model("Z", CFG)


# ---------------------------------------------------------------------------------------------- training / resume
def fit(out_dir, epochs=4, arm="A", hook=None, **kw):
    train, val = make_dataset(24, 0), make_dataset(8, 1, start=100)
    cfg = {**CFG, "epochs": epochs, "arm": arm}
    return train_weak_label(train, val, cfg, out_dir, fingerprint="fp", train_ids=train.ids, val_ids=val.ids,
                            batch_hook=hook, **kw)


def state_of(out_dir):
    return torch.load(out_dir / "last.pt", map_location="cpu", weights_only=True)


class Interrupt(RuntimeError):
    pass


class TestResumableTraining:
    def test_loss_decreases_and_checkpoint_has_full_state(self, tmp_path):
        result = fit(tmp_path / "a", epochs=6)
        assert result["status"] == "complete" and result["final_epoch"] == 6
        assert result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]
        state = state_of(tmp_path / "a")
        for key in ("model", "optimizer", "scheduler", "scaler", "config", "epoch", "global_step", "metrics",
                    "torch_rng", "python_rng", "numpy_rng", "cuda_rng", "loader_rng", "train_study_ids",
                    "valid_study_ids", "dataset_fingerprint", "best_metric"):
            assert key in state, key
        assert state["epoch"] == 6 and state["global_step"] == 6 * 6 and state["scheduler"] is not None
        assert (tmp_path / "a" / "last_best.pt").is_file()

    @pytest.mark.parametrize("arm", ["A", "B", "C"])
    def test_every_arm_trains_one_epoch(self, tmp_path, arm):
        assert fit(tmp_path / arm, epochs=1, arm=arm)["final_epoch"] == 1

    def test_interrupted_run_resumes_to_the_same_weights_as_an_uninterrupted_run(self, tmp_path):
        full = fit(tmp_path / "full", epochs=4)

        def crash(epoch, step):
            if epoch == 2 and step >= 14:  # inside the third epoch, after two checkpoints exist
                raise Interrupt

        with pytest.raises(Interrupt):
            fit(tmp_path / "crashy", epochs=4, hook=crash)
        assert state_of(tmp_path / "crashy")["epoch"] == 2  # the partial epoch left no checkpoint behind
        resumed = fit(tmp_path / "crashy", epochs=4)
        assert resumed["resumed_from_epoch"] == 2 and resumed["final_epoch"] == 4
        a, b = state_of(tmp_path / "full"), state_of(tmp_path / "crashy")
        assert a["global_step"] == b["global_step"]
        for k in a["model"]:
            assert torch.equal(a["model"][k], b["model"][k]), k
        assert [h["train_loss"] for h in full["history"]] == [h["train_loss"] for h in resumed["history"]]
        assert a["scheduler"]["last_epoch"] == b["scheduler"]["last_epoch"] == 4

    def test_training_is_deterministic_for_a_fixed_seed(self, tmp_path):
        fit(tmp_path / "x", epochs=2)
        fit(tmp_path / "y", epochs=2)
        for k, v in state_of(tmp_path / "x")["model"].items():
            assert torch.equal(v, state_of(tmp_path / "y")["model"][k])

    def test_resume_rejects_a_changed_seed_or_dataset(self, tmp_path):
        fit(tmp_path / "r", epochs=1)
        train, val = make_dataset(24, 0), make_dataset(8, 1, start=100)
        with pytest.raises(CheckpointIncompatibleError):
            train_weak_label(train, val, {**CFG, "epochs": 2, "seed": 99}, tmp_path / "r", fingerprint="fp",
                             train_ids=train.ids, val_ids=val.ids)
        with pytest.raises(CheckpointIncompatibleError):
            train_weak_label(train, val, {**CFG, "epochs": 2}, tmp_path / "r", fingerprint="other",
                             train_ids=train.ids, val_ids=val.ids)

    def test_time_budget_pauses_cleanly_then_continues(self, tmp_path):
        first = fit(tmp_path / "t", epochs=4, max_seconds=0.0)
        assert first["status"] == "paused_time_budget" and first["final_epoch"] == 1
        second = fit(tmp_path / "t", epochs=4)
        assert second["status"] == "complete" and second["resumed_from_epoch"] == 1
        summary = summarize_runtime(tmp_path / "t")
        assert summary["sessions"] == 2 and summary["epochs"] == 4 and summary["gpu_epoch_seconds"] == 0.0

    def test_durable_export_copy_matches_the_working_checkpoint(self, tmp_path):
        fit(tmp_path / "w", epochs=2, export_dir=tmp_path / "durable")
        assert (tmp_path / "durable" / "last.pt").read_bytes() == (tmp_path / "w" / "last.pt").read_bytes()
        assert (tmp_path / "durable" / "runtime_log.jsonl").is_file()

    def test_checkpoint_selection_uses_only_weak_validation(self, tmp_path):
        result = fit(tmp_path / "s", epochs=3)
        best = torch.load(tmp_path / "s" / "last_best.pt", map_location="cpu", weights_only=True)
        losses = [h["val_loss"] for h in result["history"]]
        assert abs(-best["best_metric"] - min(losses)) < 1e-9

    def test_works_with_no_network(self, tmp_path, monkeypatch):
        def blocked(*a, **k):
            raise AssertionError("network access attempted")
        monkeypatch.setattr(socket.socket, "connect", blocked)
        monkeypatch.setattr(socket, "getaddrinfo", blocked)
        assert fit(tmp_path / "net", epochs=1, arm="C")["final_epoch"] == 1
        model = build_model("C", {**CFG, "pretrained_weights": None})
        assert predict_probabilities(model, make_dataset(4), torch.device("cpu"))[1].shape == (4, T)


# ---------------------------------------------------------------------------------------------- gold evaluation
class TestGoldEvaluation:
    def gold_frame(self, ids):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"StudyInstanceUID": ids})
        for t in TARGET_COLUMNS:
            frame[t] = rng.integers(0, 2, len(ids)).astype(float)
        return frame

    def test_checkpoint_trained_on_a_gold_study_is_refused(self, tmp_path):
        fit(tmp_path / "g", epochs=1)
        ids = state_of(tmp_path / "g")["train_study_ids"]
        with pytest.raises(LeakageError):
            evaluate_gold(tmp_path / "g" / "last.pt", None, self.gold_frame(ids[:5]), ledger_path=None)

    def test_gold_scoring_report_ledger_and_one_look(self, tmp_path):
        fit(tmp_path / "g2", epochs=1)
        gold_ds = make_dataset(30, seed=7, start=500)
        frame = pd.DataFrame({"StudyInstanceUID": gold_ds.ids})
        for j, t in enumerate(TARGET_COLUMNS):
            frame[t] = (np.arange(30) + j) % 2
        ledger = tmp_path / "ledger.jsonl"
        report = evaluate_gold(tmp_path / "g2" / "last.pt", None, frame, ledger_path=ledger, rounds=50, dataset=gold_ds)
        assert report["n_studies"] == 30 and set(report["per_target_auc"]) == set(TARGET_COLUMNS)
        assert report["macro_ci95"] is not None and report["headline_macro_auc"] is not None
        with pytest.raises(GoldAlreadyScoredError):
            evaluate_gold(tmp_path / "g2" / "last.pt", None, frame, ledger_path=ledger, rounds=5, dataset=gold_ds)
        forced = evaluate_gold(tmp_path / "g2" / "last.pt", None, frame, ledger_path=ledger, force=True, rounds=5,
                               dataset=gold_ds)
        assert forced["forced"] and len(ledger.read_text().strip().splitlines()) == 2

    def test_auc_report_is_perfect_for_a_perfect_predictor_and_has_intervals(self):
        labels = np.tile(np.array([0, 1] * 20, dtype=np.float32)[:, None], (1, T))
        report = auc_report(labels, labels.copy(), rounds=50, headline_targets=("ACL",))
        assert report["macro_auc"] == 1.0 and report["per_target_ci95"]["ACL"] == [1.0, 1.0]
