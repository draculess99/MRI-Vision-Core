"""Model C readiness: tensor contract, parameter counts, masking invariance, optional GRU sequence model, random-init fallback."""

import pytest
import torch

from mri_core.weak_label_models import RSNAKneePretrainedMIL, build_model
from mri_core.weak_label_train import masked_bce

from .test_weak_label_experiment import CFG, fit, make_dataset

T = 12


def params(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestModelC:
    def test_random_init_is_the_default_and_unused_classifier_is_frozen(self):
        model = build_model("C", CFG)
        total, trainable = params(model)
        assert model.pretrained_info is None and model.sequence == "none"
        assert total == 11_762_933 and total - trainable == 513_000  # ImageNet fc (512x1000+1000) is present but frozen
        assert not any(p.requires_grad for p in model.backbone.fc.parameters())

    @pytest.mark.parametrize("sequence", ["none", "gru"])
    def test_input_output_contract_and_masking(self, sequence):
        ds = make_dataset(4)
        model = build_model("C", {**CFG, "sequence": sequence}).eval()
        out = model(ds.images, ds.mask)
        assert out.shape == (4, T) and torch.isfinite(out).all()
        noisy = ds.images.clone()
        noisy[~ds.mask] = 50.0  # padded slices never influence the result, GRU included
        assert torch.allclose(out, model(noisy, ds.mask), atol=1e-5)
        with pytest.raises(ValueError):
            model(ds.images[:, :2], ds.mask[:, :2])  # wrong number of planes

    def test_gru_adds_parameters_and_is_off_by_default(self):
        base, gru = build_model("C", CFG), build_model("C", {**CFG, "sequence": "gru"})
        assert gru.sequence_model is not None and base.sequence_model is None
        assert params(gru)[0] - params(base)[0] == sum(p.numel() for p in gru.sequence_model.parameters()) > 500_000
        with pytest.raises(ValueError):
            build_model("C", {**CFG, "sequence": "lstm"})

    def test_gradients_flow_only_through_trainable_parts_and_masked_targets_get_none(self):
        ds = make_dataset(8)
        model = build_model("C", {**CFG, "sequence": "gru"}).train()
        logits = model(ds.images, ds.mask)
        mask = torch.zeros(8, T)
        mask[:, :3] = 1
        loss = masked_bce(logits, torch.from_numpy(ds.values), mask, torch.ones(T), torch.ones(T))
        loss.backward()
        assert model.head[1].weight.grad[3:].abs().sum() == 0 and model.head[1].weight.grad[:3].abs().sum() > 0
        assert model.sequence_model.weight_ih_l0.grad is not None and model.backbone.fc.weight.grad is None

    def test_unknown_pretrained_file_never_falls_back_to_a_download(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_model("C", {**CFG, "pretrained_weights": str(tmp_path / "nope.pth")})

    @pytest.mark.parametrize("sequence", ["none", "gru"])
    def test_trains_and_resumes_exactly(self, tmp_path, sequence):
        from mri_core.weak_label_train import train_weak_label
        train, val = make_dataset(16, 0), make_dataset(8, 1, start=100)
        cfg = {**CFG, "arm": "C", "sequence": sequence, "epochs": 2, "batch_size": 4}
        a = train_weak_label(train, val, cfg, tmp_path / "a", fingerprint="fp", train_ids=train.ids, val_ids=val.ids)
        train_weak_label(train, val, cfg, tmp_path / "b", fingerprint="fp", train_ids=train.ids, val_ids=val.ids, stop_after_epochs=1)
        b = train_weak_label(train, val, cfg, tmp_path / "b", fingerprint="fp", train_ids=train.ids, val_ids=val.ids)
        sa = torch.load(tmp_path / "a" / "last.pt", map_location="cpu", weights_only=True)["model"]
        sb = torch.load(tmp_path / "b" / "last.pt", map_location="cpu", weights_only=True)["model"]
        assert b["resumed_from_epoch"] == 1 and all(torch.equal(sa[k], sb[k]) for k in sa)
        assert [h["train_loss"] for h in a["history"]] == [h["train_loss"] for h in b["history"]]
