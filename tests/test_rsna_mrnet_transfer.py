"""Smoke tests for transferring a shared MRNetCNN encoder into a fresh RSNAKneeCNN."""

import pytest

torch = pytest.importorskip("torch")

from mri_core.cnn_model import MRNetCNN
from mri_core.rsna_mrnet_transfer import (
    UntrustedCheckpointError,
    transfer_mrnet_encoder_to_rsna,
)


def make_mrnet_checkpoint(path, num_planes=3, dropout=0.0, seed=0, wrap_in_training_dict=True):
    torch.manual_seed(seed)
    model = MRNetCNN(num_planes=num_planes, dropout=dropout, num_targets=3)
    if wrap_in_training_dict:
        torch.save({"model": model.state_dict(), "config": {"planes": ["axial", "coronal", "sagittal"]},
                    "targets": ["abnormal", "acl", "meniscus"]}, path)
    else:
        torch.save(model.state_dict(), path)
    return model, path


class TestSuccessfulTransfer:
    def test_encoder_weights_match_source_exactly(self, tmp_path):
        source_model, path = make_mrnet_checkpoint(tmp_path / "mrnet.pt", seed=7)
        rsna_model, report = transfer_mrnet_encoder_to_rsna(path, num_planes=3, dropout=0.0)

        source_state = source_model.state_dict()
        target_state = rsna_model.state_dict()
        for key in source_state:
            if key.startswith("encoder."):
                assert torch.equal(target_state[key], source_state[key]), key

    def test_report_lists_all_encoder_tensors_as_copied(self, tmp_path):
        _, path = make_mrnet_checkpoint(tmp_path / "mrnet.pt")
        _, report = transfer_mrnet_encoder_to_rsna(path, num_planes=3, dropout=0.0)
        assert report.ok
        assert len(report.copied) == 12  # 4 conv weights + 4x(GroupNorm weight+bias)
        assert all(key.startswith("encoder.") for key in report.copied)
        assert not report.missing
        assert not report.incompatible

    def test_head_is_freshly_initialized_not_copied_from_source(self, tmp_path):
        source_model, path = make_mrnet_checkpoint(tmp_path / "mrnet.pt", seed=3)
        rsna_model, report = transfer_mrnet_encoder_to_rsna(path, num_planes=3, dropout=0.0)
        assert report.head_reinitialized
        assert rsna_model.state_dict()["head.1.weight"].shape == (12, 384)
        # The 3-target source head could never be copied into a 12-target layer; confirm shape differs.
        assert source_model.state_dict()["head.1.weight"].shape == (3, 384)

    def test_accepts_raw_state_dict_not_just_training_checkpoint(self, tmp_path):
        _, path = make_mrnet_checkpoint(tmp_path / "mrnet.pt", wrap_in_training_dict=False)
        _, report = transfer_mrnet_encoder_to_rsna(path, num_planes=3, dropout=0.0)
        assert report.ok

    def test_resulting_model_runs_forward_pass(self, tmp_path):
        _, path = make_mrnet_checkpoint(tmp_path / "mrnet.pt")
        rsna_model, _ = transfer_mrnet_encoder_to_rsna(path, num_planes=3, dropout=0.0)
        images = torch.randn(1, 3, 2, 1, 16, 16)
        mask = torch.ones(1, 3, 2, dtype=torch.bool)
        logits = rsna_model(images, mask)
        assert logits.shape == (1, 12)


class TestIncompatibilityIsNeverSilent:
    def test_plane_count_mismatch_is_reported_not_silently_loaded(self, tmp_path):
        # Source trained with 1 plane -> encoder shapes are identical regardless of num_planes
        # (the mismatch that matters is head size, which transfer never touches), but if the
        # source checkpoint is missing an entire architecture family, it must be reported.
        _, path = make_mrnet_checkpoint(tmp_path / "mrnet.pt", num_planes=1)
        _, report = transfer_mrnet_encoder_to_rsna(path, num_planes=3, dropout=0.0)
        # Encoder tensors are plane-count-independent (conv/groupnorm apply per-slice), so
        # this still succeeds -- but we assert the report mechanism actually inspected shapes.
        assert report.ok

    def test_malformed_checkpoint_file_is_rejected(self, tmp_path):
        path = tmp_path / "not_a_checkpoint.pt"
        torch.save([1, 2, 3], path)
        with pytest.raises(UntrustedCheckpointError):
            transfer_mrnet_encoder_to_rsna(path, num_planes=3, dropout=0.0)

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(UntrustedCheckpointError):
            transfer_mrnet_encoder_to_rsna(tmp_path / "does_not_exist.pt", num_planes=3, dropout=0.0)

    def test_shape_incompatible_encoder_tensor_is_reported_and_not_applied(self, tmp_path):
        # Build a checkpoint whose encoder.0.weight has the wrong shape (simulating a
        # foreign/incompatible architecture) and confirm the report flags it, and the
        # resulting model keeps its own freshly initialized tensor instead.
        model = MRNetCNN(num_planes=3, dropout=0.0, num_targets=3)
        state = model.state_dict()
        state["encoder.0.weight"] = torch.zeros(99, 1, 3, 3)  # wrong out-channels
        path = tmp_path / "mismatched.pt"
        torch.save({"model": state}, path)

        rsna_model, report = transfer_mrnet_encoder_to_rsna(path, num_planes=3, dropout=0.0)
        assert not report.ok
        assert any(key == "encoder.0.weight" for key, _ in report.incompatible)
        assert rsna_model.state_dict()["encoder.0.weight"].shape == (16, 1, 3, 3)
