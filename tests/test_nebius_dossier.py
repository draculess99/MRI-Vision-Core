import cv2
import numpy as np
import pytest

from hackathon.nebius_nvidia.evidence_dossier import (
    ApprovalState,
    EvidenceStatus,
    build_dossier_from_images,
)


def _disc(radius: int, size: int = 128) -> np.ndarray:
    img = np.full((size, size), 20, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, 220, -1)
    return img


def test_dossier_builds_from_synthetic_image():
    d = build_dossier_from_images(_disc(30))
    out = d.to_dict()
    assert d.quality.width > 0 and d.quality.height > 0
    assert d.changeguard is None
    assert d.requires_human_approval
    assert d.approval.state is ApprovalState.PENDING
    assert "Non-diagnostic" in out["notice"]
    assert any("No ChangeGuard" in n for n in out["review_notes"])


def test_identical_baseline_is_consistent_but_still_needs_human():
    d = build_dossier_from_images(_disc(30), baseline_image=_disc(30))
    assert d.changeguard.status is EvidenceStatus.CONSISTENT
    assert d.status is EvidenceStatus.CONSISTENT
    assert d.requires_human_approval
    assert d.approval.state is ApprovalState.PENDING


def test_large_change_triggers_review_and_requires_approval():
    d = build_dossier_from_images(_disc(50), baseline_image=_disc(15))
    assert d.changeguard.status is EvidenceStatus.REVIEW
    assert d.changeguard.deltas["foreground_percentage"] > 5
    assert d.status is EvidenceStatus.REVIEW
    assert d.requires_human_approval
    assert d.approval.state is ApprovalState.PENDING


def test_flat_image_is_uncertain_and_requires_approval():
    d = build_dossier_from_images(np.full((128, 128), 100, dtype=np.uint8))
    assert "low_contrast" in d.quality.flags
    assert d.status is EvidenceStatus.UNCERTAIN
    assert d.requires_human_approval


def test_human_decision_requires_named_reviewer():
    d = build_dossier_from_images(_disc(50), baseline_image=_disc(15))
    with pytest.raises(ValueError):
        d.decide(True, reviewer="  ")
    decided = d.decide(True, reviewer="Dr. Example", comment="checked source")
    assert decided.approval.state is ApprovalState.APPROVED
    assert decided.requires_human_approval
    assert d.approval.state is ApprovalState.PENDING
