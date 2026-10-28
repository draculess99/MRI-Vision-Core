"""ConvNeXt-T multi-plane MIL models with optional anatomical ROI box pooling (Phase 19A; research prototype, own implementation).

A  ConvNeXtMIL(roi_targets=())          plain: per-slice ConvNeXt-T features (global average pool), plane embeddings, target-specific gated attention over slices and planes.
B  ConvNeXtMIL(roi_targets=(...))       same, but for the chosen targets each slice embedding is pooled INSIDE that target's anatomical box (learned in-box spatial attention),
                                        and only slices inside the box's z-range take part in that target's aggregation. Boxes are inputs: they must come from a detector or
                                        another independent source with recorded provenance; ground-truth boxes are refused unless an oracle study is requested explicitly.

The backbone is a from-scratch implementation of ConvNeXt-T (Liu et al., 2022: depths 3-3-9-3, widths 96-192-384-768, 7x7 depthwise blocks, layer scale 1e-6) whose state-dict keys follow
torchvision's convnext_tiny so a locally stored ImageNet checkpoint can be loaded; none is bundled or downloaded. The aggregation and ROI pooling are our reading of the public description of
the 1st-place solution, not its code (see docs/phase19_convnext_reproduction.md).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .rsna_knee_dataset import TARGET_COLUMNS
from .weak_label_models import GatedAttention

TINY = {"depths": (3, 3, 9, 3), "dims": (96, 192, 384, 768)}
TRUNK_PARAMETERS = 27_820_128  # torchvision convnext_tiny (28,589,128) minus its 1000-class linear head (769,000)
MEAN, STD = 0.45, 0.225


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class _Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class CNBlock(nn.Module):
    def __init__(self, dim: int, layer_scale: float = 1e-6, drop_path: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(dim, dim, 7, padding=3, groups=dim), _Permute((0, 2, 3, 1)), nn.LayerNorm(dim, eps=1e-6),
                                   nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), _Permute((0, 3, 1, 2)))
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.drop_path = drop_path

    def forward(self, x):
        y = self.layer_scale * self.block(x)
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            y = y * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep).to(y.dtype) / keep
        return x + y


class ConvNeXtTrunk(nn.Module):
    """Stem + four stages; returns the stride-32 feature map [N, dims[-1], H/32, W/32]."""

    def __init__(self, depths: Sequence[int] = TINY["depths"], dims: Sequence[int] = TINY["dims"], drop_path: float = 0.0):
        super().__init__()
        layers = [nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm2d(dims[0], eps=1e-6))]
        total, index = sum(depths), 0
        for stage, (depth, dim) in enumerate(zip(depths, dims)):
            blocks = []
            for _ in range(depth):
                blocks.append(CNBlock(dim, drop_path=drop_path * index / max(total - 1, 1)))
                index += 1
            layers.append(nn.Sequential(*blocks))
            if stage < len(dims) - 1:
                layers.append(nn.Sequential(LayerNorm2d(dim, eps=1e-6), nn.Conv2d(dim, dims[stage + 1], 2, stride=2)))
        self.features = nn.Sequential(*layers)
        self.out_norm = LayerNorm2d(dims[-1], eps=1e-6)  # == torchvision classifier.0
        self.dims = tuple(dims)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.features(x)


def load_trunk_weights(trunk: ConvNeXtTrunk, path: Path) -> dict:
    """Load a LOCAL torchvision-style convnext_tiny state dict (features.* and classifier.0.*); the classifier head is ignored. Refuses mismatched shapes."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = state.get("model", state) if isinstance(state, dict) else state
    renamed = {("out_norm." + k[len("classifier.0."):]) if k.startswith("classifier.0.") else k: v for k, v in state.items() if k.startswith(("features.", "classifier.0."))}
    expected = trunk.state_dict()
    missing = sorted(set(expected) - set(renamed))
    bad = sorted(k for k in renamed if k not in expected or renamed[k].shape != expected[k].shape)
    if missing or bad:
        raise ValueError(f"checkpoint does not match this ConvNeXt trunk (missing {len(missing)}, mismatched {len(bad)})")
    trunk.load_state_dict(renamed, strict=True)
    return {"loaded_tensors": len(renamed)}


# ------------------------------------------------------------------------------------------------ ROI boxes
FORBIDDEN_SOURCES = ("ground_truth", "oracle", "annotation", "manual_for_evaluated_studies")


class RoiError(ValueError):
    pass


@dataclass
class RoiProvenance:
    source: str  # "detector" | "heuristic_prior" | "agent_labels" | "oracle" ...
    description: str = ""
    detector_sha256: Optional[str] = None
    trained_on: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def independent(self) -> bool:
        return self.source not in FORBIDDEN_SOURCES


FULL_BOX = (0.0, 0.0, 1.0, 1.0, 0.0, 1.0)  # x0, y0, x1, y1 (fractions of the image), z0, z1 (fractions of the series' slices)


def validate_boxes(boxes: torch.Tensor) -> None:
    """boxes [..., 6] with 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1, 0 <= z0 <= z1 <= 1."""
    b = boxes.reshape(-1, 6)
    ok = (b[:, 0] >= 0) & (b[:, 0] < b[:, 2]) & (b[:, 2] <= 1) & (b[:, 1] >= 0) & (b[:, 1] < b[:, 3]) & (b[:, 3] <= 1) & (b[:, 4] >= 0) & (b[:, 4] <= b[:, 5]) & (b[:, 5] <= 1)
    if not bool(ok.all()):
        raise RoiError("ROI boxes must satisfy 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1, 0 <= z0 <= z1 <= 1")


def box_pool_grid(features: torch.Tensor, boxes: torch.Tensor, grid: int = 4) -> torch.Tensor:
    """Bilinear sample a grid x grid lattice inside each box of its own feature map. features [N, C, h, w], boxes [N, >=4] (x0, y0, x1, y1 fractions) -> [N, C, grid, grid]."""
    t = (torch.arange(grid, device=features.device, dtype=features.dtype) + 0.5) / grid
    xs = boxes[:, 0:1] + (boxes[:, 2:3] - boxes[:, 0:1]) * t  # [N, grid]
    ys = boxes[:, 1:2] + (boxes[:, 3:4] - boxes[:, 1:2]) * t
    gx = (xs * 2 - 1).unsqueeze(1).expand(-1, grid, -1)
    gy = (ys * 2 - 1).unsqueeze(2).expand(-1, -1, grid)
    return F.grid_sample(features, torch.stack([gx, gy], dim=-1), mode="bilinear", padding_mode="border", align_corners=False)


# ------------------------------------------------------------------------------------------------ model
class ConvNeXtMIL(nn.Module):
    def __init__(self, num_planes: int = 3, num_targets: int = len(TARGET_COLUMNS), roi_targets: Sequence[int] = (), depths: Sequence[int] = TINY["depths"],
                 dims: Sequence[int] = TINY["dims"], dropout: float = 0.2, drop_path: float = 0.1, grid: int = 4, pretrained_path: Optional[Path] = None,
                 strict_rois: bool = False):
        super().__init__()
        self.trunk = ConvNeXtTrunk(depths, dims, drop_path)
        if pretrained_path is not None:
            load_trunk_weights(self.trunk, pretrained_path)
        f = dims[-1]
        self.num_planes, self.num_targets, self.grid, self.strict_rois = num_planes, num_targets, grid, strict_rois
        self.roi_targets = tuple(int(t) for t in roi_targets)
        self.plane_embedding = nn.Parameter(torch.zeros(num_planes, f))
        self.norm = nn.LayerNorm(f)
        self.attention = nn.ModuleList(GatedAttention(f) for _ in range(num_targets))
        self.heads = nn.ModuleList(nn.Linear(f, 1) for _ in range(num_targets))
        self.dropout = nn.Dropout(dropout)
        if self.roi_targets:
            self.spatial_attention = nn.Conv2d(f, len(self.roi_targets), 1)  # one in-box score map per ROI target

    def forward(self, images: torch.Tensor, mask: torch.Tensor, rois: Optional[torch.Tensor] = None, roi_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """images [B, P, S, 1, H, W], mask [B, P, S] -> logits [B, T]. rois [B, P, R, 6] / roi_valid [B, P, R] feed only the ROI targets (R = len(roi_targets))."""
        b, p, s = mask.shape
        x = images[mask][:, :1]  # valid slices only [N, 1, H, W]
        x = ((x - MEAN) / STD).expand(-1, 3, -1, -1)
        fmap = self.trunk(x)  # [N, F, h, w]
        pooled = self.trunk.out_norm(fmap).mean(dim=(2, 3))  # [N, F]
        shared = pooled.new_zeros(b, p, s, pooled.shape[-1])
        shared[mask] = pooled
        shared = self.norm(shared + self.plane_embedding[None, :, None, :])
        roi_embed, roi_ok = None, None
        if self.roi_targets:
            r = len(self.roi_targets)
            if rois is None:
                if self.strict_rois:
                    raise RoiError("this model requires ROI boxes (strict_rois=True)")
                rois = fmap.new_tensor(FULL_BOX).expand(b, p, r, 6)
                roi_valid = torch.zeros(b, p, r, dtype=torch.bool, device=fmap.device)
            elif roi_valid is None:
                roi_valid = torch.ones(b, p, r, dtype=torch.bool, device=fmap.device)
            validate_boxes(rois)
            boxes = torch.where(roi_valid.unsqueeze(-1), rois.to(fmap.dtype), fmap.new_tensor(FULL_BOX))  # [B, P, R, 6]
            bi, pi, si = mask.nonzero(as_tuple=True)
            counts = mask.sum(dim=2)  # valid slices per (b, p)
            position = si.to(fmap.dtype) / (counts[bi, pi] - 1).clamp(min=1).to(fmap.dtype)  # slice position in [0, 1] among the valid slices of its series
            n_valid = fmap.shape[0]
            roi_embed = fmap.new_zeros(b, p, s, r, pooled.shape[-1])
            roi_ok = torch.zeros(b, p, s, r, dtype=torch.bool, device=fmap.device)
            score_maps = self.spatial_attention(fmap)  # [N, R, h, w]
            for k in range(r):
                box = boxes[bi, pi, k]  # [N, 6]
                cells = box_pool_grid(fmap, box, self.grid)  # [N, F, g, g]
                scores = box_pool_grid(score_maps[:, k:k + 1], box, self.grid).flatten(1)  # [N, g*g]
                weights = scores.softmax(dim=1).view(n_valid, 1, self.grid, self.grid)
                emb = self.trunk.out_norm(cells).mul(weights).sum(dim=(2, 3))  # in-box attention pooling [N, F]
                roi_embed[bi, pi, si, k] = emb
                roi_ok[bi, pi, si, k] = (position >= box[:, 4]) & (position <= box[:, 5])
            roi_embed = self.norm(roi_embed + self.plane_embedding[None, :, None, None, :])
        logits = []
        for t in range(self.num_targets):
            if t in self.roi_targets:
                k = self.roi_targets.index(t)
                emb, valid = roi_embed[:, :, :, k], mask & roi_ok[:, :, :, k]
                empty = ~valid.flatten(1).any(dim=1)
                valid = torch.where(empty.view(b, 1, 1), mask, valid)  # a box that covers no sampled slice falls back to every valid slice
            else:
                emb, valid = shared, mask
            pooled_t, _ = self.attention[t](emb.flatten(1, 2), valid.flatten(1))
            logits.append(self.heads[t](self.dropout(pooled_t)).squeeze(-1))
        return torch.stack(logits, dim=1)


def build_convnext_model(arm: str, config: Mapping) -> nn.Module:
    """arm 'D' = plain ConvNeXt-T, 'E' = ConvNeXt-T with ROI box pooling on config['roi_targets'] (target names, default ACL and Medial Meniscus)."""
    planes = len(config.get("planes", ("Axial", "Coronal", "Sagittal")))
    roi_targets = ()
    if arm == "E":
        names = config.get("roi_targets") or ("ACL", "Medial Meniscus")
        roi_targets = tuple(TARGET_COLUMNS.index(n) for n in names)
    elif arm != "D":
        raise ValueError(f"unknown ConvNeXt arm {arm!r}")
    path = config.get("pretrained_weights")
    return ConvNeXtMIL(planes, len(TARGET_COLUMNS), roi_targets, config.get("convnext_depths", TINY["depths"]), config.get("convnext_dims", TINY["dims"]),
                       float(config.get("dropout", 0.2)), float(config.get("drop_path", 0.1)), int(config.get("roi_grid", 4)), Path(path) if path else None,
                       bool(config.get("strict_rois", False)))


# ------------------------------------------------------------------------------------------------ ROI tables and dataset wrapper
def load_roi_table(path: Path, *, allow_oracle: bool = False) -> tuple[dict, RoiProvenance]:
    """Read {'schema': 'roi-table/1', 'provenance': {...}, 'boxes': {study: {plane: {target: [x0,y0,x1,y1,z0,z1]}}}}. Ground-truth / oracle boxes are refused unless allow_oracle."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "roi-table/1":
        raise RoiError("unsupported ROI table schema")
    if not data.get("provenance", {}).get("source"):
        raise RoiError("ROI table has no provenance source")
    prov = RoiProvenance(**{k: v for k, v in data["provenance"].items() if k in RoiProvenance.__dataclass_fields__})
    if not prov.independent and not allow_oracle:
        raise RoiError(f"ROI source {prov.source!r} is not independent of the labels; an oracle upper bound must be requested explicitly")
    for study, planes in data["boxes"].items():
        for plane, targets in planes.items():
            for target, box in targets.items():
                if target not in TARGET_COLUMNS:
                    raise RoiError(f"unknown target {target!r}")
                validate_boxes(torch.tensor(box, dtype=torch.float32).view(1, 6))
    return data["boxes"], prov


def roi_tensors(boxes: Mapping, study_uids: Sequence[str], planes: Sequence[str], roi_target_names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """rois [B, P, R, 6] and roi_valid [B, P, R]; a missing box is marked invalid (the model then pools over the whole image)."""
    rois = torch.tensor(FULL_BOX).expand(len(study_uids), len(planes), len(roi_target_names), 6).clone()
    valid = torch.zeros(len(study_uids), len(planes), len(roi_target_names), dtype=torch.bool)
    for i, uid in enumerate(study_uids):
        for j, plane in enumerate(planes):
            for k, name in enumerate(roi_target_names):
                box = boxes.get(str(uid), {}).get(plane, {}).get(name)
                if box is not None:
                    rois[i, j, k], valid[i, j, k] = torch.tensor(box, dtype=torch.float32), True
    return rois, valid


class RoiDataset(torch.utils.data.Dataset):
    """Wraps a weak-label dataset and adds the study's ROI boxes ('rois', 'roi_valid') to every sample."""

    def __init__(self, base, boxes: Mapping, planes: Sequence[str], roi_target_names: Sequence[str]):
        self.base, self.boxes, self.planes, self.names = base, boxes, list(planes), list(roi_target_names)
        for attribute in ("values", "masks", "ids"):
            if hasattr(base, attribute):
                setattr(self, attribute, getattr(base, attribute))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        rois, valid = roi_tensors(self.boxes, [item["study_uid"]], self.planes, self.names)
        return {**item, "rois": rois[0], "roi_valid": valid[0]}
