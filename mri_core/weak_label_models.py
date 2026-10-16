"""Model arms for the weak-label experiment. All arms share one interface:

    forward(images [B, planes, slices, 1, H, W], mask [B, planes, slices] bool) -> logits [B, 12]

  A  RSNAKneeCNN (unchanged, imported from rsna_knee_model): max-pool over slices.
  B  RSNAKneeAttnCNN: the same 4-layer conv encoder as A with gated-attention pooling over slices per plane.
  C  RSNAKneePretrainedMIL: ResNet-18 backbone (torchvision-compatible parameter names, so ImageNet weights
     attached as a Kaggle input load without torchvision), 2.5D context (previous/current/next slice as three
     channels), gated Attention-MIL over every valid (plane, slice) instance.

Padded slices never enter an encoder and receive -inf attention, so a study's output does not depend on how many
slices it was padded with. Nothing here touches the network: pretrained weights are only ever read from a path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .rsna_knee_model import RSNA_TARGET_COUNT, RSNAKneeCNN

ARMS = ("A", "B", "C")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _check_inputs(images: torch.Tensor, mask: torch.Tensor, num_planes: int) -> None:
    if images.ndim != 6 or images.shape[1] != num_planes or images.shape[3] != 1:
        raise ValueError("Expected [batch, planes, slices, 1, height, width]")
    if mask.shape != images.shape[:3] or mask.dtype != torch.bool or not mask.any(dim=2).all():
        raise ValueError("Each exam plane requires at least one valid slice")


class GatedAttention(nn.Module):
    """Gated attention-MIL pooling (Ilse et al.): weights = softmax(w^T(tanh(Vh) * sigmoid(Uh))) over valid instances."""

    def __init__(self, features: int, hidden: int = 64):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(features, hidden), nn.Linear(features, hidden), nn.Linear(hidden, 1)

    def forward(self, embeddings: torch.Tensor, valid: torch.Tensor):
        """embeddings [..., N, F], valid [..., N] bool -> (pooled [..., F], weights [..., N])."""
        scores = self.w(torch.tanh(self.v(embeddings)) * torch.sigmoid(self.u(embeddings))).squeeze(-1)
        weights = scores.masked_fill(~valid, -torch.inf).softmax(dim=-1)
        return (weights.unsqueeze(-1) * embeddings).sum(dim=-2), weights


def _scatter_valid(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = embeddings.new_zeros((*mask.shape, embeddings.shape[-1]))
    out[mask] = embeddings
    return out


class RSNAKneeAttnCNN(nn.Module):
    def __init__(self, num_planes: int = 3, dropout: float = 0.2, num_targets: int = RSNA_TARGET_COUNT):
        super().__init__()
        if num_planes not in (1, 2, 3):
            raise ValueError("Expected one to three planes")
        self.num_planes, self.num_targets = num_planes, num_targets
        blocks, previous = [], 1
        for channels in (16, 32, 64, 128):
            blocks.extend([nn.Conv2d(previous, channels, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(8, channels), nn.ReLU()])
            previous = channels
        self.encoder = nn.Sequential(*blocks)
        self.attention = GatedAttention(128)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128 * num_planes, num_targets))

    def forward(self, images, mask):
        _check_inputs(images, mask, self.num_planes)
        embeddings = _scatter_valid(self.encoder(images[mask]).mean(dim=(-2, -1)), mask)  # [B, P, S, 128]
        pooled, _ = self.attention(embeddings, mask)  # per plane
        return self.head(pooled.flatten(start_dim=1))


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + identity)


class ResNet18Backbone(nn.Module):
    """ResNet-18 trunk with torchvision's parameter names (conv1, bn1, layer1..4, fc) and no classifier use."""

    def __init__(self, width: int = 64):
        super().__init__()
        self.inplanes = width
        self.conv1 = nn.Conv2d(3, width, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._layer(width, 2, 1)
        self.layer2 = self._layer(width * 2, 2, 2)
        self.layer3 = self._layer(width * 4, 2, 2)
        self.layer4 = self._layer(width * 8, 2, 2)
        self.out_features = width * 8
        self.fc = nn.Linear(self.out_features, 1000)  # present only so ImageNet checkpoints load strictly; unused

    def _layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [_BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x.mean(dim=(-2, -1))


def load_pretrained_backbone(backbone: ResNet18Backbone, path: Path) -> dict:
    """Load a torchvision-style ResNet-18 state dict from a local file (a Kaggle input). Never downloads."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found at {path}; attach them as a Kaggle input")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    state = {k.removeprefix("module."): v for k, v in state.items()}
    own = backbone.state_dict()
    missing = [k for k in own if k not in state and not k.startswith("fc.")]
    mismatched = [k for k in own if k in state and state[k].shape != own[k].shape and not k.startswith("fc.")]
    if missing or mismatched:
        raise ValueError(f"Pretrained weights do not match ResNet-18: missing={missing[:3]} mismatched={mismatched[:3]}")
    backbone.load_state_dict({k: v for k, v in state.items() if k in own and not k.startswith("fc.")}, strict=False)
    return {"loaded": len(own) - len([k for k in own if k.startswith("fc.")]), "source": str(path)}


class RSNAKneePretrainedMIL(nn.Module):
    def __init__(self, num_planes: int = 3, dropout: float = 0.2, num_targets: int = RSNA_TARGET_COUNT,
                 pretrained_path: Optional[Path] = None, freeze_bn: bool = True, input_size: Optional[int] = None,
                 sequence: str = "none"):
        super().__init__()
        if num_planes not in (1, 2, 3):
            raise ValueError("Expected one to three planes")
        self.num_planes, self.num_targets, self.freeze_bn, self.input_size = num_planes, num_targets, freeze_bn, input_size
        if sequence not in ("none", "gru"):
            raise ValueError("sequence must be 'none' or 'gru'")
        self.sequence = sequence
        self.backbone = ResNet18Backbone()
        self.backbone.fc.requires_grad_(False)  # kept only so ImageNet checkpoints load strictly; never used
        self.pretrained_info = load_pretrained_backbone(self.backbone, pretrained_path) if pretrained_path else None
        features = self.backbone.out_features
        self.plane_embedding = nn.Parameter(torch.zeros(num_planes, features))
        self.attention = GatedAttention(features)
        self.sequence_model = nn.GRU(features, features // 2, batch_first=True, bidirectional=True) if sequence == "gru" else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_targets))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn:  # small, padded batches make BatchNorm statistics unreliable
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    @staticmethod
    def context_stack(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B,P,S,1,H,W] -> [B,P,S,3,H,W] with (previous, current, next) valid slices as channels (edges repeat)."""
        _, _, slices = mask.shape
        count = mask.sum(dim=2, keepdim=True)  # [B,P,1]
        index = torch.arange(slices, device=images.device).view(1, 1, -1)
        prev = (index - 1).clamp(min=0)
        nxt = torch.minimum(index + 1, count - 1).clamp(min=0)
        gather = lambda i: torch.gather(images.squeeze(3), 2, i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1,
                                        images.shape[-2], images.shape[-1]))
        return torch.stack([gather(prev.expand_as(mask)), images.squeeze(3), gather(nxt.expand_as(mask))], dim=3)

    def forward(self, images, mask):
        _check_inputs(images, mask, self.num_planes)
        stacked = self.context_stack(images, mask)[mask]  # [N, 3, H, W], valid slices only
        if self.input_size:
            stacked = nn.functional.interpolate(stacked, size=(self.input_size, self.input_size), mode="bilinear",
                                                align_corners=False)
        embeddings = self.backbone((stacked - self.mean) / self.std)
        grid = _scatter_valid(embeddings, mask)
        if self.sequence_model is not None:  # bidirectional GRU over each plane's slice order; packed, so padding never enters it
            batch, planes, slices, features = grid.shape
            lengths = mask.sum(dim=2).reshape(-1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(grid.reshape(batch * planes, slices, features), lengths,
                                                       batch_first=True, enforce_sorted=False)
            out, _ = self.sequence_model(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=slices)
            grid = grid + out.to(grid.dtype).reshape(batch, planes, slices, features)  # residual
        grid = grid + self.plane_embedding.view(1, self.num_planes, 1, -1)
        flat_valid = mask.flatten(start_dim=1)
        pooled, _ = self.attention(grid.flatten(start_dim=1, end_dim=2), flat_valid)
        return self.head(pooled)


def build_model(arm: str, config: dict) -> nn.Module:
    planes = len(config.get("planes", ("Axial", "Coronal", "Sagittal")))
    dropout = float(config.get("dropout", 0.2))
    if arm == "A":
        return RSNAKneeCNN(planes, dropout)
    if arm == "B":
        return RSNAKneeAttnCNN(planes, dropout)
    if arm == "C":
        path = config.get("pretrained_weights")
        return RSNAKneePretrainedMIL(planes, dropout, pretrained_path=Path(path) if path else None,
                                     freeze_bn=bool(config.get("freeze_bn", True)), input_size=config.get("encoder_input_size"),
                                     sequence=config.get("sequence", "none"))
    raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
