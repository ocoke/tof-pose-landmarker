"""Masked heatmap and coordinate objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_argmax_2d(logits: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    """Decode differentiable ``(x, y)`` coordinates from heatmap logits."""

    batch, keypoints, height, width = logits.shape
    probabilities = torch.softmax(logits.reshape(batch, keypoints, -1) * beta, dim=-1).reshape(
        batch, keypoints, height, width
    )
    x_axis = torch.arange(width, device=logits.device, dtype=logits.dtype).view(1, 1, 1, width)
    y_axis = torch.arange(height, device=logits.device, dtype=logits.dtype).view(1, 1, height, 1)
    x_coord = (probabilities * x_axis).sum(dim=(2, 3))
    y_coord = (probabilities * y_axis).sum(dim=(2, 3))
    return torch.stack([x_coord, y_coord], dim=-1)


def masked_heatmap_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    """Binary cross-entropy over only valid keypoint heatmaps."""

    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor([positive_weight], device=logits.device, dtype=logits.dtype),
    )
    weights = valid_mask.unsqueeze(-1).unsqueeze(-1).to(logits.dtype)
    denominator = torch.clamp(weights.sum() * logits.shape[-1] * logits.shape[-2], min=1.0)
    return (loss * weights).sum() / denominator


def masked_coordinate_loss(predicted: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 coordinate loss over valid keypoints."""

    per_keypoint = F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
    weights = valid_mask.to(predicted.dtype)
    return (per_keypoint * weights).sum() / torch.clamp(weights.sum(), min=1.0)
