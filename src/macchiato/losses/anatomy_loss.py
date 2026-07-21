"""Optional anatomical consistency objective for future Macchiato studies."""

from __future__ import annotations

import torch

COCO_BONES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def anatomy_length_loss(predicted: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Compare normalized COCO limb lengths where both endpoints are valid."""

    losses = []
    for start, end in COCO_BONES:
        mask = valid_mask[:, start] & valid_mask[:, end]
        if mask.any():
            pred_length = torch.linalg.vector_norm(predicted[mask, start] - predicted[mask, end], dim=-1)
            target_length = torch.linalg.vector_norm(target[mask, start] - target[mask, end], dim=-1)
            losses.append(torch.abs(pred_length - target_length).mean())
    return torch.stack(losses).mean() if losses else predicted.sum() * 0.0
