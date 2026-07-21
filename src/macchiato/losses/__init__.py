"""Training objectives for Macchiato."""

from .heatmap_loss import masked_coordinate_loss, masked_heatmap_bce, soft_argmax_2d

__all__ = ["masked_coordinate_loss", "masked_heatmap_bce", "soft_argmax_2d"]
