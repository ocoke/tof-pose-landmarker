from __future__ import annotations

import pytest


def test_unet_forward_shapes() -> None:
    torch = pytest.importorskip("torch")
    from macchiato.models.macchiato_pose import MacchiatoPoseUNet

    model = MacchiatoPoseUNet(input_channels=3, num_keypoints=17, width_multiplier=1.15)
    inputs = torch.randn(1, 3, 240, 240)
    heatmaps, visibility = model(inputs, return_aux=True)
    assert tuple(heatmaps.shape) == (1, 17, 240, 240)
    assert tuple(visibility.shape) == (1, 17)
