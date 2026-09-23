"""Order-invariant clip features from valid frame features."""

import torch
from torch import nn


class MaskedMeanClipAggregator(nn.Module):
    """Average valid rows of [T, D], preserving dtype, device and gradients.

    The boolean [T] mask may be on CPU or the feature device. Invalid rows
    are excluded before reduction; they contribute neither values nor count.
    """

    def forward(
        self, frame_features: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if frame_features.ndim != 2:
            raise ValueError("frame_features must have shape [T, D]")
        if valid_mask.ndim != 1:
            raise ValueError("valid_mask must have shape [T]")
        if frame_features.shape[0] != valid_mask.shape[0]:
            raise ValueError("frame_features and valid_mask must have the same T")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        if not valid_mask.any().item():
            raise ValueError("valid_mask must contain at least one valid frame")
        return frame_features[valid_mask.to(frame_features.device)].mean(dim=0)
