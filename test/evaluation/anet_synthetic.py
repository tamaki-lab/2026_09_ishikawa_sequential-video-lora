"""Synthetic ActivityNet-like timestamps and a deterministic frame encoder."""

import torch
from torch import nn


def chunk_times(chunk_count, offset=0.):
    """Chunk c holds 16 frames at offset + 1.6c + 0.1k; the last chunk has 3 frames."""
    return [[round(offset + 1.6 * chunk + 0.1 * frame, 6) for frame in range(3 if chunk == chunk_count - 1 else 16)]
            for chunk in range(chunk_count)]


class ColorEncoder(nn.Module):
    """Deterministic [N, 768] features from the processed pixel colours."""

    def __init__(self, scale=1.):
        super().__init__()
        self.projection = nn.Linear(3, 768, bias=True)
        torch.manual_seed(0)
        nn.init.normal_(self.projection.weight)
        nn.init.normal_(self.projection.bias)
        self.projection.weight.data.mul_(scale)

    def forward(self, pixels):
        return self.projection(pixels[:, :, 0, 0] / 255)


class RecordingProcessor:
    """Processor stand-in: per-frame colour expanded to [N, 3, 224, 224]."""

    def __call__(self, *, images, return_tensors):
        assert return_tensors == 'pt'
        return {'pixel_values': images[:, :, :1, :1].float().expand(-1, 3, 224, 224).contiguous()}
