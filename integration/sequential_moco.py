"""Deterministic same-clip views using the existing valid-frame ViT bridge."""

from dataclasses import replace

import torch

from model.aggregators import MaskedMeanClipAggregator
from .sequential_vit import encode_chunk


def make_two_views(sample, key_transform='horizontal_flip'):
    """Keep Query raw; transform valid Key frames without changing the source."""
    if key_transform not in ('horizontal_flip', 'gbr_horizontal_flip'):
        raise ValueError(f'Unknown key_transform: {key_transform}')
    key_frames = sample.frames.clone()
    valid_frames = sample.frames[sample.valid_mask]
    if key_transform == 'gbr_horizontal_flip':
        valid_frames = valid_frames[:, [1, 2, 0]]
    key_frames[sample.valid_mask] = valid_frames.flip(-1)
    return sample, replace(sample, frames=key_frames)


def encode_query_view(sample, processor, moco, device):
    pixels, _, frames = encode_chunk(sample, processor, moco.query_encoder, device)
    clip = MaskedMeanClipAggregator()(frames, sample.valid_mask)
    return pixels, frames, clip, moco.project_query(clip)


@torch.no_grad()
def encode_key_view(sample, processor, moco, device):
    pixels, _, frames = encode_chunk(sample, processor, moco.key_encoder, device)
    clip = MaskedMeanClipAggregator()(frames, sample.valid_mask)
    return pixels, frames, clip, moco.project_key(clip)
