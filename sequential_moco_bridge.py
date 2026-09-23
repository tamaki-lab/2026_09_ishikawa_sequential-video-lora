"""Deterministic same-clip views using the existing valid-frame ViT bridge."""

from dataclasses import replace

import torch

from model.aggregators import MaskedMeanClipAggregator
from sequential_vit_bridge import encode_chunk


def make_two_views(sample):
    """Keep Query raw; flip every valid Key frame along width, without mutation."""
    key_frames = sample.frames.clone()
    key_frames[sample.valid_mask] = sample.frames[sample.valid_mask].flip(-1)
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
