"""Encode valid SequentialSample frames without dataset-specific setup."""

import sequential_loader as sl
import torch


FEATURE_SIZE = 768


def encode_chunk(sample: sl.SequentialSample, processor, encoder, device: torch.device):
    """Encode valid frames in source order and restore their chunk positions."""
    valid_mask = sample.valid_mask
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        raise RuntimeError("SequentialSample has no valid frames")

    valid_frames = sample.frames[valid_mask]
    pixel_values = processor(images=valid_frames, return_tensors="pt")["pixel_values"]
    if tuple(pixel_values.shape) != (valid_count, 3, 224, 224):
        raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

    valid_features = encoder(pixel_values.to(device))
    if tuple(valid_features.shape) != (valid_count, FEATURE_SIZE):
        raise RuntimeError(f"Unexpected valid feature shape: {tuple(valid_features.shape)}")
    if not torch.isfinite(valid_features).all().item():
        raise RuntimeError("Valid feature contains NaN or Inf")

    frame_features = torch.zeros(
        (sample.frames.shape[0], FEATURE_SIZE),
        dtype=valid_features.dtype,
        device=valid_features.device,
    )
    device_mask = valid_mask.to(frame_features.device)
    frame_features[device_mask] = valid_features
    if not torch.all(frame_features[~device_mask] == 0).item():
        raise RuntimeError("Padding feature rows are not zero")

    return pixel_values, valid_features, frame_features
