"""Encode valid SequentialSample frames without dataset-specific setup."""

import sequential_loader as sl
import torch

from utils.configuration import load_config_group


_ENCODER_CONFIG = load_config_group('encoder', 'vit_base_patch16_224')
FEATURE_SIZE = _ENCODER_CONFIG['feature_size']
IMAGE_SIZE = _ENCODER_CONFIG['image_size']
CHANNELS = _ENCODER_CONFIG['channels']


def encode_chunk(
    sample: sl.SequentialSample, processor, encoder, device: torch.device, *,
    feature_size=None, image_size=None, channels=None,
):
    """Encode valid frames in source order and restore their chunk positions."""
    valid_mask = sample.valid_mask
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        raise RuntimeError("SequentialSample has no valid frames")

    # Read dimensions only when the encoder explicitly exposes them.  Objects
    # with a dynamic ``__getattr__`` (notably unittest.mock.Mock) otherwise
    # manufacture placeholder attributes and accidentally override the
    # canonical defaults.
    encoder_attributes = getattr(encoder, '__dict__', {})
    dimensions = {
        'feature_size': FEATURE_SIZE if feature_size is None else feature_size,
        'image_size': IMAGE_SIZE if image_size is None else image_size,
        'channels': CHANNELS if channels is None else channels,
    }
    for name, override in (
        ('feature_size', feature_size),
        ('image_size', image_size),
        ('channels', channels),
    ):
        if override is None and name in encoder_attributes:
            dimensions[name] = encoder_attributes[name]
        value = dimensions[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    feature_size = dimensions['feature_size']
    image_size = dimensions['image_size']
    channels = dimensions['channels']

    valid_frames = sample.frames[valid_mask]
    pixel_values = processor(images=valid_frames, return_tensors="pt")["pixel_values"]
    if tuple(pixel_values.shape) != (valid_count, channels, image_size, image_size):
        raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

    valid_features = encoder(pixel_values.to(device))
    if tuple(valid_features.shape) != (valid_count, feature_size):
        raise RuntimeError(f"Unexpected valid feature shape: {tuple(valid_features.shape)}")
    if not torch.isfinite(valid_features).all().item():
        raise RuntimeError("Valid feature contains NaN or Inf")

    frame_features = torch.zeros(
        (sample.frames.shape[0], feature_size),
        dtype=valid_features.dtype,
        device=valid_features.device,
    )
    device_mask = valid_mask.to(frame_features.device)
    frame_features[device_mask] = valid_features
    if not torch.all(frame_features[~device_mask] == 0).item():
        raise RuntimeError("Padding feature rows are not zero")

    return pixel_values, valid_features, frame_features
