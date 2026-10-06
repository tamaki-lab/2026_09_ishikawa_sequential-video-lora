import torch
from torch import nn
from transformers import ViTModel

from utils.configuration import load_config_group


_CANONICAL = load_config_group('encoder', 'vit_base_patch16_224')
CHECKPOINT_ID = _CANONICAL['checkpoint_id']
FEATURE_SIZE = _CANONICAL['feature_size']
IMAGE_SIZE = _CANONICAL['image_size']
CHANNELS = _CANONICAL['channels']


class ViTFrameEncoder(nn.Module):
    """Extract a CLS feature from each preprocessed RGB frame."""

    def __init__(
        self, checkpoint_id: str = CHECKPOINT_ID, *, feature_size: int = FEATURE_SIZE,
        image_size: int = IMAGE_SIZE, channels: int = CHANNELS,
    ):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (feature_size, image_size, channels)):
            raise ValueError('feature_size, image_size and channels must be positive integers')
        self.vit = ViTModel.from_pretrained(checkpoint_id, add_pooling_layer=False)
        actual_size = getattr(getattr(self.vit, 'config', None), 'hidden_size', feature_size)
        if actual_size != feature_size:
            raise ValueError(f'Configured feature_size {feature_size} differs from encoder hidden_size {actual_size}')
        self.feature_size = feature_size
        self.image_size = image_size
        self.channels = channels
        self.vit.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
