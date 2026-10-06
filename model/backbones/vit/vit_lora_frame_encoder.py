"""ViT CLS frame features with trainable Q/V LoRA adapters."""

import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model
from torch import nn
from transformers import ViTModel

from utils.configuration import load_config_group


_CANONICAL = load_config_group('encoder', 'vit_base_patch16_224')
CHECKPOINT_ID = _CANONICAL['checkpoint_id']
FEATURE_SIZE = _CANONICAL['feature_size']
IMAGE_SIZE = _CANONICAL['image_size']
CHANNELS = _CANONICAL['channels']
LORA_CONFIG = _CANONICAL['lora']


class ViTLoRAFrameEncoder(nn.Module):
    """Freeze the base ViT and train rank-8 LoRA on query/value projections."""

    def __init__(
        self, checkpoint_id: str = CHECKPOINT_ID, *, feature_size: int = FEATURE_SIZE,
        image_size: int = IMAGE_SIZE, channels: int = CHANNELS, lora_config=None,
    ):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (feature_size, image_size, channels)):
            raise ValueError('feature_size, image_size and channels must be positive integers')
        vit = ViTModel.from_pretrained(checkpoint_id, add_pooling_layer=False)
        actual_size = getattr(getattr(vit, 'config', None), 'hidden_size', feature_size)
        if actual_size != feature_size:
            raise ValueError(f'Configured feature_size {feature_size} differs from encoder hidden_size {actual_size}')
        self.feature_size = feature_size
        self.image_size = image_size
        self.channels = channels
        vit.requires_grad_(False)
        values = LORA_CONFIG if lora_config is None else lora_config
        if not isinstance(values, dict):
            values = {
                'target_modules': list(values.target_modules), 'r': values.r,
                'lora_alpha': values.lora_alpha, 'lora_dropout': values.lora_dropout, 'bias': values.bias,
            }
        config = PeftLoraConfig(
            target_modules=list(values['target_modules']), r=values['r'], lora_alpha=values['lora_alpha'],
            lora_dropout=values['lora_dropout'], bias=values['bias'],
        )
        self.vit = get_peft_model(vit, config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
