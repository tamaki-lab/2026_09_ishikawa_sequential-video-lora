"""ViT CLS frame features with trainable Q/V LoRA adapters."""

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import ViTModel


class ViTLoRAFrameEncoder(nn.Module):
    """Freeze the base ViT and train rank-8 LoRA on query/value projections."""

    def __init__(self, checkpoint_id: str = "google/vit-base-patch16-224"):
        super().__init__()
        vit = ViTModel.from_pretrained(checkpoint_id, add_pooling_layer=False)
        vit.requires_grad_(False)
        config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            r=8,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
        )
        self.vit = get_peft_model(vit, config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
