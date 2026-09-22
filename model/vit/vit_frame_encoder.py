import torch
from torch import nn
from transformers import ViTModel


class ViTFrameEncoder(nn.Module):
    """Extract a CLS feature from each preprocessed RGB frame."""

    def __init__(self, checkpoint_id: str = "google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(checkpoint_id)
        self.vit.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
