from typing import Optional

import lightning.pytorch as pl
import torch
from transformers import ViTMAEForPreTraining


DEFAULT_CHECKPOINT_ID = "facebook/vit-mae-large"


class MAELightningModule(pl.LightningModule):

    def __init__(
            self,
            checkpoint_id: str = DEFAULT_CHECKPOINT_ID,
    ):
        super().__init__()
        self.checkpoint_id = checkpoint_id
        self.model = ViTMAEForPreTraining.from_pretrained(
            self.checkpoint_id,
        )

    def forward(
            self,
            pixel_values: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
    ):
        return self.model(
            pixel_values=pixel_values,
            noise=noise,
        )
