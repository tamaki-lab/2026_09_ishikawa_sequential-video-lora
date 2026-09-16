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

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        outputs = self(pixel_values=pixel_values)
        loss = outputs.loss

        if loss is None:
            raise AssertionError("outputs.loss is None")
        if loss.ndim != 0:
            raise AssertionError(
                f"Expected scalar loss, got shape {tuple(loss.shape)}"
            )
        if not bool(torch.isfinite(loss).item()):
            raise AssertionError(f"Loss is not finite: {loss.item()}")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            batch_size=pixel_values.shape[0],
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
