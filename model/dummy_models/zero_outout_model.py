from typing import Optional

import torch
from torch import nn

from model import ModelConfig, ClassificationBaseModel, ModelOutput


class ZeroOutputModel(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        self.model = nn.Linear(
            1,
            self.model_config.n_classes
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        dummy_logits = torch.zeros(
            (
                batch_size,
                self.model_config.n_classes
            ),
            device=device
        )

        if labels is None:
            return ModelOutput(
                logits=dummy_logits
            )

        dummy_loss = torch.tensor(
            0.0,
            requires_grad=True,
            device=device
        )

        return ModelOutput(
            logits=dummy_logits,
            loss=dummy_loss
        )
