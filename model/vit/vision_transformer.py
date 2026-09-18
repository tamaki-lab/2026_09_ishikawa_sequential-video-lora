from typing import Optional

import torch

from transformers import ViTForImageClassification

from model import ModelConfig, ClassificationBaseModel, ModelOutput


class ViTb(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.prepare_model()

    def prepare_model(self):
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=self.model_config.n_classes,
            ignore_mismatched_sizes=True
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:

        output = self.model(
            pixel_values=pixel_values,
            labels=labels,
        )

        return ModelOutput(
            logits=output.logits,
            loss=output.loss
        )
