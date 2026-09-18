from torch import nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

from model import ModelConfig, ClassificationBaseModel


class ResNet18(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.prepare_model()
        self.replace_pretrained_head()

    def prepare_model(self):
        self.model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
            if self.model_config.use_pretrained else None)

    def replace_pretrained_head(self):
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.model_config.n_classes
        )
