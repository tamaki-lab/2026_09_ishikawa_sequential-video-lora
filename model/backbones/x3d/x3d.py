import torch
from torch import nn

from model.model_config import ModelConfig
from model.base_model import ClassificationBaseModel


class X3DM(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.prepare_model()
        self.replace_pretrained_head()

    def prepare_model(self):
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=self.model_config.use_pretrained,
            head_activation=None,  # removing nn.Softmax
        )

    def replace_pretrained_head(self):
        in_features = self.model.blocks[5].proj.in_features
        self.model.blocks[5].proj = nn.Linear(
            in_features,
            self.model_config.n_classes
        )
