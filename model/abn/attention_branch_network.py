import copy
from typing import Optional

import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d

from model import ModelConfig, ClassificationBaseModel, ModelOutput


class ABNResNet50(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.prepare_model()

    def prepare_model(self):
        self.model = ABNResNet50Model(
            self.model_config.n_classes,
            self.model_config.use_pretrained
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:

        perception_branch_logits, attention_branch_logits = \
            self.model(pixel_values)

        if labels is None:
            return ModelOutput(
                logits=perception_branch_logits
            )

        perception_branch_loss = self.criterion(perception_branch_logits, labels)
        attention_branch_loss = self.criterion(attention_branch_logits, labels)
        loss = perception_branch_loss + attention_branch_loss

        return ModelOutput(
            logits=perception_branch_logits,
            loss=loss
        )


class ABNResNet50Model(nn.Module):
    """Attention Branch Network: Learning of Attention Mechanism for Visual Explanation, CVPR2019
    https://openaccess.thecvf.com/content_CVPR_2019/html/Fukui_Attention_Branch_Network_Learning_of_Attention_Mechanism_for_Visual_Explanation_CVPR_2019_paper.html
    """

    def __init__(self, n_classes: int, use_pretrained: bool):
        super().__init__()

        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar100_resnet56",
            pretrained=use_pretrained,
        )

        (
            self.feature_extractor,
            self.perception_branch,
            extractor_num_features
        ) = self.__split_r50_into_half(n_classes)

        (
            self.attention_branch1,
            self.attention_branch2,
            self.attention_branch3,
        ) = self.__create_attention_branch(extractor_num_features, n_classes)

        self.attention_map = None

    def __create_attention_branch(self, num_features, n_classes):
        attention_branch1 = nn.Sequential(
            # reusing layer3 block structure
            #     (except the first block with downsampling)
            copy.deepcopy(self.model.layer3[1:]),

            nn.BatchNorm2d(num_features),
            nn.Conv2d(
                num_features,
                n_classes,
                kernel_size=1
            ),
            nn.ReLU(inplace=True)
        )
        attention_branch2 = nn.Sequential(
            nn.Conv2d(
                n_classes,
                1,
                kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        attention_branch3 = nn.Sequential(
            nn.Conv2d(
                n_classes,
                n_classes,
                kernel_size=1),
        )
        return attention_branch1, attention_branch2, attention_branch3

    def __split_r50_into_half(self, n_classes: int):
        resnet50_bottom = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
        )
        r50b_num_features = self.__get_last_num_features(resnet50_bottom)

        head = nn.Linear(
            self.model.fc.in_features,
            n_classes
        )

        resnet50_top = nn.Sequential(
            self.model.avgpool,
            nn.Flatten(),
            head
        )

        return resnet50_bottom, resnet50_top, r50b_num_features

    def __get_last_num_features(self, module: nn.Module):
        return list(module.modules())[-1].num_features

    def get_attention_map(self):
        return self.attention_map

    def __gap(self, x: torch.Tensor):
        """Global Average Pooling (GAP)"""
        x = adaptive_avg_pool2d(x, (1, 1))  # BCHW -> BC11
        x = x.squeeze((-2, -1))  # BC11 -> BC
        return x

    def attention_branch(self, x1):
        ax1 = self.attention_branch1(x1)

        attention_map = self.attention_branch2(ax1)

        ax2 = self.attention_branch3(ax1)  # B,C,H,W
        logits = self.__gap(ax2)  # B,C,1,1 -> B,C

        return attention_map, logits

    def forward(self, x):
        x1 = self.feature_extractor(x)
        self.attention_map, attention_branch_logits = self.attention_branch(x1)

        x2 = x1 * self.attention_map
        perception_branch_logits = self.perception_branch(x2)

        return (
            perception_branch_logits,
            attention_branch_logits,
        )
