import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from model import ViTFrameEncoder


class FakeViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        cls = torch.full((batch_size, 768), 3.0) + self.weight
        patch = torch.full((batch_size, 768), 7.0) + self.weight
        return SimpleNamespace(last_hidden_state=torch.stack((cls, patch), dim=1))


class TestViTFrameEncoder(unittest.TestCase):
    def test_cls_feature_and_frozen_backbone(self):
        with patch('model.vit.vit_frame_encoder.ViTModel.from_pretrained', return_value=FakeViT()) as load:
            encoder = ViTFrameEncoder()

        load.assert_called_once_with('google/vit-base-patch16-224', add_pooling_layer=False)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.vit.parameters()))
        features = encoder(torch.zeros(2, 3, 224, 224))
        self.assertEqual(tuple(features.shape), (2, 768))
        self.assertTrue(torch.all(features == 4.0).item())
        self.assertTrue(torch.isfinite(features).all().item())
        encoder.eval()
        self.assertFalse(encoder.vit.training)
        encoder.train()
        self.assertTrue(encoder.vit.training)


if __name__ == '__main__':
    unittest.main()
