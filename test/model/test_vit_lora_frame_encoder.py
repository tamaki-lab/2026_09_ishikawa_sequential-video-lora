from unittest.mock import Mock

import pytest
import torch
from peft.tuners.lora import LoraLayer
from transformers import ViTConfig, ViTModel

from model import ViTLoRAFrameEncoder


@pytest.fixture
def encoder(monkeypatch):
    # Real 12-layer, width-768 attention and PEFT; smaller MLP for offline unit tests.
    backbone = ViTModel(ViTConfig(intermediate_size=32), add_pooling_layer=False)
    load = Mock(return_value=backbone)
    monkeypatch.setattr('model.vit.vit_lora_frame_encoder.ViTModel.from_pretrained', load)
    model = ViTLoRAFrameEncoder()
    load.assert_called_once_with('google/vit-base-patch16-224', add_pooling_layer=False)
    return model


def test_real_peft_targets_config_and_frozen_base(encoder):
    assert encoder.vit.get_base_model().pooler is None
    targets = {name: module for name, module in encoder.vit.named_modules() if isinstance(module, LoraLayer)}
    assert len(targets) == 24
    assert sum(name.endswith('.q_proj') for name in targets) == 12
    assert sum(name.endswith('.v_proj') for name in targets) == 12
    config = encoder.vit.peft_config['default']
    assert config.target_modules == {'q_proj', 'v_proj'}
    assert (config.r, config.lora_alpha, config.lora_dropout, config.bias) == (8, 8, 0.0, 'none')
    assert all(module.scaling['default'] == 1.0 for module in targets.values())
    trainable = {name: p for name, p in encoder.named_parameters() if p.requires_grad}
    expected = {
        f'vit.{name}.lora_{factor}.default.weight'
        for name in targets for factor in ('A', 'B')
    }
    assert set(trainable) == expected
    assert len(trainable) == 48
    assert sum(p.numel() for p in trainable.values()) == 294_912
    assert all(not p.requires_grad for name, p in encoder.named_parameters() if name not in expected)


@pytest.mark.parametrize('training', [True, False])
def test_cls_forward_shape_finite_and_gradient_mode(encoder, training):
    encoder.train(training)
    pixels = torch.randn(2, 3, 224, 224)
    features = encoder(pixels)
    assert features.shape == (2, 768)
    assert torch.isfinite(features).all()
    assert features.requires_grad
    assert encoder.vit.training == training
    with torch.no_grad():
        expected = encoder.vit(pixel_values=pixels).last_hidden_state[:, 0, :]
    torch.testing.assert_close(features, expected)
