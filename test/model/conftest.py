import copy

import pytest
import torch
from transformers import ViTConfig, ViTModel


TINY_VIT = dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
    image_size=32, patch_size=8, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
)


@pytest.fixture
def fake_pretrained_vit(monkeypatch):
    """Serve deep copies of one fixed random ViT through ViTModel.from_pretrained.

    Every load returns identical base weights, like loading the same checkpoint.
    """
    def install(loading_info=None, **config_overrides):
        torch.manual_seed(0)
        source = ViTModel(ViTConfig(**{**TINY_VIT, **config_overrides}), add_pooling_layer=False).eval()
        info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": [],
                **(loading_info or {})}
        calls = []

        def from_pretrained(checkpoint_id, **kwargs):
            calls.append((checkpoint_id, kwargs))
            return copy.deepcopy(source), copy.deepcopy(info)

        monkeypatch.setattr("model.vit.vit_lora_frame_encoder.ViTModel.from_pretrained", from_pretrained)
        return source, calls
    return install
