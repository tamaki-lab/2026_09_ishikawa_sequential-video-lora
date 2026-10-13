"""Q/K/V injection, frozen-base and adapter checkpoint tests without downloads."""

from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from model.vit.lora_linear import LoRALinear
from model.vit.vit_lora_frame_encoder import (
    ViTLoRAFrameEncoder, inject_attention_lora, validate_targets,
)


class FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = nn.Module()
        self.attention.q_proj = nn.Linear(d, d)
        self.attention.k_proj = nn.Linear(d, d)
        self.attention.v_proj = nn.Linear(d, d)


class FakeViT(nn.Module):
    def __init__(self, d=8, blocks=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d, num_hidden_layers=blocks)
        self.layers = nn.ModuleList(FakeBlock(d) for _ in range(blocks))

    def forward(self, pixel_values):
        # Small mock with all Q/K/V projections participating in autograd.
        x = pixel_values.mean((2, 3)).mean(1, keepdim=True).expand(-1, self.config.hidden_size)
        for block in self.layers:
            attn = block.attention
            x = x + .01 * (attn.q_proj(x) + attn.k_proj(x) + attn.v_proj(x))
        return SimpleNamespace(last_hidden_state=x[:, None, :])


def build_encoder(targets=("q", "v"), enabled=True):
    with patch("model.vit.vit_lora_frame_encoder.ViTModel.from_pretrained",
               return_value=FakeViT()) as loader:
        model = ViTLoRAFrameEncoder(
            "test-vit", enabled=enabled, target_modules=targets, rank=2, alpha=2, dropout=0
        )
    loader.assert_called_once_with("test-vit", add_pooling_layer=False)
    return model


ALL_TARGETS = [combo for size in range(1, 4) for combo in combinations("qkv", size)]


@pytest.mark.parametrize("targets", ALL_TARGETS)
def test_all_seven_target_combinations(targets):
    torch.manual_seed(0)
    model = build_encoder(targets)
    wrapped = [(name, layer) for name, layer in model.vit.named_modules()
               if isinstance(layer, LoRALinear)]
    assert len(wrapped) == 12 * len(targets)
    for name, layer in wrapped:
        assert name.rsplit(".", 1)[-1] == {"q": "q_proj", "k": "k_proj", "v": "v_proj"}[name.rsplit(".", 1)[-1][0]]
        assert layer.base.in_features == layer.base.out_features == 8
        assert layer.lora_a.weight.shape == (2, 8)
        assert layer.lora_b.weight.shape == (8, 2)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 12 * len(targets) * (2 * 8 + 8 * 2)
    x = torch.ones(2, 3, 8, 8)
    features = model(x)
    assert features.shape == (2, 8)
    assert torch.isfinite(features).all()
    features.square().mean().backward()
    assert all(p.grad is None for p in model.vit.parameters()
               if not p.requires_grad)
    assert any(layer.lora_b.weight.grad is not None and
               torch.count_nonzero(layer.lora_b.weight.grad) > 0
               for _, layer in wrapped)


def test_initial_baseline_equivalence_and_off_mode():
    torch.manual_seed(0)
    base = FakeViT()
    original = {name: p.detach().clone() for name, p in base.named_parameters()}
    x = torch.randn(2, 3, 8, 8)
    expected = base(x).last_hidden_state[:, 0, :]
    base.requires_grad_(False)
    inject_attention_lora(base, ("q", "v"), rank=2, alpha=2, dropout=0)
    torch.testing.assert_close(base(x).last_hidden_state[:, 0, :], expected, rtol=0, atol=0)
    for name, p in base.named_parameters():
        if name.endswith(".base.weight") or name.endswith(".base.bias"):
            original_key = name.replace(".base.", ".")
            assert torch.equal(p, original[original_key])
    off = build_encoder(enabled=False)
    assert all(not p.requires_grad for p in off.parameters())
    assert sum(isinstance(m, LoRALinear) for m in off.modules()) == 0


def test_second_step_updates_a_and_adapter_checkpoint_roundtrip():
    torch.manual_seed(7)
    model = build_encoder(("k", "v"))
    image = torch.randn(2, 3, 8, 8)
    base_before = {n: p.detach().clone() for n, p in model.vit.named_parameters()
                   if not p.requires_grad}
    initial = model.lora_state_dict()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-2, weight_decay=0)
    for _ in range(2):
        optimizer.zero_grad()
        model(image).square().mean().backward()
        optimizer.step()
    changed = model.lora_state_dict()
    assert any(not torch.equal(initial[k], changed[k]) for k in changed if ".lora_b." in k)
    assert any(not torch.equal(initial[k], changed[k]) for k in changed if ".lora_a." in k)
    for name, parameter in model.vit.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, base_before[name])
    model.eval()
    expected = model(image).detach().clone()
    model.load_lora_state_dict(initial)
    model.load_lora_state_dict(changed)
    torch.testing.assert_close(model(image), expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model.load_lora_state_dict({})
    corrupted = changed.copy()
    first = next(iter(corrupted))
    corrupted[first] = torch.randn(3, 3)
    with pytest.raises(ValueError):
        model.load_lora_state_dict(corrupted)


@pytest.mark.parametrize("targets", [
    [], ["q", "q"], ["w"], ["q", "a"], "q", ["Q"]
])
def test_invalid_targets(targets):
    with pytest.raises(ValueError):
        validate_targets(targets)


def test_failed_injection_does_not_mutate_other_modules():
    model = FakeViT()
    model.layers[2].attention.k_proj = nn.ReLU()
    with pytest.raises(RuntimeError):
        inject_attention_lora(model, ("q", "k"), rank=2, alpha=2)
    assert all(not isinstance(block.attention.q_proj, LoRALinear) for block in model.layers)
