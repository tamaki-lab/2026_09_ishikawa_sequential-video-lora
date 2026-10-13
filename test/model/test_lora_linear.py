"""Unit checks for the standalone LoRA residual."""

import pytest
import torch
from torch import nn

from model.vit.lora_linear import LoRALinear


def test_zero_b_starts_at_exact_base_output_and_keeps_base_frozen():
    torch.manual_seed(0)
    base = nn.Linear(4, 5)
    x = torch.randn(3, 4)
    baseline = base(x).detach().clone()
    base_before = {name: p.detach().clone() for name, p in base.named_parameters()}

    layer = LoRALinear(base, rank=2, alpha=2.0, dropout=0.0)
    torch.testing.assert_close(layer(x), baseline, rtol=0, atol=0)
    assert all(not p.requires_grad for p in layer.base.parameters())
    assert layer.lora_a.weight.requires_grad and layer.lora_b.weight.requires_grad
    assert torch.count_nonzero(layer.lora_b.weight) == 0

    optimizer = torch.optim.AdamW((p for p in layer.parameters() if p.requires_grad), lr=1e-3, weight_decay=0)
    loss = layer(x).square().mean()
    loss.backward()
    assert layer.lora_b.weight.grad is not None
    assert torch.count_nonzero(layer.lora_b.weight.grad) > 0
    assert layer.lora_a.weight.grad is not None
    assert torch.count_nonzero(layer.lora_a.weight.grad) == 0
    assert all(p.grad is None for p in layer.base.parameters())
    optimizer.step()
    assert torch.count_nonzero(layer.lora_b.weight) > 0

    optimizer.zero_grad(set_to_none=True)
    layer(x).square().mean().backward()
    assert torch.count_nonzero(layer.lora_a.weight.grad) > 0
    for name, p in base.named_parameters():
        assert torch.equal(p, base_before[name])


@pytest.mark.parametrize("rank, alpha, dropout", [
    (0, 8, 0), (-1, 8, 0), (8, 0, 0), (8, -1, 0),
    (8, 8, -0.1), (8, 8, 1.0),
])
def test_invalid_hyperparameters_rejected(rank, alpha, dropout):
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), rank, alpha, dropout)


def test_refuses_non_linear_base():
    with pytest.raises(TypeError):
        LoRALinear(nn.ReLU())
