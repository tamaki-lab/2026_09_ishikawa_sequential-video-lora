import pytest
import torch
from torch import nn

from model import LoRALinear


def make_base(in_features=12, out_features=10, dtype=torch.float32):
    torch.manual_seed(0)
    base = nn.Linear(in_features, out_features, dtype=dtype)
    base.requires_grad_(False)
    return base


def test_shapes_initialization_and_base_identity():
    base = make_base()
    weight, bias = base.weight.clone(), base.bias.clone()
    lora = LoRALinear(base, rank=4, alpha=8)

    assert lora.base is base
    assert lora.lora_a.weight.shape == (4, 12) and lora.lora_a.bias is None
    assert lora.lora_b.weight.shape == (10, 4) and lora.lora_b.bias is None
    assert torch.count_nonzero(lora.lora_b.weight) == 0
    assert torch.count_nonzero(lora.lora_a.weight) == lora.lora_a.weight.numel()
    assert lora.scaling == 2.0
    assert torch.equal(base.weight, weight) and torch.equal(base.bias, bias)
    assert not base.weight.requires_grad and not base.bias.requires_grad
    assert lora.lora_a.weight.requires_grad and lora.lora_b.weight.requires_grad


def test_initial_output_is_exactly_base_output():
    base = make_base()
    lora = LoRALinear(base, rank=4, alpha=8)
    x = torch.randn(3, 5, 12)
    assert torch.equal(lora(x), base(x))


def test_forward_matches_formula_and_keeps_base_weights():
    base = make_base()
    weight, bias = base.weight.clone(), base.bias.clone()
    lora = LoRALinear(base, rank=4, alpha=6)
    with torch.no_grad():
        lora.lora_b.weight.normal_()
    x = torch.randn(3, 12)
    expected = x @ weight.T + bias + (6 / 4) * (x @ lora.lora_a.weight.T @ lora.lora_b.weight.T)
    torch.testing.assert_close(lora(x), expected)
    assert torch.equal(base.weight, weight) and torch.equal(base.bias, bias)


def test_dropout_only_affects_lora_branch():
    base = make_base()
    lora = LoRALinear(base, rank=4, alpha=4, dropout=0.5).train()
    x = torch.randn(64, 12)
    assert torch.equal(lora(x), base(x))  # B == 0: dropout cannot leak into the base path
    with torch.no_grad():
        lora.lora_b.weight.normal_()
    lora.eval()
    expected = base(x) + lora.lora_b(lora.lora_a(x))
    torch.testing.assert_close(lora(x), expected)


def test_lora_follows_base_dtype():
    lora = LoRALinear(make_base(dtype=torch.float64), rank=2, alpha=2)
    assert lora.lora_a.weight.dtype == torch.float64
    assert lora.lora_b.weight.dtype == torch.float64


@pytest.mark.parametrize("kwargs", [
    {"rank": 0}, {"rank": -1}, {"rank": 1.5}, {"rank": True},
    {"alpha": 0}, {"alpha": -1.0}, {"alpha": float("nan")},
    {"dropout": -0.1}, {"dropout": 1.0},
])
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        LoRALinear(make_base(), **{"rank": 4, "alpha": 8, "dropout": 0.0, **kwargs})


@pytest.mark.parametrize("base", [nn.Identity(), nn.Conv2d(3, 3, 1)])
def test_non_linear_base_is_rejected(base):
    with pytest.raises(TypeError):
        LoRALinear(base, rank=4, alpha=8)


def test_double_wrap_is_rejected():
    lora = LoRALinear(make_base(), rank=4, alpha=8)
    with pytest.raises(TypeError, match="already wrapped"):
        LoRALinear(lora, rank=4, alpha=8)
