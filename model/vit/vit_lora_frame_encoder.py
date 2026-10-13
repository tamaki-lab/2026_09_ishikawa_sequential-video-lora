"""Frozen image-pretrained ViT with independently selectable Q/K/V LoRA."""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from transformers import ViTModel

from .lora_linear import LoRALinear


_PROJECTIONS = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}


def validate_targets(target_modules: Sequence[str]) -> tuple[str, ...]:
    """Reject unknown, repeated or empty targets; preserve the YAML order."""
    if isinstance(target_modules, str):
        raise ValueError("target_modules must be a sequence of q, k, v")
    result = tuple(target_modules)
    if not result or any(not isinstance(name, str) or name not in _PROJECTIONS for name in result):
        raise ValueError("target_modules must be a non-empty selection from q, k, v")
    if len(set(result)) != len(result):
        raise ValueError("target_modules must not contain duplicates")
    return result


def inject_attention_lora(
    vit: nn.Module, target_modules: Sequence[str], rank: int = 8,
    alpha: float = 8.0, dropout: float = 0.0,
) -> tuple[str, ...]:
    """Wrap selected attention projections, checking *all* targets before mutation."""
    selected = validate_targets(target_modules)
    if type(rank) is not int or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    layers = getattr(vit, "layers", None)
    config = getattr(vit, "config", None)
    if layers is None or config is None or len(layers) != config.num_hidden_layers:
        raise RuntimeError("Expected ViTModel.layers with config.num_hidden_layers blocks")

    expected_dim = config.hidden_size
    slots = []
    for index, block in enumerate(layers):
        attention = getattr(block, "attention", None)
        if attention is None:
            raise RuntimeError(f"Missing attention in ViT block {index}")
        for name in selected:
            attr = _PROJECTIONS[name]
            linear = getattr(attention, attr, None)
            if not isinstance(linear, nn.Linear) or isinstance(linear, LoRALinear):
                raise RuntimeError(f"Expected unwrapped nn.Linear at layers.{index}.attention.{attr}")
            if (linear.in_features, linear.out_features) != (expected_dim, expected_dim):
                raise RuntimeError(f"Unexpected projection shape at layers.{index}.attention.{attr}")
            slots.append((attention, attr, linear))

    for attention, attr, linear in slots:
        setattr(attention, attr, LoRALinear(linear, rank, alpha, dropout))
    return selected


class ViTLoRAFrameEncoder(nn.Module):
    """Return CLS features [batch, hidden_size]; only configured adapters train."""

    def __init__(
        self, checkpoint_id: str = "google/vit-base-patch16-224",
        enabled: bool = True, target_modules: Sequence[str] = ("q", "v"),
        rank: int = 8, alpha: float = 8.0, dropout: float = 0.0,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(checkpoint_id, add_pooling_layer=False)
        self.vit.requires_grad_(False)
        self.target_modules = validate_targets(target_modules)
        self.enabled = enabled
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if enabled:
            inject_attention_lora(self.vit, self.target_modules, rank, alpha, dropout)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Store only adapter weights; the pretrained ViT is restored separately."""
        state = {}
        for name, module in self.vit.named_modules():
            if isinstance(module, LoRALinear):
                state[f"{name}.lora_a.weight"] = module.lora_a.weight.detach().cpu().clone()
                state[f"{name}.lora_b.weight"] = module.lora_b.weight.detach().cpu().clone()
        return state

    def load_lora_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load with exact key and shape checks; never silently skip adapters."""
        expected = {}
        for name, module in self.vit.named_modules():
            if isinstance(module, LoRALinear):
                expected[f"{name}.lora_a.weight"] = module.lora_a.weight
                expected[f"{name}.lora_b.weight"] = module.lora_b.weight
        if set(state) != set(expected):
            raise ValueError(f"LoRA state keys differ: missing={sorted(set(expected)-set(state))}, "
                             f"unexpected={sorted(set(state)-set(expected))}")
        for key, parameter in expected.items():
            if not isinstance(state[key], torch.Tensor) or state[key].shape != parameter.shape:
                raise ValueError(f"LoRA state shape mismatch: {key}")
        with torch.no_grad():
            for key, parameter in expected.items():
                parameter.copy_(state[key].to(device=parameter.device, dtype=parameter.dtype))
