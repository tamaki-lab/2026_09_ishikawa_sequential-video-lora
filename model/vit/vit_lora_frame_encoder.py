from typing import Dict, Iterator, List, Sequence, Tuple

import torch
from torch import nn
from transformers import ViTModel

from .lora_linear import LoRALinear


# target_modules name -> attribute of transformers ViTAttention
PROJECTIONS = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}


def canonical_targets(target_modules: Sequence[str]) -> Tuple[str, ...]:
    """Validate Q/K/V target names and return them in q, k, v order."""
    if isinstance(target_modules, str):
        raise TypeError(f"target_modules must be a list such as [q, v], got {target_modules!r}")
    targets = list(target_modules)
    if not targets:
        raise ValueError("target_modules must not be empty when LoRA is enabled")
    unknown = [target for target in targets if target not in PROJECTIONS]
    if unknown:
        raise ValueError(f"Unknown target_modules {unknown}; allowed: {list(PROJECTIONS)}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"Duplicate target_modules: {targets}")
    return tuple(target for target in PROJECTIONS if target in targets)


def inject_lora(
    vit: ViTModel,
    targets: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    """Wrap the selected Q/K/V projections of every block with LoRALinear.

    All targets are checked before any module is replaced, so a failure leaves
    the model unchanged. Returns the replaced module paths relative to vit.
    """
    hidden_size = vit.config.hidden_size
    if len(vit.layers) != vit.config.num_hidden_layers:
        raise RuntimeError(
            f"Expected {vit.config.num_hidden_layers} ViT layers, got {len(vit.layers)}")

    replacements = []
    for index, layer in enumerate(vit.layers):
        attention = layer.attention
        for target in targets:
            name = PROJECTIONS[target]
            path = f"layers.{index}.attention.{name}"
            projection = getattr(attention, name, None)
            if isinstance(projection, LoRALinear):
                raise RuntimeError(f"LoRA is already injected into {path}")
            if type(projection) is not nn.Linear:
                raise TypeError(f"Expected nn.Linear at {path}, got {type(projection).__name__}")
            if (projection.in_features, projection.out_features) != (hidden_size, hidden_size):
                raise RuntimeError(
                    f"Unexpected shape at {path}: {projection.in_features}->{projection.out_features}, "
                    f"expected {hidden_size}->{hidden_size}")
            replacements.append((attention, name, path, projection))

    for attention, name, _, projection in replacements:
        setattr(attention, name, LoRALinear(projection, rank, alpha, dropout))
    return [path for _, _, path, _ in replacements]


class ViTLoRAFrameEncoder(nn.Module):
    """Frozen pretrained ViT with optional Q/K/V LoRA; returns the CLS feature.

    Same interface as ViTFrameEncoder: pixel_values [B,3,224,224] -> CLS [B,768].
    The unused pooler is not built, so every base weight comes from the
    checkpoint. Order: load -> freeze the whole base -> inject LoRA (A/B stay
    trainable). With enabled=False no LoRA module is created.
    """

    def __init__(
        self,
        checkpoint_id: str = "google/vit-base-patch16-224",
        enabled: bool = True,
        target_modules: Sequence[str] = ("q", "v"),
        rank: int = 8,
        alpha: float = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vit, self.loading_info = ViTModel.from_pretrained(
            checkpoint_id, add_pooling_layer=False, output_loading_info=True)
        for key in ("missing_keys", "mismatched_keys", "error_msgs"):
            if self.loading_info[key]:
                raise RuntimeError(f"Pretrained ViT loading reported {key}: {self.loading_info[key]}")
        self.vit.requires_grad_(False)

        self.target_modules = canonical_targets(target_modules) if enabled else ()
        self.lora_paths = (
            inject_lora(self.vit, self.target_modules, rank, alpha, dropout) if enabled else [])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]

    def named_lora_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, parameter in self.named_parameters():
            if ".lora_a." in name or ".lora_b." in name:
                yield name, parameter

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.named_lora_parameters()}

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load LoRA A/B weights saved by lora_state_dict(); keys must match exactly."""
        expected = {name for name, _ in self.named_lora_parameters()}
        if set(state_dict) != expected:
            missing = sorted(expected - set(state_dict))
            unexpected = sorted(set(state_dict) - expected)
            raise KeyError(f"LoRA state_dict mismatch: missing={missing}, unexpected={unexpected}")
        self.load_state_dict(state_dict, strict=False)
