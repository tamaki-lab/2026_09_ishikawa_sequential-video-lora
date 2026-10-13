"""A minimal, inspectable LoRA residual around an existing linear layer."""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen pretrained Linear plus (alpha / rank) * B(A(dropout(x)))."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires an existing nn.Linear")
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False).to(base.weight)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False).to(base.weight)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(self.dropout(x)))
