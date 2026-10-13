import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Add a low-rank update to an existing nn.Linear.

    output = base(x) + (alpha / rank) * lora_b(lora_a(dropout(x)))

    The given base layer object is kept as-is (its weight and bias are never
    modified). A is randomly initialized and B is zero-initialized, so the
    wrapped layer initially returns exactly base(x). Freezing the base is the
    caller's responsibility; only lora_a / lora_b are created here.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if isinstance(base, LoRALinear):
            raise TypeError("base is already wrapped by LoRALinear")
        if type(base) is not nn.Linear:
            raise TypeError(f"base must be nn.Linear, got {type(base).__name__}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank!r}")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not alpha > 0:
            raise ValueError(f"alpha must be > 0, got {alpha!r}")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError(f"dropout must satisfy 0 <= dropout < 1, got {dropout!r}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, **factory_kwargs)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, **factory_kwargs)
        # A: random kaiming-uniform (the nn.Linear / PEFT default), B: zero.
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(self.lora_dropout(x)))
