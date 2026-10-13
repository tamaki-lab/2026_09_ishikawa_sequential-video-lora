"""Check that standalone adapter and classifier serialization is exact."""

from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.smoke.smoke_vit_lora_cifar10 import (
    load_adapter_head_checkpoint, save_adapter_head_checkpoint,
)


class DummyAdapter:
    def __init__(self):
        self.weight = torch.tensor([1.0, 2.0])

    def lora_state_dict(self):
        return {"layer.lora_a.weight": self.weight.clone()}

    def load_lora_state_dict(self, state):
        if set(state) != {"layer.lora_a.weight"} or state["layer.lora_a.weight"].shape != (2,):
            raise ValueError("invalid adapter")
        self.weight = state["layer.lora_a.weight"].clone()


def test_adapter_and_head_roundtrip(tmp_path: Path):
    adapter = DummyAdapter()
    head = nn.Linear(2, 10)
    original_adapter = adapter.lora_state_dict()
    original_head = {key: value.clone() for key, value in head.state_dict().items()}
    metadata = {
        "checkpoint_id": "test",
        "enabled": True,
        "target_modules": ["q", "v"],
        "rank": 8,
        "alpha": 8.0,
        "dropout": 0.0,
        "num_labels": 10,
    }
    output = tmp_path / "adapter_head.pt"
    save_adapter_head_checkpoint(output, adapter, head, metadata)
    assert output.exists()

    adapter.weight.fill_(9)
    with torch.no_grad():
        head.weight.fill_(10)
    with pytest.raises(ValueError, match="metadata"):
        load_adapter_head_checkpoint(output, adapter, head, {**metadata, "target_modules": ["k", "v"]})
    load_adapter_head_checkpoint(output, adapter, head, metadata)

    torch.testing.assert_close(adapter.lora_state_dict()["layer.lora_a.weight"],
                               original_adapter["layer.lora_a.weight"], rtol=0, atol=0)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, original_head[key], rtol=0, atol=0)
