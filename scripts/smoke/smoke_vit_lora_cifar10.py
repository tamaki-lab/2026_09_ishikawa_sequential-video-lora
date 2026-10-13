"""Short, supervised CIFAR-10 sanity check for standalone Q/K/V LoRA.

Run (after caching CIFAR-10 and ViT weights):
    python -m scripts.smoke.smoke_vit_lora_cifar10
Override targets without code changes:
    python -m scripts.smoke.smoke_vit_lora_cifar10 'lora.target_modules=[k,v]'

This does not use Sequential Loader, MoCo, the existing main.py, or Comet.
"""

import json
import subprocess
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from transformers import AutoImageProcessor

from model.vit import ViTLoRAFrameEncoder


def _image_transform(processor):
    def convert(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
    return convert


def _resolved_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    raise ValueError(f"Unsupported or unavailable training.device: {choice}")


def save_adapter_head_checkpoint(path: Path, encoder, head, metadata: dict) -> None:
    """Never serialize the full frozen backbone as part of this pilot."""
    torch.save({
        "metadata": metadata,
        "adapter": encoder.lora_state_dict(),
        "classifier": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
    }, path)


def load_adapter_head_checkpoint(path: Path, encoder, head, metadata: dict) -> None:
    """Require exactly the matching checkpoint identity and LoRA selection."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["metadata"] != metadata:
        raise ValueError("Checkpoint metadata does not match the requested encoder/config")
    encoder.load_lora_state_dict(checkpoint["adapter"])
    head.load_state_dict(checkpoint["classifier"], strict=True)


def _git_provenance() -> dict:
    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        state = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout
        return {"commit": revision, "dirty": bool(state.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def run(cfg: DictConfig) -> dict:
    if cfg.data.name != "CIFAR10":
        raise ValueError("The standalone pilot supports CIFAR10 only")
    if cfg.training.optimizer != "adamw":
        raise ValueError("The standalone pilot supports AdamW only")
    if int(cfg.data.train_subset_size) < 1 or int(cfg.training.batch_size) < 1 or int(cfg.training.max_steps) < 1:
        raise ValueError("subset, batch_size and max_steps must be positive")
    if cfg.logging.comet_enabled:
        raise ValueError("Comet logging is not part of the standalone sanity check")

    torch.manual_seed(int(cfg.training.seed))
    device = _resolved_device(str(cfg.training.device))
    processor = AutoImageProcessor.from_pretrained(cfg.model.checkpoint_id)
    dataset = CIFAR10(
        root=cfg.data.root, train=True, download=bool(cfg.data.download),
        transform=_image_transform(processor),
    )
    if cfg.data.train_subset_size > len(dataset):
        raise ValueError("train_subset_size exceeds CIFAR-10 train size")
    generator = torch.Generator().manual_seed(int(cfg.training.seed))
    indices = torch.randperm(len(dataset), generator=generator)[:int(cfg.data.train_subset_size)].tolist()
    train_loader = DataLoader(
        Subset(dataset, indices), batch_size=int(cfg.training.batch_size),
        shuffle=True, generator=torch.Generator().manual_seed(int(cfg.training.seed)),
        drop_last=False, num_workers=0,
    )

    encoder = ViTLoRAFrameEncoder(
        checkpoint_id=cfg.model.checkpoint_id, enabled=bool(cfg.lora.enabled),
        target_modules=list(cfg.lora.target_modules), rank=int(cfg.lora.rank),
        alpha=float(cfg.lora.alpha), dropout=float(cfg.lora.dropout),
    ).to(device)
    head = nn.Linear(encoder.vit.config.hidden_size, 10).to(device)
    trainable = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=float(cfg.training.learning_rate), weight_decay=float(cfg.training.weight_decay),
    )
    head_before = {k: v.detach().clone() for k, v in head.state_dict().items()}
    adapter_before = encoder.lora_state_dict()
    base_before = {
        name: value.detach().cpu().clone()
        for name, value in encoder.vit.named_parameters() if not value.requires_grad
    }
    # All pretrained parameters remain frozen. The optimizer contains only adapters + head.
    if any(p.requires_grad for name, p in encoder.vit.named_parameters()
           if ".lora_a." not in name and ".lora_b." not in name):
        raise RuntimeError("Pretrained ViT weights must stay frozen")

    history = []
    stream = iter(train_loader)
    encoder.train()
    head.train()
    for step in range(1, int(cfg.training.max_steps) + 1):
        try:
            images, labels = next(stream)
        except StopIteration:
            stream = iter(train_loader)
            images, labels = next(stream)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = head(encoder(images))
        if logits.shape != (labels.shape[0], 10):
            raise RuntimeError(f"Unexpected classification logits: {logits.shape}")
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        adapter_grad = sum(
            float(p.grad.detach().norm().item()) for p in encoder.parameters()
            if p.requires_grad and p.grad is not None
        )
        head_grad = sum(float(p.grad.detach().norm().item()) for p in head.parameters()
                        if p.grad is not None)
        if not torch.isfinite(torch.tensor([adapter_grad, head_grad])).all():
            raise RuntimeError(f"Non-finite gradient at step {step}")
        if any(p.grad is not None for name, p in encoder.vit.named_parameters()
               if ".lora_a." not in name and ".lora_b." not in name):
            raise RuntimeError("Frozen ViT parameter received a gradient")
        optimizer.step()
        accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
        item = {"step": step, "loss": float(loss.detach().item()), "train_accuracy": accuracy,
                "adapter_grad_norm_sum": adapter_grad, "head_grad_norm_sum": head_grad}
        history.append(item)
        print(json.dumps(item))

    adapter_after = encoder.lora_state_dict()
    adapter_update = sum(float((v - adapter_before[k]).norm().item()) for k, v in adapter_after.items())
    head_update = sum(float((value.detach().cpu() - head_before[name].cpu()).norm().item())
                      for name, value in head.state_dict().items())
    if any(not torch.equal(parameter.detach().cpu(), base_before[name])
           for name, parameter in encoder.vit.named_parameters() if not parameter.requires_grad):
        raise RuntimeError("A frozen ViT parameter changed during training")
    if cfg.lora.enabled and adapter_update <= 0:
        raise RuntimeError("No adapter parameter changed during real-image training")
    if head_update <= 0:
        raise RuntimeError("No classifier parameter changed")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_id": str(cfg.model.checkpoint_id),
        "enabled": bool(cfg.lora.enabled),
        "target_modules": list(cfg.lora.target_modules),
        "rank": int(cfg.lora.rank),
        "alpha": float(cfg.lora.alpha),
        "dropout": float(cfg.lora.dropout),
        "num_labels": 10,
    }
    path = output_dir / "adapter_head.pt"
    save_adapter_head_checkpoint(path, encoder, head, metadata)

    # Round-trip the saved adapter + head in eval mode, without a second 86M-param ViT.
    encoder.eval()
    head.eval()
    with torch.no_grad():
        sample = images[:1]
        expected_logits = head(encoder(sample)).detach().clone()
    load_adapter_head_checkpoint(path, encoder, head, metadata)
    with torch.no_grad():
        torch.testing.assert_close(head(encoder(sample)), expected_logits, rtol=1e-5, atol=1e-6)

    report = {
        "model_checkpoint": str(cfg.model.checkpoint_id),
        "selected_targets": list(cfg.lora.target_modules) if cfg.lora.enabled else [],
        "dataset": "CIFAR10 train",
        "dataset_indices": indices,
        "device": str(device),
        "git": _git_provenance(),
        "steps": len(history),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "adapter_update_norm_sum": adapter_update,
        "head_update_norm_sum": head_update,
        "history": history,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    print(f"Saved: {path}, report.json, resolved_config.yaml")
    return report


@hydra.main(version_base="1.3", config_path="../../conf", config_name="lora_standalone")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
