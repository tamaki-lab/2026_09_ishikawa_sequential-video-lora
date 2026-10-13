"""Train frozen ViT + Q/K/V LoRA + a linear head on a fixed CIFAR-10 train subset.

Short supervised smoke for the standalone ViT LoRA verification spec (Stage B).
This is LoRA + classifier head training, not LoRA-only training. Loss decrease
and accuracy are logged as observations only, not as pass conditions.

Usage:
    python -m scripts.smoke.smoke_vit_lora_cifar10
    python -m scripts.smoke.smoke_vit_lora_cifar10 'lora.target_modules=[q,k,v]'
    python -m scripts.smoke.smoke_vit_lora_cifar10 --cfg job --resolve

CIFAR-10 is never downloaded unless data.download=true is given.
"""

import logging
import math
import subprocess
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import transformers
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from transformers import AutoImageProcessor

from logger import configure_logger
from model import LoRALinear, ViTLoRAFrameEncoder


FEATURE_SIZE = 768
NUM_CLASSES = 10
IMAGE_SIZE = 224
CHECKPOINT_FILENAME = "vit_lora_cifar10_adapter_head.pt"

log = logging.getLogger(__name__)


def git_output(repository: Path, *args: str) -> str:
    # rstrip only: the leading space of `git status --porcelain` is meaningful.
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).rstrip()


def checkpoint_preprocess(processor):
    """Resize and normalize exactly as the checkpoint's image processor (no augmentation)."""
    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]
    return preprocess


def total_norm(tensors) -> float:
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(tensor.detach()) for tensor in tensors])
    ).item()


def run(cfg: DictConfig, output_dir: Path) -> dict:
    if not cfg.lora.enabled:
        raise ValueError("This entrypoint trains LoRA + classifier head; lora.enabled=false is not supported")
    if cfg.training.optimizer != "adamw":
        raise ValueError(f"Only training.optimizer=adamw is supported, got {cfg.training.optimizer}")
    if cfg.data.name != "CIFAR10":
        raise ValueError(f"Only data.name=CIFAR10 is supported, got {cfg.data.name}")

    start_time = time.perf_counter()
    repository = Path(__file__).resolve().parents[2]
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    log.info("implementation branch: %s", git_output(repository, "branch", "--show-current"))
    log.info("implementation revision: %s", git_output(repository, "rev-parse", "HEAD"))
    log.info("implementation dirty files: %s", git_output(repository, "status", "--porcelain").splitlines())
    log.info("torch %s / transformers %s", torch.__version__, transformers.__version__)

    seed = cfg.training.seed
    batch_size = cfg.training.batch_size
    max_steps = cfg.training.max_steps
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("resolved device: %s", device)

    # Data: CIFAR-10 train split only, fixed seeded subset, checkpoint preprocessing.
    processor = AutoImageProcessor.from_pretrained(cfg.model.checkpoint_id)
    dataset = CIFAR10(
        root=cfg.data.root,
        train=True,
        download=cfg.data.download,
        transform=checkpoint_preprocess(processor))
    if not batch_size <= cfg.data.train_subset_size <= len(dataset):
        raise ValueError(
            f"Need batch_size <= train_subset_size <= {len(dataset)}, "
            f"got {batch_size} and {cfg.data.train_subset_size}")
    subset_indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[:cfg.data.train_subset_size].tolist()
    loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed))
    image_mean = [float(value) for value in processor.image_mean]
    image_std = [float(value) for value in processor.image_std]
    log.info("dataset: CIFAR10 train split, %d images, root=%s", len(dataset), cfg.data.root)
    log.info("subset indices (seed %d): %s", seed, subset_indices)
    log.info("preprocess: %s resize to %d, mean=%s std=%s, no augmentation",
             type(processor).__name__, IMAGE_SIZE, image_mean, image_std)

    # Model: frozen ViT + LoRA, then a new trainable head.
    encoder = ViTLoRAFrameEncoder(
        checkpoint_id=cfg.model.checkpoint_id,
        enabled=True,
        target_modules=list(cfg.lora.target_modules),
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout).to(device)
    head = nn.Linear(FEATURE_SIZE, NUM_CLASSES).to(device)

    vit_config = encoder.vit.config
    if vit_config.hidden_size != FEATURE_SIZE:
        raise RuntimeError(f"Expected hidden size {FEATURE_SIZE}, got {vit_config.hidden_size}")
    lora_parameters = dict(encoder.named_lora_parameters())
    base_parameters = {
        name: parameter for name, parameter in encoder.named_parameters() if name not in lora_parameters}
    wrapped = [name for name, module in encoder.named_modules() if isinstance(module, LoRALinear)]
    expected_modules = vit_config.num_hidden_layers * len(encoder.target_modules)
    expected_lora_numel = expected_modules * cfg.lora.rank * 2 * vit_config.hidden_size
    lora_numel = sum(parameter.numel() for parameter in lora_parameters.values())
    if (sorted(wrapped) != sorted(f"vit.{path}" for path in encoder.lora_paths)
            or len(wrapped) != expected_modules
            or len(lora_parameters) != 2 * expected_modules
            or lora_numel != expected_lora_numel):
        raise RuntimeError(
            f"LoRA injection audit failed: {len(wrapped)} modules / {len(lora_parameters)} tensors / "
            f"{lora_numel} parameters, expected {expected_modules} / {2 * expected_modules} / "
            f"{expected_lora_numel}")
    if any(parameter.requires_grad for parameter in base_parameters.values()):
        raise RuntimeError("Base ViT parameters must be frozen")
    if not all(parameter.requires_grad for parameter in lora_parameters.values()):
        raise RuntimeError("LoRA A/B parameters must be trainable")

    trainable = [*lora_parameters.values(), *head.parameters()]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    declared_ids = {
        id(parameter) for module in (encoder, head) for parameter in module.parameters() if parameter.requires_grad}
    if optimizer_ids != declared_ids or len(optimizer_ids) != len(trainable):
        raise RuntimeError("Optimizer parameters differ from the declared trainable set")

    head_numel = sum(parameter.numel() for parameter in head.parameters())
    log.info("checkpoint ID: %s", cfg.model.checkpoint_id)
    log.info("pretrained loading report: missing=%s mismatched=%s unexpected=%s",
             encoder.loading_info["missing_keys"], encoder.loading_info["mismatched_keys"],
             encoder.loading_info["unexpected_keys"])
    log.info("ViT layers: %d, hidden size: %d", vit_config.num_hidden_layers, vit_config.hidden_size)
    log.info("target modules: %s -> %d wrapped projections", list(encoder.target_modules), len(wrapped))
    log.info("LoRA rank=%s alpha=%s dropout=%s scaling=%s",
             cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.alpha / cfg.lora.rank)
    log.info("LoRA A/B: %d tensors / %d parameters; head: %d parameters; frozen base: %d parameters",
             len(lora_parameters), lora_numel, head_numel,
             sum(parameter.numel() for parameter in base_parameters.values()))
    log.info("trainable: LoRA A/B + classifier head = %d parameters (this is not LoRA-only training)",
             lora_numel + head_numel)
    log.info("optimizer: AdamW lr=%s weight_decay=%s, no scheduler; batch_size=%d max_steps=%d",
             cfg.training.learning_rate, cfg.training.weight_decay, batch_size, max_steps)

    # Training: CLS -> head -> cross entropy -> AdamW over LoRA A/B + head.
    groups = {
        "lora_a": [parameter for name, parameter in lora_parameters.items() if ".lora_a." in name],
        "lora_b": [parameter for name, parameter in lora_parameters.items() if ".lora_b." in name],
        "head": list(head.parameters()),
    }
    initial = {group: [parameter.detach().clone() for parameter in params] for group, params in groups.items()}
    base_initial = {name: parameter.detach().clone() for name, parameter in base_parameters.items()}
    experiment = configure_logger(
        logged_params=OmegaConf.to_container(cfg, resolve=True),
        model_name="vit_lora_cifar10",
        disable_logging=not cfg.logging.comet_enabled)

    encoder.train()
    head.train()
    history = []
    step = 0
    while step < max_steps:
        for images, labels in loader:
            step += 1
            images, labels = images.to(device), labels.to(device)
            if tuple(images.shape) != (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"Unexpected pixel_values shape: {tuple(images.shape)}")
            logits = head(encoder(images))
            if tuple(logits.shape) != (batch_size, NUM_CLASSES):
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in base_parameters.values()):
                raise RuntimeError("Frozen base parameters received gradients")
            if any(parameter.grad is None for parameter in trainable):
                raise RuntimeError("A trainable parameter received no gradient")
            grad_norm = {
                group: total_norm(parameter.grad for parameter in params) for group, params in groups.items()}
            before = {group: [parameter.detach().clone() for parameter in params] for group, params in groups.items()}
            optimizer.step()
            update_norm = {
                group: total_norm(parameter.detach() - previous for parameter, previous in zip(params, before[group]))
                for group, params in groups.items()}
            record = {
                "step": step,
                "loss": loss.item(),
                "accuracy": (logits.argmax(dim=1) == labels).float().mean().item(),
                **{f"grad_norm/{group}": value for group, value in grad_norm.items()},
                **{f"update_norm/{group}": value for group, value in update_norm.items()},
            }
            if not all(math.isfinite(value) for value in record.values()):
                raise RuntimeError(f"Non-finite value at step {step}: {record}")
            if not all(torch.isfinite(parameter).all() for parameter in trainable):
                raise RuntimeError(f"Non-finite trainable parameter after step {step}")
            history.append(record)
            experiment.log_metrics(record, step=step)
            log.info(
                "step %d/%d loss=%.6f acc=%.4f logits=%s | grad_norm A=%.3e B=%.3e head=%.3e | "
                "update_norm A=%.3e B=%.3e head=%.3e",
                step, max_steps, record["loss"], record["accuracy"], tuple(logits.shape),
                grad_norm["lora_a"], grad_norm["lora_b"], grad_norm["head"],
                update_norm["lora_a"], update_norm["lora_b"], update_norm["head"])
            if step == max_steps:
                break
    experiment.end()

    base_changed = [name for name, parameter in base_parameters.items()
                    if not torch.equal(parameter, base_initial[name])]
    base_update_norm = total_norm(
        parameter.detach() - base_initial[name] for name, parameter in base_parameters.items())
    changed = {
        group: sum(not torch.equal(parameter, start) for parameter, start in zip(params, initial[group]))
        for group, params in groups.items()}
    log.info("tensors changed from init: lora_a %d/%d, lora_b %d/%d, head %d/%d",
             changed["lora_a"], len(groups["lora_a"]), changed["lora_b"], len(groups["lora_b"]),
             changed["head"], len(groups["head"]))
    log.info("base tensors changed: %d, base update norm: %.3e", len(base_changed), base_update_norm)
    if base_changed or base_update_norm != 0:
        raise RuntimeError(f"Frozen base parameters changed: {base_changed[:5]}")
    if changed["lora_b"] == 0 or changed["head"] == 0:
        raise RuntimeError("Training did not update LoRA B and the classifier head")

    # Save LoRA A/B + head, rebuild from the pretrained checkpoint, compare eval outputs.
    eval_images = next(iter(DataLoader(Subset(dataset, subset_indices[:batch_size]), batch_size=batch_size)))[0]
    eval_images = eval_images.to(device)
    encoder.eval()
    head.eval()
    with torch.no_grad():
        trained_cls = encoder(eval_images)
        trained_logits = head(trained_cls)

    checkpoint_path = Path(output_dir) / CHECKPOINT_FILENAME
    torch.save({
        "checkpoint_id": cfg.model.checkpoint_id,
        "lora": {
            "target_modules": list(encoder.target_modules),
            "rank": cfg.lora.rank,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
        },
        "seed": seed,
        "data": {
            "name": "CIFAR10",
            "split": "train",
            "subset_indices": subset_indices,
            "image_size": IMAGE_SIZE,
            "image_mean": image_mean,
            "image_std": image_std,
        },
        "steps": step,
        "adapter": {name: tensor.cpu() for name, tensor in encoder.lora_state_dict().items()},
        "head": {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()},
    }, checkpoint_path)

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded_encoder = ViTLoRAFrameEncoder(
        checkpoint_id=saved["checkpoint_id"], enabled=True, **saved["lora"]).to(device).eval()
    reloaded_head = nn.Linear(FEATURE_SIZE, NUM_CLASSES).to(device).eval()
    reloaded_head.load_state_dict(saved["head"])
    with torch.no_grad():
        head_only_logits = reloaded_head(reloaded_encoder(eval_images))
    reloaded_encoder.load_lora_state_dict(saved["adapter"])
    with torch.no_grad():
        reloaded_cls = reloaded_encoder(eval_images)
        reloaded_logits = reloaded_head(reloaded_cls)
    torch.testing.assert_close(reloaded_cls, trained_cls)
    torch.testing.assert_close(reloaded_logits, trained_logits)
    if torch.allclose(head_only_logits, trained_logits):
        raise RuntimeError("Head-only reload reproduces the trained logits; the adapter check is not discriminative")
    reload_max_abs_diff = (reloaded_logits - trained_logits).abs().max().item()
    log.info("saved LoRA A/B + head: %s", checkpoint_path)
    log.info("reload eval logits max abs diff: %.3e (head-only reload differs: True)", reload_max_abs_diff)

    elapsed = time.perf_counter() - start_time
    log.info("Stage B checks passed: %d steps, base unchanged, LoRA B and head updated, reload matched "
             "(%.1f s on %s)", step, elapsed, device)
    return {
        "steps": step,
        "history": history,
        "changed": changed,
        "base_changed": len(base_changed),
        "base_update_norm": base_update_norm,
        "reload_max_abs_diff": reload_max_abs_diff,
        "checkpoint_path": checkpoint_path,
        "subset_indices": subset_indices,
        "device": str(device),
    }


@hydra.main(version_base="1.3", config_path="../../conf", config_name="lora_standalone")
def main(cfg: DictConfig):
    run(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
