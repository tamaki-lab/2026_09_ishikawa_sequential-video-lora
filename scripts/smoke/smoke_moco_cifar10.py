"""Standalone MoCo v2-style CIFAR-10 mechanics check (no ViT, LoRA, Sequential Loader or video).

Gate B, 20-step smoke (defaults: fresh run, constant LR, CPU):
    python -m scripts.smoke.smoke_moco_cifar10 data.root=/path/to/CIFAR10
Gate C, 500-step pilot (fresh run, cosine LR; run only with separate approval):
    python -m scripts.smoke.smoke_moco_cifar10 data.root=/path/to/CIFAR10 \
        training.max_steps=500 training.scheduler=cosine
Resume a saved checkpoint into a new run directory:
    python -m scripts.smoke.smoke_moco_cifar10 ... training.resume=<run_dir>/checkpoint.pt

Each run writes log/moco_standalone/<YYYY-MM-DD>/<HH-MM-SS-ffffff>/. The MoCo step order is
documented in self_supervised/moco/standalone_moco.py. Comet logging is not part of this check.
"""

import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from scripts.smoke.moco_verification_reporting import (
    ARTIFACTS, build_diagnostics, evaluate_checks, expected_shapes, gate_statuses, write_verification_artifacts,
)
from self_supervised.moco import CifarResNet18, StandaloneMoCo
from self_supervised.moco.standalone_moco import UNIT_NORM_ATOL

SCHEMA_VERSION = "moco_standalone_report/v1"
CHECKPOINT_FORMAT = "moco_standalone_checkpoint/v1"
REFERENCE = {
    "repository": "facebookresearch/moco",
    "commit": "a419ef282fc9885015674e77f58c59447206a27f",
    "files": ["moco/builder.py", "main_moco.py", "README.md", "LICENSE"],
}
CIFAR10_TRAIN_SIZE = 50_000
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ORDER_POLICY = ("1 epoch内の固定順序: torch.randperm(N, seed) の先頭から連続消費"
                "（batch単位、drop_last、num_workers 0、同一画像の再読込なし）")
EMA_RTOL, EMA_ATOL = 1e-5, 1e-6
RELOAD_RTOL, RELOAD_ATOL = 1e-5, 1e-6
# Approved Gate B/C settings (spec 2026-10-15); any other combination is reported as a custom run.
APPROVED_SETTINGS = {
    "model.projection_dim": 128, "moco.queue_size": 1024, "moco.momentum": 0.999, "moco.temperature": 0.2,
    "training.seed": 0, "training.batch_size": 32, "training.learning_rate": 0.00375,
    "training.momentum": 0.9, "training.weight_decay": 0.0001, "training.nesterov": False,
}
RUN_PRESETS = {"gate_b_smoke20": (20, "none"), "gate_c_pilot500": (500, "cosine")}


def moco_v2_cifar_augmentation() -> transforms.Compose:
    """Official MoCo v2 ``aug_plus`` adapted to 32x32: GaussianBlur kernel 3 instead of the 224-px PIL blur."""
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class TwoCropsTransform:
    """Two independent random draws of one transform from the same image (reference: moco/loader.py)."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        return [self.base_transform(image), self.base_transform(image)]


class TwoViewDataset(Dataset):
    """Yield ``(im_q, im_k, index)``; the class label of the wrapped dataset is discarded here."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        (im_q, im_k), _ = self.base[index]
        return im_q, im_k, index


def single_epoch_order(num_samples: int, seed: int) -> torch.Tensor:
    return torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))


def _sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _sha256_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def config_identity(config: dict) -> dict:
    """Fields a resumed run must share with its checkpoint; LR-schedule horizon only matters for cosine."""
    identity = copy.deepcopy(config)
    training = identity["training"]
    for key in ("resume", "device"):
        training.pop(key, None)
    if training["scheduler"] == "none":
        training.pop("max_steps", None)
    return identity


def _config_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config_identity(config), sort_keys=True).encode()).hexdigest()


def validate_config(cfg: DictConfig) -> None:
    if cfg.data.name != "CIFAR10" or cfg.data.split != "train":
        raise ValueError("The standalone MoCo check supports CIFAR10 train only")
    if cfg.model.name != "resnet18_cifar":
        raise ValueError("The standalone MoCo check supports model.name=resnet18_cifar only")
    if cfg.logging.comet_enabled:
        raise ValueError("Comet logging is not implemented for the standalone MoCo check; "
                         "local artifacts are the record")
    if cfg.training.scheduler not in ("none", "cosine"):
        raise ValueError(f"training.scheduler must be none or cosine, got {cfg.training.scheduler}")
    batch_size, max_steps = int(cfg.training.batch_size), int(cfg.training.max_steps)
    queue_size = int(cfg.moco.queue_size)
    if batch_size < 1 or max_steps < 1:
        raise ValueError("training.batch_size and training.max_steps must be positive")
    if queue_size % batch_size:
        raise ValueError(f"moco.queue_size={queue_size} must be a multiple of training.batch_size={batch_size}")


def resolve_device(choice: str) -> torch.device:
    """CPU unless CUDA is requested explicitly; the GPU index is chosen outside (CUDA_VISIBLE_DEVICES)."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device=cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    raise ValueError(f"training.device must be cpu or cuda, got {choice}")


def run_kind(cfg: DictConfig, sample_count: int | None) -> str:
    """gate_b_smoke20 / gate_c_pilot500 only for fresh runs with the approved settings on full CIFAR-10."""
    if cfg.training.resume is not None or sample_count not in (None, CIFAR10_TRAIN_SIZE):
        return "custom"
    if any(OmegaConf.select(cfg, key) != value for key, value in APPROVED_SETTINGS.items()):
        return "custom"
    for kind, (steps, scheduler) in RUN_PRESETS.items():
        if int(cfg.training.max_steps) == steps and cfg.training.scheduler == scheduler:
            return kind
    return "custom"


def build_training(cfg: DictConfig, device: torch.device):
    """Model, SGD over the query backbone + projector only, and the optional cosine LR schedule."""
    model = StandaloneMoCo(
        CifarResNet18(), dim=int(cfg.model.projection_dim), K=int(cfg.moco.queue_size),
        m=float(cfg.moco.momentum), T=float(cfg.moco.temperature),
    ).to(device)
    optimizer = torch.optim.SGD(
        model.query_parameters(), lr=float(cfg.training.learning_rate), momentum=float(cfg.training.momentum),
        weight_decay=float(cfg.training.weight_decay), nesterov=bool(cfg.training.nesterov),
    )
    scheduler = None
    if cfg.training.scheduler == "cosine":
        max_steps = int(cfg.training.max_steps)
        # lr(t) = lr0 * 0.5 * (1 + cos(pi * t / max_steps)), t = completed optimizer steps.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda t: 0.5 * (1.0 + math.cos(math.pi * t / max_steps)),
        )
    return model, optimizer, scheduler


def _global_norm(tensors) -> float | None:
    present = [t.detach() for t in tensors if t is not None]
    if not present:
        return None
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t) for t in present])))


def _ema_error(m: float, key_before, query_before, key_after) -> tuple[float, bool]:
    """Independent float64 reference of theta_k <- m * theta_k + (1 - m) * theta_q."""
    worst, within = 0.0, True
    for k_old, q_old, k_new in zip(key_before, query_before, key_after, strict=True):
        expected = k_old.double() * m + q_old.double() * (1.0 - m)
        diff = (k_new.detach().double() - expected).abs()
        worst = max(worst, float(diff.max()))
        within = within and bool((diff <= EMA_ATOL + EMA_RTOL * expected.abs()).all())
    return worst, within


def train_step(model: StandaloneMoCo, optimizer, im_q: torch.Tensor, im_k: torch.Tensor):
    """One MoCo step: forward (query, EMA, key, logits, enqueue) -> backward -> SGD; returns (output, record)."""
    backbone_params, projector_params = list(model.backbone_q.parameters()), list(model.projector_q.parameters())
    query_params, key_params = backbone_params + projector_params, model.key_parameters()
    query_before = [p.detach().clone() for p in query_params]
    key_before = [p.detach().clone() for p in key_params]

    out = model(im_q, im_k)
    ema_max_abs_error, ema_ok = _ema_error(model.m, key_before, query_before, key_params)
    if not ema_ok:
        raise RuntimeError(f"Key EMA deviates from m*theta_k + (1-m)*theta_q (max abs error {ema_max_abs_error:.3g})")

    optimizer.zero_grad(set_to_none=True)
    out.loss.backward()
    key_grad_tensors = sum(p.grad is not None for p in key_params)
    if key_grad_tensors:
        raise RuntimeError(f"{key_grad_tensors} key tensors received a gradient")
    backbone_grad = _global_norm(p.grad for p in backbone_params)
    projector_grad = _global_norm(p.grad for p in projector_params)
    if not all(value is not None and math.isfinite(value) for value in (backbone_grad, projector_grad)):
        raise FloatingPointError(f"missing or non-finite query gradient: backbone={backbone_grad}, "
                                 f"projector={projector_grad}")
    lr = float(optimizer.param_groups[0]["lr"])
    optimizer.step()

    n_backbone = len(backbone_params)
    backbone_update = _global_norm(p.detach() - before for p, before in zip(backbone_params, query_before[:n_backbone]))
    projector_update = _global_norm(p.detach() - before
                                    for p, before in zip(projector_params, query_before[n_backbone:]))
    if not math.isfinite(backbone_update) or not math.isfinite(projector_update):
        raise FloatingPointError("non-finite query parameter update")

    with torch.no_grad():
        negative = out.negative_cosine
        z_q_norm, z_k_norm = out.z_q.detach().norm(dim=1), out.z_k.norm(dim=1)
        q_variance, k_variance = out.q.detach().var(dim=0, correction=0), out.k.var(dim=0, correction=0)
        record = {
            "lr": lr,
            "loss": float(out.loss.detach()),
            "positive_cosine_mean": float(out.positive_cosine.mean()),
            "negative_cosine_mean": float(negative.mean()),
            "hardest_negative_cosine_mean": float(negative.max(dim=1).values.mean()),
            "hardest_negative_cosine_max": float(negative.max()),
            "z_q_norm_min": float(z_q_norm.min()), "z_q_norm_mean": float(z_q_norm.mean()),
            "z_k_norm_min": float(z_k_norm.min()), "z_k_norm_mean": float(z_k_norm.mean()),
            "q_variance_mean": float(q_variance.mean()), "q_variance_min": float(q_variance.min()),
            "k_variance_mean": float(k_variance.mean()), "k_variance_min": float(k_variance.min()),
            "query_backbone_grad_norm": backbone_grad, "query_projector_grad_norm": projector_grad,
            "query_backbone_update_norm": backbone_update, "query_projector_update_norm": projector_update,
            "ema_max_abs_error": ema_max_abs_error, "ema_within_tolerance": ema_ok,
            "key_grad_tensors": key_grad_tensors,
            "queue_ptr": int(model.queue_ptr), "real_key_count": int(model.real_key_count),
            "random_remaining_count": model.random_key_remaining,
        }
    return out, record


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def capture_rng_state(device: torch.device) -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"bit_generator": np_state[0], "keys": np_state[1].tolist(), "pos": int(np_state[2]),
                  "has_gauss": int(np_state[3]), "cached_gaussian": float(np_state[4])},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng_state(rng: dict, device: torch.device) -> None:
    random.setstate(rng["python"])
    numpy = rng["numpy"]
    np.random.set_state((numpy["bit_generator"], np.array(numpy["keys"], dtype=np.uint32), numpy["pos"],
                         numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.set_rng_state(rng["torch"])
    if device.type == "cuda" and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def save_checkpoint(path: Path, model, optimizer, scheduler, *, global_step: int, history: list[dict],
                    sampler: dict, config: dict, device: torch.device, status: str, resumable: bool) -> None:
    """CPU tensors only; the key branch and queue carry no autograd graph."""
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "status": status,
        "resumable": resumable,
        "global_step": global_step,
        "model": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "scheduler": None if scheduler is None else _to_cpu(scheduler.state_dict()),
        "sampler": sampler,
        "rng": capture_rng_state(device),
        "config": config,
        "config_identity_sha256": _config_sha256(config),
        "reference": REFERENCE,
        "history": history,
    }, path)


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    return checkpoint


def restore_training_state(checkpoint: dict, model: StandaloneMoCo, optimizer, scheduler) -> None:
    model.load_state_dict(checkpoint["model"], strict=True)
    model.check_queue_state()
    optimizer.load_state_dict(checkpoint["optimizer"])
    if (scheduler is None) != (checkpoint["scheduler"] is None):
        raise ValueError("checkpoint scheduler presence does not match training.scheduler")
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])


def restore_for_resume(checkpoint: dict, config: dict, model, optimizer, scheduler, sampler: dict,
                       device: torch.device) -> int:
    """Restore model/optimizer/scheduler/RNG and return the completed global step."""
    if not checkpoint["resumable"]:
        raise ValueError(f"checkpoint status={checkpoint['status']} was saved mid-failure and cannot be resumed")
    if config_identity(checkpoint["config"]) != config_identity(config):
        raise ValueError("resume config differs from the checkpoint; only training.resume, training.device and "
                         "(with scheduler none) training.max_steps may change")
    global_step = int(checkpoint["global_step"])
    if global_step >= int(config["training"]["max_steps"]):
        raise ValueError(f"checkpoint already completed {global_step} >= training.max_steps steps")
    saved = checkpoint["sampler"]
    if saved["order_sha256"] != sampler["order_sha256"] or saved["cursor"] != global_step * sampler["batch_size"]:
        raise ValueError("checkpoint sampler order or cursor does not match this dataset/config")
    restore_training_state(checkpoint, model, optimizer, scheduler)
    restore_rng_state(checkpoint["rng"], device)  # last: model construction above consumed the global RNG
    return global_step


def verify_checkpoint_roundtrip(path: Path, cfg: DictConfig, device: torch.device, model, optimizer, scheduler,
                                sample: torch.Tensor) -> dict:
    """Load the saved file into a separately constructed instance and compare state and eval features."""
    fresh_model, fresh_optimizer, fresh_scheduler = build_training(cfg, device)
    restore_training_state(load_checkpoint(path), fresh_model, fresh_optimizer, fresh_scheduler)
    state_equal = _nested_equal(model.state_dict(), fresh_model.state_dict())
    optimizer_equal = _nested_equal(optimizer.state_dict(), fresh_optimizer.state_dict())
    scheduler_equal = (scheduler is None and fresh_scheduler is None) or _nested_equal(
        scheduler.state_dict(), fresh_scheduler.state_dict())
    queue_equal = all(torch.equal(getattr(model, name), getattr(fresh_model, name))
                      for name in ("queue", "queue_ptr", "real_key_count"))
    model.eval()
    fresh_model.eval()
    with torch.no_grad():
        pairs = [(model.encode_query(sample), fresh_model.encode_query(sample)),
                 (model.encode_key(sample), fresh_model.encode_key(sample))]
    diff = max(float((a - b).abs().max()) for a, b in pairs)
    features_close = all(torch.allclose(a, b, rtol=RELOAD_RTOL, atol=RELOAD_ATOL) for a, b in pairs)
    return {
        "state_equal": state_equal, "optimizer_equal": optimizer_equal, "scheduler_equal": scheduler_equal,
        "queue_and_counters_equal": queue_equal, "eval_feature_max_abs_diff": diff,
        "tolerance": {"rtol": RELOAD_RTOL, "atol": RELOAD_ATOL},
        "passed": state_equal and optimizer_equal and scheduler_equal and queue_equal and features_close,
    }


def initial_query_key_audit(model: StandaloneMoCo) -> dict:
    pairs = [(model.backbone_q.state_dict(), model.backbone_k.state_dict()),
             (model.projector_q.state_dict(), model.projector_k.state_dict())]
    state_equal = all(q.keys() == k.keys() and all(torch.equal(q[name], k[name]) for name in q) for q, k in pairs)
    query_storage = {t.untyped_storage().data_ptr() for q, _ in pairs for t in q.values()}
    shared = sum(t.untyped_storage().data_ptr() in query_storage for _, k in pairs for t in k.values())
    return {"state_equal": state_equal, "shared_storage_tensors": shared,
            "tensors": sum(len(q) for q, _ in pairs)}


def _optimizer_matches(optimizer, model: StandaloneMoCo) -> bool:
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    key_ids = {id(p) for p in model.key_parameters()}
    return len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.query_parameters()} \
        and not key_ids & set(ids)


def _git_provenance() -> dict:
    repository = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True,
                              text=True).stdout

    try:
        status = git("status", "--porcelain", "--untracked-files=all").splitlines()
        return {"repository": "tamaki-lab/2026_09_ishikawa_sequential-video-lora",
                "commit": git("rev-parse", "HEAD").strip(), "branch": git("branch", "--show-current").strip(),
                "dirty": bool(status), "dirty_paths": status[:50]}
    except (OSError, subprocess.CalledProcessError):
        return {"repository": None, "commit": "unavailable", "branch": "unavailable", "dirty": None,
                "dirty_paths": None}


def runtime_provenance() -> dict:
    return {
        "git": _git_provenance(), "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__, "hydra": hydra.__version__,
        "cuda": torch.version.cuda, "cuda_device": None, "platform": platform.platform(), "host": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def new_report(cfg: DictConfig, config: dict) -> dict:
    batch_size, queue_size, dim = int(cfg.training.batch_size), int(cfg.moco.queue_size), int(cfg.model.projection_dim)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "run_kind": run_kind(cfg, None),
        "seed": int(cfg.training.seed),
        "device": str(cfg.training.device),
        "elapsed_seconds": None,
        "provenance": runtime_provenance(),
        "reference": REFERENCE,
        "config_identity_sha256": _config_sha256(config),
        "dataset": {
            "name": str(cfg.data.name), "split": str(cfg.data.split), "root": str(cfg.data.root),
            "download": bool(cfg.data.download), "sample_count": None, "order_policy": ORDER_POLICY,
            "order_seed": int(cfg.training.seed), "order_sha256": None, "samples_consumed": None,
            "num_workers": 0, "drop_last": True, "augmentation": None,
        },
        "model": {
            "backbone": "torchvision resnet18(weights=None)、conv1 3x3/stride1/pad1、maxpool=Identity、"
                        "fc除去（512-d pooled feature）",
            "feature_dim": CifarResNet18.feature_dim,
            "projector": f"Linear(512,512) → ReLU → Linear(512,{dim}) → L2 normalize(dim=1)",
            "projection_dim": dim, "query_parameters": None, "key_parameters": None,
            "initial_query_state_sha256": None,
        },
        "moco": {
            "queue_size": queue_size, "momentum": float(cfg.moco.momentum),
            "temperature": float(cfg.moco.temperature),
            "queue_init": "randn(dim, K)を列ごとにL2正規化（warm-upなし）",
            "negatives": "Queue全K列をNegative（sample-id/label maskなし）",
            "step_order": "query forward → key EMA（更新前θq）→ key forward → logits（enqueue前snapshot）"
                          " → enqueue → backward → SGD",
            "key_batchnorm": "Query/Keyとも train()。Key BN bufferはKey forwardのみで更新、parameterのみEMA",
            "initial_queue_sha256": None,
        },
        "optimizer": {
            "name": "SGD", "learning_rate": float(cfg.training.learning_rate),
            "momentum": float(cfg.training.momentum), "weight_decay": float(cfg.training.weight_decay),
            "nesterov": bool(cfg.training.nesterov), "scheduler": str(cfg.training.scheduler),
            "scheduler_formula": "lr(t) = lr0 * 0.5 * (1 + cos(pi * t / max_steps))"
                                 if cfg.training.scheduler == "cosine" else "constant lr0",
            "max_steps": int(cfg.training.max_steps), "batch_size": batch_size, "completed_steps": 0,
            "gradient_accumulation": False, "mixed_precision": False, "gradient_clipping": False,
        },
        "resume": {"resumed_from": None, "start_step": 0},
        "history": [],
        "verification": {
            "unit_norm_atol": UNIT_NORM_ATOL, "ema_tolerance": {"rtol": EMA_RTOL, "atol": EMA_ATOL},
            "expected_shapes": expected_shapes(batch_size, queue_size, dim),
        },
        "checks": {},
        "gates": {},
        "diagnostics": {},
        "artifacts": {},
        "failure": None,
    }


def history_audit(history: list[dict]) -> dict:
    """Aggregates of the measured steps; empty when no step was measured."""
    if not history:
        return {}
    return {
        "history_finite": all(math.isfinite(value) for item in history for value in item.values()
                              if isinstance(value, (int, float)) and not isinstance(value, bool)),
        "key_grad_tensors_max": max(item["key_grad_tensors"] for item in history),
        "ema_max_abs_error_max": max(item["ema_max_abs_error"] for item in history),
        "ema_within_tolerance_all_steps": all(item["ema_within_tolerance"] for item in history),
    }


@torch.no_grad()
def model_state_audit(state: dict, batch_size: int) -> dict:
    """Parameter change since the training loop started and the queue state, measured on the live model."""
    model, global_step = state["model"], state["global_step"]
    queue_size = model.K
    real = min(queue_size, global_step * batch_size)
    query_buffers, key_buffers = dict(model.backbone_q.named_buffers()), dict(model.backbone_k.named_buffers())
    running = [name for name in query_buffers if "running_" in name]
    backbone_pairs = zip(model.backbone_q.parameters(), state["backbone_start"])
    projector_pairs = zip(model.projector_q.parameters(), state["projector_start"])
    return {
        "query_backbone_tensors": len(state["backbone_start"]),
        "query_backbone_changed_tensors": sum(not torch.equal(p.detach(), before) for p, before in backbone_pairs),
        "query_projector_tensors": len(state["projector_start"]),
        "query_projector_changed_tensors": sum(not torch.equal(p.detach(), before) for p, before in projector_pairs),
        "expected_queue": {"queue_ptr": (global_step * batch_size) % queue_size, "real_key_count": real,
                           "random_remaining_count": queue_size - real},
        "final_queue": {
            "queue_ptr": int(model.queue_ptr), "real_key_count": int(model.real_key_count),
            "random_remaining_count": model.random_key_remaining, "finite": bool(torch.isfinite(model.queue).all()),
            "column_norm_max_abs_error": float((model.queue.norm(dim=0) - 1.0).abs().max()),
        },
        "bn_running_buffers_query_key_differing": {
            "differing": sum(not torch.equal(query_buffers[name], key_buffers[name]) for name in running),
            "total": len(running),
        },
    }


def _train(cfg: DictConfig, config: dict, output_dir: Path, report: dict, state: dict, context: dict) -> None:
    validate_config(cfg)
    device = resolve_device(str(cfg.training.device))
    report["device"] = str(device)
    if device.type == "cuda":
        report["provenance"]["cuda_device"] = torch.cuda.get_device_name(device)
    seed, batch_size, max_steps = int(cfg.training.seed), int(cfg.training.batch_size), int(cfg.training.max_steps)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    context["stage"] = "data"
    two_crops = TwoCropsTransform(moco_v2_cifar_augmentation())
    dataset = TwoViewDataset(CIFAR10(root=str(cfg.data.root), train=True, download=bool(cfg.data.download),
                                     transform=two_crops))
    order = single_epoch_order(len(dataset), seed)
    sampler = {"policy": ORDER_POLICY, "order_seed": seed, "num_samples": len(dataset),
               "order_sha256": _sha256_tensor(order), "batch_size": batch_size, "cursor": 0}
    report["dataset"].update(sample_count=len(dataset), order_sha256=sampler["order_sha256"],
                             samples_consumed=0, augmentation=repr(two_crops.base_transform))
    report["run_kind"] = run_kind(cfg, len(dataset))
    if max_steps * batch_size > len(dataset):
        raise ValueError(f"{max_steps} steps x batch {batch_size} exceed one epoch of {len(dataset)} samples")

    context["stage"] = "model"
    model, optimizer, scheduler = build_training(cfg, device)
    state.update(model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler, device=device,
                 global_step=0)
    verification = report["verification"]
    verification["initial_query_key"] = initial_query_key_audit(model)
    verification["key_requires_grad_tensors"] = sum(p.requires_grad for p in model.key_parameters())
    verification["optimizer_matches_query_parameters"] = _optimizer_matches(optimizer, model)
    query_state = {**{f"backbone.{k}": v for k, v in model.backbone_q.state_dict().items()},
                   **{f"projector.{k}": v for k, v in model.projector_q.state_dict().items()}}
    report["model"].update(query_parameters=sum(p.numel() for p in model.query_parameters()),
                           key_parameters=sum(p.numel() for p in model.key_parameters()),
                           initial_query_state_sha256=_sha256_state(query_state))
    report["moco"]["initial_queue_sha256"] = _sha256_tensor(model.queue)

    global_step, history = 0, report["history"]
    if cfg.training.resume is not None:
        context["stage"] = "resume"
        checkpoint = load_checkpoint(Path(str(cfg.training.resume)))
        global_step = restore_for_resume(checkpoint, config, model, optimizer, scheduler, sampler, device)
        history.extend(checkpoint["history"])
        report["resume"] = {"resumed_from": str(cfg.training.resume), "start_step": global_step}
        state["global_step"] = global_step

    context["stage"] = "train"
    state["backbone_start"] = [p.detach().clone() for p in model.backbone_q.parameters()]
    state["projector_start"] = [p.detach().clone() for p in model.projector_q.parameters()]
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=order[global_step * batch_size:max_steps * batch_size].tolist(),
        drop_last=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),  # keeps DataLoader seeding off the global RNG
    )
    observed = {name: [] for name in verification["expected_shapes"]}
    verification.update(data_order_matches_all_steps=True, targets_all_zero_all_steps=True)
    model.train()
    last_im_q = None
    for im_q, im_k, index in loader:
        context["step"] = global_step + 1
        expected_index = order[global_step * batch_size:(global_step + 1) * batch_size]
        verification["data_order_matches_all_steps"] &= bool(torch.equal(index, expected_index))
        im_q, im_k = im_q.to(device), im_k.to(device)
        out, record = train_step(model, optimizer, im_q, im_k)
        if scheduler is not None:
            scheduler.step()
        global_step += 1
        state["global_step"] = global_step
        verification["targets_all_zero_all_steps"] &= bool((out.targets == 0).all())
        for name, tensor in (("im_q", im_q), ("im_k", im_k), ("q", out.q), ("k", out.k), ("logits", out.logits)):
            if list(tensor.shape) not in observed[name]:
                observed[name].append(list(tensor.shape))
        verification["observed_shapes"] = observed
        report["optimizer"]["completed_steps"] = global_step
        report["dataset"]["samples_consumed"] = global_step * batch_size
        record = {"global_step": global_step, "sample_count": global_step * batch_size, **record}
        history.append(record)
        print(json.dumps(record), flush=True)
        last_im_q = im_q

    verification.update(model_state_audit(state, batch_size))

    context["stage"] = "checkpoint"
    sampler["cursor"] = global_step * batch_size
    path = output_dir / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, global_step=global_step, history=history, sampler=sampler,
                    config=config, device=device, status="completed", resumable=True)
    verification["checkpoint_roundtrip"] = verify_checkpoint_roundtrip(
        path, cfg, device, model, optimizer, scheduler, last_im_q[:8])


def _finalize(output_dir: Path, report: dict, started: float) -> None:
    report["elapsed_seconds"] = time.perf_counter() - started
    report["verification"].update(history_audit(report["history"]))
    report["checks"] = evaluate_checks(report)
    report["gates"] = gate_statuses(report)
    report["diagnostics"] = build_diagnostics(report["history"])
    report["artifacts"] = {name: name != "checkpoint.pt" or (output_dir / name).exists() for name, _ in ARTIFACTS}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_verification_artifacts(output_dir, report)


def run(cfg: DictConfig) -> dict:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name, _ in ARTIFACTS if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"{output_dir} already holds {existing}; runs never overwrite artifacts")
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    config = OmegaConf.to_container(cfg, resolve=True)
    report = new_report(cfg, config)
    started = time.perf_counter()
    state, context = {}, {"stage": "setup", "step": None}
    try:
        _train(cfg, config, output_dir, report, state, context)
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["failure"] = {"stage": context["stage"], "step": context["step"], "type": type(error).__name__,
                             "message": str(error)}
        if "backbone_start" in state and "final_queue" not in report["verification"]:
            report["verification"].update(model_state_audit(state, int(cfg.training.batch_size)))
        if "model" in state and not (output_dir / "checkpoint.pt").exists():
            try:
                state["sampler"]["cursor"] = state["global_step"] * int(cfg.training.batch_size)
                save_checkpoint(output_dir / "checkpoint.pt", state["model"], state["optimizer"], state["scheduler"],
                                global_step=state["global_step"], history=report["history"],
                                sampler=state["sampler"], config=config, device=state["device"],
                                status="failed", resumable=False)
            except Exception as save_error:  # keep the original failure as the reported cause
                report["failure"]["checkpoint_error"] = repr(save_error)
        _finalize(output_dir, report, started)
        raise
    report["status"] = "completed"
    _finalize(output_dir, report, started)
    print(f"Saved: {', '.join(name for name, _ in ARTIFACTS)} in {output_dir}")
    return report


@hydra.main(version_base="1.3", config_path="../../conf", config_name="moco_standalone")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
