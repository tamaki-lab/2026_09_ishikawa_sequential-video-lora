"""Spec A runner: one MoCo v2 implementation (reference clone or self) on ImageNet ResNet50 with 2-rank DDP.

    # canonical fixture first (CPU only)
    python -m scripts.parity.build_moco_parity_fixture fixture.kind=imagenet
    # one run per implementation (GPU runs need separate approval: spec §6, §11.2)
    CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=meta_reference \
        gate=A2 fixture.path=<fixture dir>/fixture.pt
    CUDA_VISIBLE_DEVICES=0,1 python -m scripts.parity.moco_imagenet_parity implementation=self \
        gate=A2 fixture.path=<fixture dir>/fixture.pt
    python -m scripts.parity.compare_moco_parity --reference <meta run dir> --candidate <self run dir>

run_kind=parity: both runs load the fixture state and views; right before each forward every rank seeds the CPU
global RNG (only) with the fixture's shuffle seed, so the permutation that rank 0 draws with ``torch.randperm`` and
broadcasts (shuffle-BN) is the same in both runs. Module hooks record each phase (after the DDP buffer broadcast,
after the query, after the key EMA, after the key forward, logits/enqueue); the harness records before forward,
after backward and after SGD. Hooks only clone tensors and never touch RNG state.
run_kind=bench (A5): ImageNet train through DistributedSampler + aug-plus; VRAM, step time, images/sec, data wait.

The reference side never edits the clone: ``main_moco.py`` is imported through ``scripts/parity/meta_moco_shim`` and
its parser, ``adjust_learning_rate`` and ``save_checkpoint`` are called; model construction, the DDP wrapper,
optimizer, criterion and step order restate ``main_worker``/``train``. Failures stop the run; nothing falls back.

Gate recipes (run both implementations with the same overrides, then compare; every GPU run needs approval):
    A1  fixture moco.queue_size=64 training.local_batch_size=4 data.image_size=64 (synthetic), then
        gate=A1 training.max_steps=16 trace.full_steps=[1,8,9,16] plus the same three overrides
    A2  synthetic and imagenet fixtures with the defaults; gate=A2 training.max_steps=1
    A3  synthetic fixture; gate=A3 training.max_steps=1025 trace.full_steps=[1,1024,1025] (queue fully replaced at 1024)
    A4  gate=A4 training.max_steps=3 training.checkpoint_steps=[2], then the same with
        training.resume=<run>/checkpoint_step000002.pt; compare --pair-kind resume; plus the CPU non-regression pytest
    A5  run_kind=bench gate=A5 training.max_steps=100..300 (no fixture); resource_benchmark.json
    threshold: two meta_reference runs of one fixture, compare --pair-kind rerun, review, then --threshold-status fixed
"""

import hashlib
import json
import math
import os
import random
import socket
import subprocess
import time
import traceback
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFilter
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from scripts.parity.moco_parity_state import (
    REFERENCE, REPOSITORY_ROOT, canonical_to_self_state, environment, expected_shuffle, file_sha256, git_info,
    group_stats, import_reference, mapping_table, reference_args, reference_argv, reference_two_crops,
    self_to_canonical, shuffle_seed, state_sha256, synthetic_views, tensor_sha256,
)
from scripts.smoke.smoke_moco_cifar10 import TwoCropsTransform, capture_rng_state, restore_rng_state
from self_supervised.moco import DistributedMoCo, ImageNetResNet50, concat_all_gather

RUN_SCHEMA = "moco_imagenet_parity_run/v1"
SIDECAR_FORMAT = "moco_imagenet_parity_sidecar/v1"
SELF_CHECKPOINT_FORMAT = "moco_imagenet_ddp_checkpoint/v1"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMPLEMENTATIONS, RUN_KINDS, GATES = ("meta_reference", "self"), ("parity", "bench"), ("A1", "A2", "A3", "A4", "A5")
PASS, FAIL, BLOCKED, NOT_RUN = "PASS", "FAIL", "BLOCKED", "NOT_RUN"
RUN_ARTIFACTS = ("resolved_config.yaml", "run_report.json", "provenance.json", "fixture_manifest.json",
                 "resource_benchmark.json", "verification_summary.md")
# Per-step booleans recorded by PhaseTracer; a check passes when it holds on every rank and step. Formula checks use
# |actual - float64 reference| <= atol + rtol * (sum of |operands|): a float32 rounding bound robust to cancellation.
WITHIN_RUN_CHECKS = {
    "shuffle_input_matches_rank0_permutation": "Key入力 = 全rank key画像のgather[rank0 permutationの自rank分]（完全一致）",
    "buffers_equal_rank0_after_broadcast": "DDP forward冒頭でBN buffer・Queue・ptrがrank0と完全一致（broadcast_buffers）",
    "key_params_equal_rank0_after_ema": "EMA後のKey parameterが全rank完全一致",
    "ema_within_tolerance": "EMA後Key = m·θk + (1−m)·θq（更新前Query、float64参照、許容誤差内）",
    "enqueued_keys_match_key_forward": "enqueueされた列 = 観測したshuffled key projectionのnormalize→gather→unshuffle",
    "logits_use_queue_before_enqueue": "logits = [q·k, q·Queue(当step enqueue前)]/T（float64再計算、許容誤差内）",
    "queue_unchanged_outside_slot": "enqueue対象のglobal batch列以外のQueue列は不変（完全一致）",
    "ptr_advanced_by_global_batch": "queue_ptr = (前ptr + global batch) mod K",
    "queue_equal_rank0_after_enqueue": "enqueue後のQueue・ptrが全rank完全一致",
    "targets_zero_long": "targetsが全0・long・[local batch]",
    "loss_matches_cross_entropy": "loss = CE(logits, 0)（float64再計算、許容誤差内）",
    "key_receives_no_gradient": "Key parameterにgradientなし",
    "grads_equal_rank0_after_allreduce": "backward後のQuery gradientが全rank完全一致（DDP all-reduce）",
    "sgd_matches_reference_update": "SGD更新後のQuery・momentum = float64参照式（lr, momentum, weight decay）",
    "params_and_momentum_equal_rank0_after_sgd": "SGD後のQuery parameter・momentum bufferが全rank完全一致",
    "finite": "loss・logits・q・k・gradient・更新後parameterが有限",
}
INFORMATIONAL = {
    "bn_buffers_differing_from_rank0_before_forward": "forward前（broadcast前）にrank0と異なるBN buffer数",
    "key_bn_buffers_differing_from_rank0_after_key_forward": "Key forward後にrank0と異なるKey BN buffer数",
}


# ----------------------------------------------------------------------------------------------- configuration
def validate_config(cfg: DictConfig) -> None:
    if cfg.implementation not in IMPLEMENTATIONS or cfg.run_kind not in RUN_KINDS or cfg.gate not in GATES:
        raise ValueError(f"implementation/run_kind/gate must be in {IMPLEMENTATIONS}/{RUN_KINDS}/{GATES}")
    if cfg.reference.commit != REFERENCE["commit"]:
        raise ValueError(f"reference.commit must stay {REFERENCE['commit']}")
    if cfg.precision.dtype != "float32" or cfg.precision.amp:
        raise ValueError("Spec A runs FP32 without AMP on both implementations")
    if not cfg.distributed.broadcast_buffers:
        raise ValueError("distributed.broadcast_buffers must stay true (the DDP default used by main_moco.py)")
    if not cfg.training.cos:
        raise ValueError("Spec A uses the main_moco --cos epoch schedule (training.cos=true)")
    world, batch, queue = int(cfg.distributed.world_size), int(cfg.training.local_batch_size), int(cfg.moco.queue_size)
    if world < 2 or batch < 1 or queue % (world * batch):
        raise ValueError(f"need world_size >= 2 and moco.queue_size % (world_size * local_batch_size) == 0, "
                         f"got {world}, {batch}, {queue}")
    max_steps = int(cfg.training.max_steps)
    if max_steps < 1 or any(not 1 <= int(step) <= max_steps for step in cfg.training.checkpoint_steps):
        raise ValueError("training.max_steps must be positive and checkpoint_steps within [1, max_steps]")
    if cfg.training.device == "cuda":
        if cfg.distributed.backend != "nccl":
            raise ValueError("training.device=cuda uses distributed.backend=nccl")
        if torch.cuda.device_count() < world:
            raise RuntimeError(f"{world} GPUs requested, {torch.cuda.device_count()} visible (CUDA_VISIBLE_DEVICES)")
    elif cfg.training.device == "cpu":
        if cfg.implementation != "self" or cfg.distributed.backend != "gloo" or cfg.run_kind != "parity":
            raise ValueError("training.device=cpu is only for gloo parity tests of implementation=self")
    else:
        raise ValueError(f"training.device must be cuda or cpu, got {cfg.training.device}")
    if cfg.run_kind == "parity" and cfg.fixture.path is None:
        raise ValueError("run_kind=parity needs fixture.path (scripts.parity.build_moco_parity_fixture)")
    if cfg.run_kind == "bench" and (cfg.training.resume is not None or list(cfg.training.checkpoint_steps)):
        raise ValueError("run_kind=bench neither resumes nor saves checkpoints")


def config_identity(config: dict) -> dict:
    """Fields a resumed run must share with its checkpoint (steps, tracing, logging and gate may change)."""
    training = config["training"]
    return {
        "implementation": config["implementation"], "reference_commit": config["reference"]["commit"],
        "model": config["model"], "moco": config["moco"], "image_size": config["data"]["image_size"],
        "world_size": config["distributed"]["world_size"],
        "broadcast_buffers": config["distributed"]["broadcast_buffers"], "precision": config["precision"],
        "training": {key: training[key]
                     for key in ("seed", "local_batch_size", "learning_rate", "momentum", "weight_decay", "epochs",
                                 "cos")},
    }


def config_identity_sha256(config: dict) -> str:
    return hashlib.sha256(json.dumps(config_identity(config), sort_keys=True).encode()).hexdigest()


def apply_backend_flags(cfg: DictConfig) -> dict:
    """Same FP32 backend state on both implementations (new TF32 API only; mixing the legacy one errors)."""
    torch.backends.cudnn.deterministic = bool(cfg.precision.cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(cfg.precision.cudnn_benchmark)
    torch.backends.fp32_precision = str(cfg.precision.fp32_precision)
    backends = torch.backends
    return {
        "cudnn_deterministic": backends.cudnn.deterministic, "cudnn_benchmark": backends.cudnn.benchmark,
        "fp32_precision": backends.fp32_precision, "cuda_matmul_fp32_precision": backends.cuda.matmul.fp32_precision,
        "cudnn_fp32_precision": backends.cudnn.fp32_precision,
        "cudnn_conv_fp32_precision": backends.cudnn.conv.fp32_precision,
        "cudnn_rnn_fp32_precision": backends.cudnn.rnn.fp32_precision,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


# ----------------------------------------------------------------------------------------------- self data path
class PILGaussianBlur:
    """Gaussian blur of a PIL image with radius ~ U(sigma) from Python's ``random`` (MoCo v2 aug-plus)."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = tuple(sigma)

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))

    def __repr__(self):
        return f"{type(self).__name__}(sigma={self.sigma})"


def moco_v2_imagenet_augmentation(image_size: int = 224) -> transforms.Compose:
    """Self MoCo v2 aug-plus for ImageNet (224 px): crop, color jitter, grayscale, PIL blur, flip, normalize."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([PILGaussianBlur((0.1, 2.0))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def epoch_cosine_lr(lr0: float, epoch: int, epochs: int) -> float:
    """lr0 * (0.5 * (1 + cos(pi * epoch / epochs))), set at the start of each epoch (as main_moco --cos)."""
    return lr0 * (0.5 * (1.0 + math.cos(math.pi * epoch / epochs)))


# ----------------------------------------------------------------------------------------------- adapters
class MetaReferenceAdapter:
    """The unmodified clone: builder.MoCo in DDP, SGD over model.parameters(), CrossEntropyLoss (main_worker)."""

    implementation = "meta_reference"
    checkpoint_format = "main_moco.save_checkpoint: {epoch, arch, state_dict (DDP 'module.' keys), optimizer}"

    def __init__(self, cfg: DictConfig, device: torch.device):
        if device.type != "cuda":
            raise ValueError("the reference builder runs on CUDA only (it calls .cuda())")
        self.cfg, self.device = cfg, device
        self.reference = import_reference(cfg.reference.root)
        self.main_moco, self.builder, self.loader = (self.reference[n] for n in ("main_moco", "builder", "loader"))
        self.args = reference_args(self.main_moco, cfg)
        self.args.gpu = device.index

    def build(self, canonical_state: dict | None) -> None:
        args = self.args
        model = self.builder.MoCo(torchvision.models.__dict__[args.arch], args.moco_dim, args.moco_k, args.moco_m,
                                  args.moco_t, args.mlp)
        if canonical_state is not None:
            model.load_state_dict(canonical_state, strict=True)
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        self.model = model
        self.ddp = DistributedDataParallel(model, device_ids=[args.gpu])
        self.criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        self.optimizer = torch.optim.SGD(self.ddp.parameters(), args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.query_head, self.key_entry, self.key_head = model.encoder_q, model.encoder_k, model.encoder_k

    def set_epoch_lr(self, epoch: int) -> None:
        self.main_moco.adjust_learning_rate(self.optimizer, epoch, self.args)

    def forward_loss(self, im_q, im_k):
        output, target = self.ddp(im_q=im_q, im_k=im_k)
        return self.criterion(output, target), output, target

    @staticmethod
    def split_output(output):
        logits, labels = output
        return logits, labels

    def named_query_params(self):
        return [(f"encoder_q.{name}", p) for name, p in self.model.encoder_q.named_parameters()]

    def named_key_params(self):
        return [(f"encoder_k.{name}", p) for name, p in self.model.encoder_k.named_parameters()]

    def named_bn_buffers(self, branch: str):
        return [(f"encoder_{branch}.{name}", b) for name, b in getattr(self.model, f"encoder_{branch}").named_buffers()]

    def canonical_state(self) -> dict:
        return self.model.state_dict()

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        self.main_moco.save_checkpoint({"epoch": epoch, "arch": self.args.arch, "state_dict": self.ddp.state_dict(),
                                        "optimizer": self.optimizer.state_dict()}, is_best=False, filename=str(path))

    def load_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=f"cuda:{self.args.gpu}")  # as main_moco.py --resume
        self.ddp.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint["epoch"])

    def bench_transform(self):
        return reference_two_crops(self.loader, int(self.cfg.data.image_size))

    def provenance(self) -> dict:
        args = {key: value for key, value in vars(self.args).items()
                if isinstance(value, (str, int, float, bool, list, type(None)))}
        return {"loaded": self.reference["provenance"], "main_moco_argv": reference_argv(self.cfg),
                "main_moco_args": args}


class SelfAdapter:
    """self_supervised.moco.DistributedMoCo(ImageNetResNet50()) in DDP, SGD over query_parameters() only."""

    implementation = "self"
    checkpoint_format = f"{SELF_CHECKPOINT_FORMAT}: {{format, epoch, model, optimizer}}"

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg, self.device, self.mapping = cfg, device, None

    def build(self, canonical_state: dict | None) -> None:
        cfg = self.cfg
        model = DistributedMoCo(ImageNetResNet50(), dim=int(cfg.model.projection_dim), K=int(cfg.moco.queue_size),
                                m=float(cfg.moco.momentum), T=float(cfg.moco.temperature))
        if canonical_state is not None:
            self.mapping = mapping_table(canonical_state, model.state_dict())
            model.load_state_dict(canonical_to_self_state(canonical_state, model.state_dict()), strict=True)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        model.to(self.device)
        self.model = model
        # DDP defaults as main_moco.py (broadcast_buffers=True; validate_config keeps it so).
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        self.ddp = DistributedDataParallel(model, device_ids=device_ids)
        self.optimizer = torch.optim.SGD(model.query_parameters(), lr=float(cfg.training.learning_rate),
                                         momentum=float(cfg.training.momentum),
                                         weight_decay=float(cfg.training.weight_decay), nesterov=False)
        self.query_head, self.key_entry, self.key_head = model.projector_q, model.backbone_k, model.projector_k

    def set_epoch_lr(self, epoch: int) -> None:
        lr = epoch_cosine_lr(float(self.cfg.training.learning_rate), epoch, int(self.cfg.training.epochs))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def forward_loss(self, im_q, im_k):
        out = self.ddp(im_q, im_k)
        return out.loss, out.logits, out.targets

    @staticmethod
    def split_output(output):
        return output.logits, output.targets

    def _named(self, items, prefixes):
        return [(self_to_canonical(name), t) for name, t in items if name.startswith(prefixes)]

    def named_query_params(self):
        return self._named(self.model.named_parameters(), ("backbone_q.", "projector_q."))

    def named_key_params(self):
        return self._named(self.model.named_parameters(), ("backbone_k.", "projector_k."))

    def named_bn_buffers(self, branch: str):
        return self._named(self.model.named_buffers(), (f"backbone_{branch}.", f"projector_{branch}."))

    def canonical_state(self) -> dict:
        return {canonical: t for name, t in self.model.state_dict().items()
                if (canonical := self_to_canonical(name)) is not None}

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save({"format": SELF_CHECKPOINT_FORMAT, "epoch": epoch, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    def load_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("format") != SELF_CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {SELF_CHECKPOINT_FORMAT} checkpoint")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.check_queue_state()
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint["epoch"])

    def bench_transform(self):
        return TwoCropsTransform(moco_v2_imagenet_augmentation(int(self.cfg.data.image_size)))

    def provenance(self) -> dict:
        files = [REPOSITORY_ROOT / "self_supervised" / "moco" / name for name in ("ddp_moco.py", "standalone_moco.py")]
        return {"files": {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in files}}


ADAPTERS = {"meta_reference": MetaReferenceAdapter, "self": SelfAdapter}


def momentum_buffers(adapter) -> list:
    return [adapter.optimizer.state.get(p, {}).get("momentum_buffer") for _, p in adapter.named_query_params()]


def check_optimizer_targets(adapter) -> dict:
    """The optimizer must step the query parameters in canonical order (reference: plus inert key params)."""
    params = [p for group in adapter.optimizer.param_groups for p in group["params"]]
    query = [p for _, p in adapter.named_query_params()]
    key_ids = {id(p) for _, p in adapter.named_key_params()}
    extra = params[len(query):]
    ok = [id(p) for p in params[:len(query)]] == [id(p) for p in query] and \
        all(id(p) in key_ids and not p.requires_grad for p in extra)
    if not ok:
        raise RuntimeError("optimizer parameters are not the query parameters in canonical order")
    return {"query_tensors": len(query), "inert_key_tensors_in_optimizer": len(extra),
            "hyperparameters": {k: v for k, v in adapter.optimizer.param_groups[0].items() if k != "params"}}


# ----------------------------------------------------------------------------------------------- tracing
@torch.no_grad()
def per_tensor_equal_to_rank0(tensors) -> torch.Tensor:
    """[world_size, n] bool: whether each rank's tensor i is bitwise equal to rank 0's (collective)."""
    local = [t.detach().contiguous().reshape(-1).view(torch.uint8) for t in tensors]
    flat = torch.cat(local)
    reference = flat.clone()
    dist.broadcast(reference, src=0)
    sizes = [chunk.numel() for chunk in local]
    equal = [torch.equal(a, b) for a, b in zip(flat.split(sizes), reference.split(sizes))]
    return concat_all_gather(torch.tensor([equal], dtype=torch.int32, device=flat.device)).bool()


def _cpu(named) -> dict:
    return {name: t.detach().cpu().clone() for name, t in named}


def _global_norm(tensors) -> float:
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t.double()) for t in tensors])))


class PhaseTracer:
    """Records one training step: module hooks plus explicit harness calls. Hooks clone tensors and return None."""

    def __init__(self, adapter, cfg: DictConfig, rank: int, world_size: int, device: torch.device):
        self.adapter, self.rank, self.world_size, self.device = adapter, rank, world_size, device
        self.T, self.m = float(cfg.moco.temperature), float(cfg.moco.momentum)
        self.weight_decay, self.momentum = float(cfg.training.weight_decay), float(cfg.training.momentum)
        self.rtol, self.atol = float(cfg.verification.rtol), float(cfg.verification.atol)
        self.local_batch = int(cfg.training.local_batch_size)
        self.global_batch = world_size * self.local_batch
        self.K = int(cfg.moco.queue_size)
        self.active = False
        adapter.model.register_forward_pre_hook(self._after_buffer_broadcast)
        adapter.query_head.register_forward_hook(self._after_query)
        adapter.key_entry.register_forward_pre_hook(self._after_key_ema)
        adapter.key_head.register_forward_hook(self._after_key_forward)
        adapter.model.register_forward_hook(self._after_enqueue)

    def _groups(self, *names) -> dict:
        a = self.adapter
        groups = {
            "query_params": lambda: a.named_query_params(), "key_params": lambda: a.named_key_params(),
            "query_bn": lambda: a.named_bn_buffers("q"), "key_bn": lambda: a.named_bn_buffers("k"),
            "queue": lambda: [("queue", a.model.queue), ("queue_ptr", a.model.queue_ptr)],
            "query_grads": lambda: [(name, p.grad) for name, p in a.named_query_params()],
            "momentum": lambda: list(zip([n for n, _ in a.named_query_params()], momentum_buffers(a))),
        }
        return {name: groups[name]() for name in names}

    def _capture(self, phase: str, groups: dict, cross_rank: bool) -> dict:
        entry = self.record["phases"].setdefault(phase, {"stats": {}, "full": {}, "rank0_equal": {}})
        for group, named in groups.items():
            tensors = [t for _, t in named]
            entry["stats"][group] = group_stats(tensors)
            if cross_rank:
                equal = per_tensor_equal_to_rank0(tensors)[self.rank]
                entry["rank0_equal"][group] = {"all": bool(equal.all()), "differing": int((~equal).sum()),
                                               "tensors": len(tensors)}
            if self.full:
                entry["full"][group] = _cpu(named)
        return entry

    def begin(self, step: int, full: bool, global_im_k: torch.Tensor, idx_shuffle: torch.Tensor, seed: int):
        self.step, self.full, self.global_im_k = step, full, global_im_k
        self.idx_shuffle = idx_shuffle.to(self.device)
        self.idx_this = self.idx_shuffle.view(self.world_size, -1)[self.rank]
        self.record = {"step": step, "rank": self.rank, "shuffle_seed": seed, "full": full, "phases": {},
                       "small": {"idx_shuffle": idx_shuffle.clone(), "idx_this": self.idx_this.cpu()}, "checks": {}}
        self.active = True

    def before_forward(self) -> None:
        a = self.adapter
        self.query_before = [p.detach().clone() for _, p in a.named_query_params()]
        self.key_before = [p.detach().clone() for _, p in a.named_key_params()]
        self.momentum_before = [None if b is None else b.clone() for b in momentum_buffers(a)]
        entry = self._capture("before_forward", self._groups("query_params", "key_params", "query_bn", "key_bn",
                                                             "queue"), cross_rank=True)
        self.record["checks"]["bn_buffers_differing_from_rank0_before_forward"] = \
            entry["rank0_equal"]["query_bn"]["differing"] + entry["rank0_equal"]["key_bn"]["differing"]

    def _after_buffer_broadcast(self, module, args):
        if not self.active:
            return
        entry = self._capture("after_buffer_broadcast", self._groups("query_bn", "key_bn", "queue"), cross_rank=True)
        self.record["checks"]["buffers_equal_rank0_after_broadcast"] = all(
            value["all"] for value in entry["rank0_equal"].values())
        self.queue_before, self.ptr_before = module.queue.detach().clone(), int(module.queue_ptr)

    def _after_query(self, module, args, output):
        if not self.active:
            return
        self.z_q = output.detach().clone()
        self.record["small"]["z_q"] = self.z_q.cpu()
        self._capture("after_query", self._groups("query_bn"), cross_rank=False)

    def _after_key_ema(self, module, args):
        if not self.active:
            return
        checks = self.record["checks"]
        checks["shuffle_input_matches_rank0_permutation"] = bool(torch.equal(args[0],
                                                                             self.global_im_k[self.idx_this]))
        entry = self._capture("after_key_ema", self._groups("key_params"), cross_rank=True)
        checks["key_params_equal_rank0_after_ema"] = entry["rank0_equal"]["key_params"]["all"]
        worst, within = 0.0, True
        for (_, k_new), k_old, q_old in zip(self.adapter.named_key_params(), self.key_before, self.query_before,
                                            strict=True):
            k_old, q_old = k_old.double(), q_old.double()
            expected = k_old * self.m + q_old * (1.0 - self.m)
            diff = (k_new.detach().double() - expected).abs()
            worst = max(worst, float(diff.max()))
            within = within and bool((diff <= self._bound(k_old.abs() * self.m + q_old.abs() * (1.0 - self.m))).all())
        checks["ema_max_abs_error"], checks["ema_within_tolerance"] = worst, within

    def _bound(self, operand_scale: torch.Tensor) -> torch.Tensor:
        """atol + rtol * (sum of |operands|): float32 rounding bound of a reference formula, robust to cancellation."""
        return self.atol + self.rtol * operand_scale

    def _after_key_forward(self, module, args, output):
        if not self.active:
            return
        self.z_k_shuffled = output.detach().clone()
        self.record["small"]["z_k_shuffled"] = self.z_k_shuffled.cpu()
        entry = self._capture("after_key_forward", self._groups("key_bn"), cross_rank=True)
        self.record["checks"]["key_bn_buffers_differing_from_rank0_after_key_forward"] = \
            entry["rank0_equal"]["key_bn"]["differing"]

    @torch.no_grad()
    def _after_enqueue(self, module, args, output):
        if not self.active:
            return
        checks, small = self.record["checks"], self.record["small"]
        logits, targets = self.adapter.split_output(output)
        logits = logits.detach()
        queue_after, ptr_after = module.queue.detach(), int(module.queue_ptr)
        start, stop = self.ptr_before, self.ptr_before + self.global_batch
        enqueued = queue_after[:, start:stop].T.clone()  # keys of all ranks, rank order
        k = enqueued[self.rank * self.local_batch:(self.rank + 1) * self.local_batch]
        q = F.normalize(self.z_q, dim=1)

        # Keys re-derived from the observed shuffled projections: normalize, gather (rank order), unshuffle.
        shuffled = concat_all_gather(F.normalize(self.z_k_shuffled, dim=1))
        expected_keys = torch.empty_like(shuffled)
        expected_keys[self.idx_shuffle] = shuffled
        key_error = float((enqueued - expected_keys).abs().max())
        checks["enqueued_keys_max_abs_error"] = key_error
        checks["enqueued_keys_match_key_forward"] = bool(
            ((enqueued - expected_keys).abs() <= self.atol + self.rtol * expected_keys.abs()).all())

        q64, k64, queue64 = q.double(), k.double(), self.queue_before.double()
        expected_logits = torch.cat([(q64 * k64).sum(dim=1, keepdim=True), q64 @ queue64], dim=1) / self.T
        scale = torch.cat([(q64.abs() * k64.abs()).sum(dim=1, keepdim=True), q64.abs() @ queue64.abs()], dim=1) / self.T
        logit_diff = (logits.double() - expected_logits).abs()
        checks["logits_max_abs_error"] = float(logit_diff.max())
        checks["logits_use_queue_before_enqueue"] = bool((logit_diff <= self._bound(scale)).all())
        outside = torch.ones(self.K, dtype=torch.bool, device=queue_after.device)
        outside[start:stop] = False
        checks["queue_unchanged_outside_slot"] = bool(torch.equal(queue_after[:, outside],
                                                                  self.queue_before[:, outside]))
        checks["ptr_advanced_by_global_batch"] = ptr_after == stop % self.K
        checks["targets_zero_long"] = targets.dtype == torch.long and tuple(targets.shape) == (self.local_batch,) \
            and bool((targets == 0).all())
        entry = self._capture("after_enqueue", self._groups("queue"), cross_rank=True)
        checks["queue_equal_rank0_after_enqueue"] = entry["rank0_equal"]["queue"]["all"]
        phase = self.record["phases"].setdefault("after_logits_before_enqueue",
                                                 {"stats": {}, "full": {}, "rank0_equal": {}})
        phase["stats"]["logits"] = group_stats([logits])
        if self.full:
            phase["full"]["logits"] = {"logits": logits.cpu().clone()}
            phase["full"]["queue_used_for_logits"] = {"queue": self.queue_before.cpu()}
        small.update(q=q.cpu(), k=k.cpu(), enqueued_keys=enqueued.cpu(), positive_logits=logits[:, 0].cpu(),
                     targets=targets.cpu(), queue_ptr_before=torch.tensor(self.ptr_before),
                     queue_ptr_after=torch.tensor(ptr_after))
        self.logits, self.targets, self.q, self.k = logits, targets, q, k

    def after_forward(self, loss: torch.Tensor) -> None:
        checks = self.record["checks"]
        self.record["small"]["loss"] = loss.detach().cpu()
        expected = F.cross_entropy(self.logits.double(), self.targets)
        error = abs(float(loss.detach().double() - expected))
        checks["loss_abs_error"] = error
        checks["loss_matches_cross_entropy"] = error <= self.atol + self.rtol * abs(float(expected))
        self.finite = all(bool(torch.isfinite(t).all()) for t in (loss, self.logits, self.q, self.k))

    def after_backward(self) -> None:
        a, checks = self.adapter, self.record["checks"]
        missing = [name for name, p in a.named_query_params() if p.grad is None]
        if missing:
            raise RuntimeError(f"{len(missing)} query parameters received no gradient, e.g. {missing[:3]}")
        checks["key_grad_tensors"] = sum(p.grad is not None for _, p in a.named_key_params())
        checks["key_receives_no_gradient"] = checks["key_grad_tensors"] == 0
        entry = self._capture("after_backward", self._groups("query_grads"), cross_rank=True)
        checks["grads_equal_rank0_after_allreduce"] = entry["rank0_equal"]["query_grads"]["all"]
        grads = [p.grad for _, p in a.named_query_params()]
        self.grad_norm = _global_norm(grads)
        self.finite = self.finite and all(bool(torch.isfinite(g).all()) for g in grads)

    def after_sgd(self) -> dict:
        a, checks = self.adapter, self.record["checks"]
        lr = float(a.optimizer.param_groups[0]["lr"])
        entry = self._capture("after_sgd", self._groups("query_params", "momentum"), cross_rank=True)
        checks["params_and_momentum_equal_rank0_after_sgd"] = all(v["all"] for v in entry["rank0_equal"].values())
        param_error, buffer_error, within = 0.0, 0.0, True
        params, buffers = a.named_query_params(), momentum_buffers(a)
        for (_, p), p_old, b_old, b_new in zip(params, self.query_before, self.momentum_before, buffers, strict=True):
            grad, p_old = p.grad.double(), p_old.double()
            d_p, d_scale = grad + self.weight_decay * p_old, grad.abs() + self.weight_decay * p_old.abs()
            if b_old is None:  # first step: buf = d_p
                b_ref, b_scale = d_p, d_scale
            else:
                b_ref = self.momentum * b_old.double() + d_p
                b_scale = self.momentum * b_old.double().abs() + d_scale
            p_ref, p_scale = p_old - lr * b_ref, p_old.abs() + lr * b_scale
            p_diff, b_diff = (p.detach().double() - p_ref).abs(), (b_new.double() - b_ref).abs()
            param_error, buffer_error = max(param_error, float(p_diff.max())), max(buffer_error, float(b_diff.max()))
            within = within and bool((p_diff <= self._bound(p_scale)).all()) and \
                bool((b_diff <= self._bound(b_scale)).all())
        checks.update(sgd_param_max_abs_error=param_error, sgd_momentum_max_abs_error=buffer_error,
                      sgd_matches_reference_update=within)
        update_norm = _global_norm([p.detach() - before for (_, p), before in zip(params, self.query_before)])
        self.finite = self.finite and all(bool(torch.isfinite(p).all()) for _, p in params)
        checks["finite"] = self.finite
        return {"lr": lr, "query_grad_norm": self.grad_norm, "query_update_norm": update_norm}

    def finish(self) -> dict:
        record = self.record
        self.active = False
        for name in ("query_before", "key_before", "momentum_before", "queue_before", "global_im_k", "z_q",
                     "z_k_shuffled", "logits", "targets", "q", "k", "record"):
            setattr(self, name, None)
        return record


def step_metrics(loss, logits, temperature: float, queue_ptr: int, extra: dict) -> dict:
    """Same definitions for both implementations, in local artifacts and Comet (one step = one SGD update)."""
    logits = logits.detach()
    return {"loss": float(loss.detach()), "positive_cosine_mean": float(logits[:, 0].mean()) * temperature,
            "negative_cosine_mean": float(logits[:, 1:].mean()) * temperature, "queue_ptr": queue_ptr, **extra}


# ----------------------------------------------------------------------------------------------- comet
class CometRun:
    """rank-0 Comet experiment with automatic logging off; logging happens after a step's computation."""

    def __init__(self, cfg: DictConfig, rank: int, params: dict):
        self.experiment = None
        if rank != 0 or not cfg.logging.comet_enabled:
            return
        from comet_ml import Experiment

        self.experiment = Experiment(
            project_name=cfg.logging.comet_project, log_code=False, log_graph=False, auto_param_logging=False,
            auto_metric_logging=False, parse_args=False, auto_output_logging=False, log_git_patch=False,
            auto_weight_logging=False, auto_histogram_weight_logging=False, auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False, auto_log_co2=False)
        self.experiment.set_name(f"moco_imagenet_{cfg.implementation}_{cfg.run_kind}_{cfg.gate}")
        self.experiment.add_tags(["moco_imagenet_parity", "spec_a", str(cfg.implementation), str(cfg.run_kind),
                                  str(cfg.gate)])
        self.experiment.log_parameters(params)

    def info(self) -> dict | None:
        if self.experiment is None:
            return None
        return {"key": self.experiment.get_key(), "url": self.experiment.url}

    def log_metrics(self, metrics: dict, step: int) -> None:
        if self.experiment is not None:
            self.experiment.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step)

    def end(self, status: str) -> None:
        if self.experiment is not None:
            self.experiment.log_other("status", status)
            self.experiment.end()


def comet_parameters(cfg: DictConfig, world_size: int) -> dict:
    self_git = git_info(REPOSITORY_ROOT)
    return {
        "implementation": cfg.implementation, "run_kind": cfg.run_kind, "gate": cfg.gate,
        "self_commit": self_git.get("commit"), "self_dirty": self_git.get("dirty"),
        "reference_commit": cfg.reference.commit, "K": int(cfg.moco.queue_size),
        "local_batch": int(cfg.training.local_batch_size),
        "global_batch": world_size * int(cfg.training.local_batch_size), "seed": int(cfg.training.seed),
        "precision": f"{cfg.precision.dtype}/fp32_precision={cfg.precision.fp32_precision}/amp={cfg.precision.amp}",
        "world_size": world_size, "lr": float(cfg.training.learning_rate), "moco_m": float(cfg.moco.momentum),
        "moco_t": float(cfg.moco.temperature), "image_size": int(cfg.data.image_size),
    }


# ----------------------------------------------------------------------------------------------- checkpoints
def checkpoint_name(step: int) -> str:
    return f"checkpoint_step{step:06d}"


def save_checkpoint(adapter, config: dict, ctx: dict, step: int, fixture_sha: str | None) -> None:
    """Native checkpoint from rank 0 (main_moco saves on rank 0 only) plus one RNG/progress sidecar per rank."""
    output_dir, rank, name = Path(ctx["output_dir"]), ctx["rank"], checkpoint_name(step)
    if rank == 0:
        adapter.save_checkpoint(output_dir / f"{name}.pt", epoch=0)
    torch.save({"format": SIDECAR_FORMAT, "implementation": adapter.implementation, "rank": rank,
                "global_step": step, "epoch": 0, "rng": capture_rng_state(ctx["device"]),
                "fixture_sha256": fixture_sha, "config_identity_sha256": config_identity_sha256(config)},
               output_dir / f"{name}.rank{rank}.sidecar.pt")
    dist.barrier()
    ctx["record"]["checkpoints"].append(name)


def resume(adapter, config: dict, ctx: dict, fixture_sha: str | None) -> int:
    path = Path(config["training"]["resume"])
    sidecar = torch.load(path.with_name(f"{path.stem}.rank{ctx['rank']}.sidecar.pt"), map_location="cpu",
                         weights_only=True)
    if sidecar.get("format") != SIDECAR_FORMAT or sidecar["implementation"] != adapter.implementation:
        raise ValueError(f"{path} sidecar is not a {adapter.implementation} {SIDECAR_FORMAT} file")
    if sidecar["config_identity_sha256"] != config_identity_sha256(config) or sidecar["fixture_sha256"] != fixture_sha:
        raise ValueError("resume config or fixture differs from the checkpoint")
    step = int(sidecar["global_step"])
    if step >= int(config["training"]["max_steps"]):
        raise ValueError(f"checkpoint already completed {step} >= training.max_steps steps")
    epoch = adapter.load_checkpoint(path)
    restore_rng_state(sidecar["rng"], ctx["device"])
    ctx["record"]["resume"] = {"checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
                               "native_epoch": epoch, "start_step": step}
    return step


def checkpoint_manifest(adapter_cls, config: dict, output_dir: Path, names: list[str]) -> dict:
    entries = {}
    for name in names:
        files = sorted(output_dir.glob(f"{name}*"))
        entries[name] = {
            "files": {f.name: file_sha256(f) for f in files},
            "native_format": adapter_cls.checkpoint_format,
            "native_contents": "query/key encoders (params + BN buffers), queue, queue_ptr, optimizer (momentum)",
            "sidecar_contents": "global_step, epoch, per-rank RNG (python/numpy/torch/cuda), fixture and config "
                                "identity",
            "saved_by": "native: rank 0 only (as main_moco.py); sidecar: every rank",
            "resume_restores": "native model + optimizer on every rank, then each rank's RNG; the fixture supplies "
                               "the next step's views and shuffle seed",
            "not_restorable": ["rank-1 BN running buffers of the saved step (rank 0's are loaded; DDP rebroadcasts "
                               "rank 0's at the next forward anyway)",
                               "DataLoader worker RNG / sampler position (parity runs use no DataLoader; bench runs "
                               "do not resume)"],
            "native_epoch_value": "0: the epoch in progress (main_moco stores epoch + 1 only at epoch end)",
        }
    return {"config_identity_sha256": config_identity_sha256(config), "checkpoints": entries}


# ----------------------------------------------------------------------------------------------- loops
def load_fixture(cfg: DictConfig) -> tuple[dict, str]:
    path = Path(str(cfg.fixture.path))
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    meta = fixture["meta"]
    expected = {"world_size": int(cfg.distributed.world_size), "local_batch_size": int(cfg.training.local_batch_size),
                "image_size": int(cfg.data.image_size), "arch": str(cfg.model.arch), "mlp": bool(cfg.model.mlp),
                "projection_dim": int(cfg.model.projection_dim), "queue_size": int(cfg.moco.queue_size),
                "momentum": float(cfg.moco.momentum), "temperature": float(cfg.moco.temperature),
                "seed": int(cfg.training.seed)}
    differing = {key: (meta.get(key), value) for key, value in expected.items() if meta.get(key) != value}
    if differing:
        raise ValueError(f"config differs from the fixture (fixture, config): {differing}")
    if meta["kind"] == "imagenet" and int(cfg.training.max_steps) > int(meta["steps"]):
        raise ValueError(f"the imagenet fixture stores {meta['steps']} steps < training.max_steps")
    if state_sha256(fixture["state"]) != meta["state_sha256"]:
        raise ValueError("fixture state hash does not match its manifest")
    return fixture, file_sha256(path)


def fixture_views(fixture: dict, step: int, world_size: int) -> list:
    meta = fixture["meta"]
    if meta["kind"] == "synthetic":
        return [synthetic_views(meta["seed"], step, rank, meta["local_batch_size"], meta["image_size"])
                for rank in range(world_size)]
    return [tuple(view.clone() for view in fixture["inputs"][step - 1][rank]) for rank in range(world_size)]


def start_state_hashes(adapter) -> dict:
    groups = {"query_params": adapter.named_query_params(), "key_params": adapter.named_key_params(),
              "query_bn": adapter.named_bn_buffers("q"), "key_bn": adapter.named_bn_buffers("k"),
              "queue": [("queue", adapter.model.queue), ("queue_ptr", adapter.model.queue_ptr)]}
    hashes = {name: state_sha256(dict(named)) for name, named in groups.items()}
    hashes["canonical_state"] = state_sha256(adapter.canonical_state())
    hashes["optimizer_state_entries"] = len(adapter.optimizer.state)
    return hashes


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parity_loop(cfg: DictConfig, config: dict, adapter, ctx: dict, comet: CometRun) -> None:
    rank, world_size, device, record = ctx["rank"], ctx["world_size"], ctx["device"], ctx["record"]
    ctx["stage"] = "fixture"
    fixture, fixture_sha = load_fixture(cfg)
    meta = fixture["meta"]
    record["fixture"] = {"path": str(cfg.fixture.path), "sha256": fixture_sha, "kind": meta["kind"],
                         "state_sha256": meta["state_sha256"]}
    ctx["stage"] = "model"
    torch.manual_seed(int(cfg.training.seed))
    start_step = 0
    if cfg.training.resume is not None:
        adapter.build(None)
        ctx["stage"] = "resume"
        start_step = resume(adapter, config, ctx, fixture_sha)
    else:
        adapter.build(fixture["state"])
    record["optimizer"] = check_optimizer_targets(adapter)
    record["start_state"] = start_state_hashes(adapter)  # fixture state, or the checkpoint when resuming
    record["mapping_table"] = getattr(adapter, "mapping", None)
    del fixture["state"]
    adapter.set_epoch_lr(0)  # every Spec A step lies in epoch 0 of the main_moco schedule
    adapter.ddp.train()
    tracer = PhaseTracer(adapter, cfg, rank, world_size, device)
    trace_dir = Path(ctx["output_dir"]) / "trace" / f"rank{rank}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    full_steps = {int(s) for s in cfg.trace.full_steps}
    checkpoint_steps = {int(s) for s in cfg.training.checkpoint_steps}
    batch = int(cfg.training.local_batch_size)

    for step in range(start_step + 1, int(cfg.training.max_steps) + 1):
        ctx["stage"], ctx["step"] = "train", step
        views = fixture_views(fixture, step, world_size)
        im_q, im_k = views[rank]
        hashes = {"im_q": tensor_sha256(im_q), "im_k": tensor_sha256(im_k)}
        idx_shuffle = expected_shuffle(meta["seed"], step, world_size * batch)
        seed = shuffle_seed(meta["seed"], step, rank)
        tracer.begin(step, step in full_steps, torch.cat([v[1] for v in views]).to(device), idx_shuffle, seed)
        im_q, im_k = im_q.to(device), im_k.to(device)
        _sync(device)
        started = time.perf_counter()
        # State injection: the CPU global RNG that torch.randperm (shuffle-BN) draws from; rank 0's is broadcast.
        torch.default_generator.manual_seed(seed)
        tracer.before_forward()
        loss, logits, _ = adapter.forward_loss(im_q, im_k)
        tracer.after_forward(loss)
        adapter.optimizer.zero_grad()
        loss.backward()
        tracer.after_backward()
        adapter.optimizer.step()
        extra = tracer.after_sgd()
        _sync(device)
        extra["step_time_sec_with_tracing"] = time.perf_counter() - started
        metrics = step_metrics(loss, logits, float(cfg.moco.temperature), int(adapter.model.queue_ptr), extra)
        trace = tracer.finish()
        trace.update(input_sha256=hashes, metrics=metrics, implementation=adapter.implementation)
        torch.save(trace, trace_dir / f"step_{step:06d}.pt")
        record["history"].append({"global_step": step, "input_sha256": hashes, "shuffle_seed": seed,
                                  "idx_shuffle": idx_shuffle.tolist(), "metrics": metrics,
                                  "checks": trace["checks"]})
        comet.log_metrics(metrics, step)
        if rank == 0:
            print(json.dumps({"global_step": step, **metrics}), flush=True)
        if step in checkpoint_steps:
            ctx["stage"] = "checkpoint"
            save_checkpoint(adapter, config, ctx, step, fixture_sha)


def bench_loop(cfg: DictConfig, config: dict, adapter, ctx: dict, comet: CometRun) -> None:
    rank, device, record = ctx["rank"], ctx["device"], ctx["record"]
    ctx["stage"] = "model"
    torch.manual_seed(int(cfg.training.seed))
    adapter.build(None)
    record["optimizer"] = check_optimizer_targets(adapter)
    ctx["stage"] = "data"
    two_crops = adapter.bench_transform()
    dataset = ImageFolder(str(Path(str(cfg.data.root)) / str(cfg.data.split)), two_crops)
    sampler = DistributedSampler(dataset)  # shuffle=True, seed=0 as main_moco.py
    loader = DataLoader(dataset, batch_size=int(cfg.training.local_batch_size), shuffle=False,
                        num_workers=int(cfg.data.num_workers), pin_memory=True, sampler=sampler, drop_last=True)
    record["data"] = {"samples": len(dataset), "classes": len(dataset.classes),
                      "num_workers": int(cfg.data.num_workers), "augmentation": repr(two_crops.base_transform),
                      "batches_per_rank": len(loader)}
    sampler.set_epoch(0)
    adapter.set_epoch_lr(0)
    adapter.ddp.train()
    torch.cuda.reset_peak_memory_stats(device)
    # Same DataLoader worker seeds in both implementations (model construction consumed the RNG differently).
    torch.default_generator.manual_seed(int(cfg.training.seed))
    iterator = iter(loader)
    for step in range(1, int(cfg.training.max_steps) + 1):
        ctx["stage"], ctx["step"] = "data", step
        waited = time.perf_counter()
        images, _ = next(iterator)
        data_wait = time.perf_counter() - waited
        ctx["stage"] = "train"
        query_before = [p.detach().clone() for _, p in adapter.named_query_params()]
        _sync(device)
        started = time.perf_counter()
        im_q, im_k = images[0].cuda(device, non_blocking=True), images[1].cuda(device, non_blocking=True)
        loss, logits, _ = adapter.forward_loss(im_q, im_k)
        adapter.optimizer.zero_grad()
        loss.backward()
        adapter.optimizer.step()
        _sync(device)
        step_time = time.perf_counter() - started
        params = [p for _, p in adapter.named_query_params()]
        extra = {"lr": float(adapter.optimizer.param_groups[0]["lr"]),
                 "query_grad_norm": _global_norm([p.grad for p in params]),
                 "query_update_norm": _global_norm([p.detach() - b for p, b in zip(params, query_before)]),
                 "step_time_sec": step_time, "data_wait_sec": data_wait,
                 "images_per_sec_global": ctx["world_size"] * int(cfg.training.local_batch_size) / step_time,
                 "images_per_sec_global_with_data_wait":
                     ctx["world_size"] * int(cfg.training.local_batch_size) / (step_time + data_wait),
                 "max_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}
        metrics = step_metrics(loss, logits, float(cfg.moco.temperature), int(adapter.model.queue_ptr), extra)
        if not math.isfinite(metrics["loss"]):
            raise FloatingPointError(f"non-finite loss at step {step}")
        record["history"].append({"global_step": step, "metrics": metrics})
        comet.log_metrics(metrics, step)
        if rank == 0:
            print(json.dumps({"global_step": step, **metrics}), flush=True)


def resource_record(device: torch.device, history: list, warmup: int) -> dict:
    record = {"device": str(device)}
    if device.type == "cuda":
        record.update(gpu=torch.cuda.get_device_name(device),
                      total_memory_mib=torch.cuda.get_device_properties(device).total_memory / 2**20,
                      max_memory_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                      max_memory_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    times = [h["metrics"].get("step_time_sec", h["metrics"].get("step_time_sec_with_tracing")) for h in history]
    measured = times[warmup:]
    if measured:
        ordered = sorted(measured)
        record["step_time_sec_after_warmup"] = {"steps": len(measured), "mean": sum(measured) / len(measured),
                                                "median": ordered[len(ordered) // 2],
                                                "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]}
        waits = [h["metrics"]["data_wait_sec"] for h in history[warmup:] if "data_wait_sec" in h["metrics"]]
        if waits:
            record["data_wait_sec_after_warmup"] = {"mean": sum(waits) / len(waits), "max": max(waits)}
            for name in ("images_per_sec_global", "images_per_sec_global_with_data_wait"):
                values = [h["metrics"][name] for h in history[warmup:]]
                record[f"{name}_after_warmup_mean"] = sum(values) / len(values)
    return record


# ----------------------------------------------------------------------------------------------- processes
def _worker(rank: int, config: dict, output_dir: str, port: int) -> None:
    cfg = OmegaConf.create(config)
    world_size = int(cfg.distributed.world_size)
    record = {"rank": rank, "status": "running", "history": [], "checkpoints": [], "failure": None}
    ctx = {"rank": rank, "world_size": world_size, "output_dir": output_dir, "record": record, "stage": "setup",
           "step": None}
    comet = None
    try:
        if cfg.training.device == "cuda":
            torch.cuda.set_device(rank)
            device = torch.device("cuda", rank)
        else:
            device = torch.device("cpu")
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // world_size))
        ctx["device"] = device
        record["backend_flags"] = apply_backend_flags(cfg)
        dist.init_process_group(str(cfg.distributed.backend), init_method=f"tcp://127.0.0.1:{port}",
                                world_size=world_size, rank=rank,
                                timeout=timedelta(minutes=float(cfg.distributed.timeout_minutes)))
        ctx["stage"] = "adapter"
        adapter = ADAPTERS[cfg.implementation](cfg, device)
        record["implementation_provenance"] = adapter.provenance()
        comet = CometRun(cfg, rank, comet_parameters(cfg, world_size))
        record["comet"] = comet.info()
        (parity_loop if cfg.run_kind == "parity" else bench_loop)(cfg, config, adapter, ctx, comet)
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "failed"
        record["failure"] = {"stage": ctx["stage"], "step": ctx["step"], "rank": rank, "type": type(error).__name__,
                             "message": str(error), "traceback": traceback.format_exc(limit=20)}
        if isinstance(error, torch.cuda.OutOfMemoryError):
            record["failure"]["memory_summary"] = torch.cuda.memory_summary(ctx["device"], abbreviated=True)
        raise
    finally:
        if "device" in ctx:
            record["resources"] = resource_record(ctx["device"], record["history"], int(cfg.bench.warmup_steps)
                                                  if cfg.run_kind == "bench" else 0)
        if comet is not None:
            comet.end(record["status"])
        Path(output_dir, f"rank{rank}_report.json").write_text(json.dumps(record, indent=2, ensure_ascii=False,
                                                                          default=str), encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def aggregate_checks(ranks: list) -> dict:
    """PASS when a per-step check holds on every rank and step; FAIL names the first (step, rank)."""
    results = {}
    histories = [r["history"] if r else [] for r in ranks]
    for name, description in WITHIN_RUN_CHECKS.items():
        observed = [(h["global_step"], rank, h["checks"].get(name)) for rank, history in enumerate(histories)
                    for h in history]
        if not observed or any(r is None for r in ranks):
            results[name] = {"status": NOT_RUN, "description": description, "detail": "no complete measurement"}
            continue
        failures = sorted((step, rank) for step, rank, value in observed if value is not True)
        results[name] = {"status": FAIL if failures else PASS, "description": description,
                         "first_failure": {"step": failures[0][0], "rank": failures[0][1]} if failures else None,
                         "measured": len(observed)}
    for name, description in INFORMATIONAL.items():
        results[name] = {"status": "INFO", "description": description,
                         "per_rank_step": {rank: [h["checks"].get(name) for h in history]
                                           for rank, history in enumerate(histories)}}
    for name in ("ema_max_abs_error", "enqueued_keys_max_abs_error", "logits_max_abs_error", "loss_abs_error",
                 "sgd_param_max_abs_error", "sgd_momentum_max_abs_error"):
        values = [h["checks"][name] for history in histories for h in history if name in h["checks"]]
        if values:
            results.setdefault("max_errors", {})[name] = max(values)
    return results


def provenance(cfg: DictConfig, ranks: list, output_dir: Path) -> dict:
    clone = git_info(cfg.reference.root)
    diff = ""
    if clone.get("commit"):
        diff = subprocess.run(["git", "-C", str(cfg.reference.root), "diff", str(cfg.reference.commit)],
                              capture_output=True, text=True).stdout
        if diff:
            (output_dir / "reference_clone.diff").write_text(diff, encoding="utf-8")
    preflight = OmegaConf.select(cfg, "preflight.report")
    return {
        "self_repository": {"repository": "tamaki-lab/2026_09_ishikawa_sequential-video-lora",
                            **git_info(REPOSITORY_ROOT)},
        "reference_clone": {**REFERENCE, **clone, "diff_vs_pinned_commit_sha256": hashlib.sha256(
            diff.encode()).hexdigest(), "diff_empty": diff == ""},
        "implementation": [r.get("implementation_provenance") if r else None for r in ranks],
        "environment": environment(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": [r.get("resources", {}).get("gpu") if r else None for r in ranks],
        "backend_flags": [r.get("backend_flags") if r else None for r in ranks],
        "preflight_report": {"path": str(preflight), "sha256": file_sha256(preflight)} if preflight else None,
    }


def fixture_manifest(cfg: DictConfig, ranks: list) -> dict | None:
    if cfg.run_kind != "parity":
        return None
    first = next((r for r in ranks if r and "fixture" in r), None)
    return {
        "fixture": first["fixture"] if first else None,
        "state_mapping": "identity (reference names)" if cfg.implementation == "meta_reference" else
        "explicit table below (self name -> canonical reference name)",
        "mapping_table": first.get("mapping_table") if first else None,
        "per_rank": [{"rank": rank, "start_state": r.get("start_state"), "optimizer": r.get("optimizer"),
                      "resume": r.get("resume"),
                      "steps": [{k: h[k] for k in ("global_step", "input_sha256", "shuffle_seed", "idx_shuffle")}
                                for h in r["history"]]} if r else None for rank, r in enumerate(ranks)],
    }


def render_summary(report: dict) -> str:
    lines = [
        f"# MoCo ImageNet parity run: {report['implementation']} / {report['run_kind']} / {report['gate']}", "",
        f"- status: **{report['status']}**",
        f"- run directory: `{report['output_dir']}`",
        f"- steps: {report['completed_steps']} / {report['max_steps']}",
        f"- tolerance (provisional unless reviewed): rtol={report['tolerance']['rtol']}, "
        f"atol={report['tolerance']['atol']}",
    ]
    for failure in report["failures"]:
        lines.append(f"- failure: rank {failure['rank']} stage {failure['stage']} step {failure['step']}: "
                     f"{failure['type']}: {failure['message']}")
    if report["run_kind"] == "parity":
        lines += ["", "## Within-run checks", "", "| check | status | detail |", "|---|---|---|"]
        for name, result in report["verification"].items():
            if name == "max_errors" or result["status"] == "INFO":
                continue
            detail = result.get("first_failure") or result.get("detail") or f"{result.get('measured')} rank-steps"
            lines.append(f"| {result['description']} | {result['status']} | {detail} |")
        lines += ["", "最大誤差（float64参照との差）: " + json.dumps(report["verification"].get("max_errors", {})), "",
                  "BN bufferのrank差（情報、判定しない）: " + json.dumps(
                      {k: report["verification"][k]["per_rank_step"] for k in INFORMATIONAL})]
    lines += ["", "## Resources", ""]
    for rank, resources in enumerate(report["resources"]):
        lines.append(f"- rank {rank}: {json.dumps(resources, ensure_ascii=False)}")
    lines += ["", "## Not established by this run", "",
              "- 公式との数値一致は `compare_moco_parity.py` の比較reportでのみ判定する（このrun単体では主張しない）",
              "- 許容誤差は正式確定前（spec §5.3, B2）。表現品質・10 epoch学習・Linear ProbeはSpec Aの対象外",
              "", "## Reproduction", "",
              "`resolved_config.yaml` の値で `python -m scripts.parity.moco_imagenet_parity` を実行する"
              "（Hydraの実際のoverrideは `.hydra/overrides.yaml`、GPUは `CUDA_VISIBLE_DEVICES` で2枚指定）。"
              "fixtureは `fixture_manifest.json` のpath/sha256のものを使う。",
              "", "## Artifacts", ""]
    lines += [f"- `{name}`" for name in report["artifacts"]]
    return "\n".join(lines) + "\n"


def run(cfg: DictConfig) -> dict:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in RUN_ARTIFACTS if (output_dir / name).exists()]
    if existing or (output_dir / "trace").exists():
        raise FileExistsError(f"{output_dir} already holds {existing or ['trace/']}; runs never overwrite artifacts")
    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    config = OmegaConf.to_container(cfg, resolve=True)
    validate_config(cfg)
    world_size = int(cfg.distributed.world_size)
    started, error = time.perf_counter(), None
    try:
        mp.spawn(_worker, args=(config, str(output_dir), _free_port()), nprocs=world_size, join=True)
    except Exception as exc:  # the rank reports carry the cause; re-raised after the artifacts are written
        error = exc
    ranks = []
    for rank in range(world_size):
        path = output_dir / f"rank{rank}_report.json"
        ranks.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
    failures = [r["failure"] for r in ranks if r and r.get("failure")]
    status = "completed" if error is None and all(r and r["status"] == "completed" for r in ranks) else "failed"
    if error is not None and not failures:
        failures.append({"rank": None, "stage": "spawn", "step": None, "type": type(error).__name__,
                         "message": str(error)})
    report = {
        "schema_version": RUN_SCHEMA, "implementation": cfg.implementation, "run_kind": cfg.run_kind,
        "gate": cfg.gate, "status": status, "output_dir": str(output_dir),
        "elapsed_seconds": time.perf_counter() - started, "max_steps": int(cfg.training.max_steps),
        "completed_steps": min((len(r["history"]) for r in ranks if r), default=0),
        "config_identity_sha256": config_identity_sha256(config),
        "tolerance": {"rtol": float(cfg.verification.rtol), "atol": float(cfg.verification.atol),
                      "status": "provisional (spec §5.3)"},
        "failures": failures,
        "missing_rank_reports": [rank for rank, r in enumerate(ranks) if r is None],
        "verification": aggregate_checks(ranks) if cfg.run_kind == "parity" else {},
        "resources": [r.get("resources") if r else None for r in ranks],
        "comet": [r.get("comet") if r else None for r in ranks],
        "checkpoints": ranks[0]["checkpoints"] if ranks[0] else [],
        "ranks": [{k: r[k] for k in ("rank", "status", "history")} if r else None for r in ranks],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance(cfg, ranks, output_dir), indent=2,
                                                           ensure_ascii=False, default=str), encoding="utf-8")
    (output_dir / "fixture_manifest.json").write_text(json.dumps(fixture_manifest(cfg, ranks), indent=2,
                                                                 ensure_ascii=False, default=str), encoding="utf-8")
    (output_dir / "resource_benchmark.json").write_text(json.dumps(
        {"run_kind": cfg.run_kind, "status": status,
         "warmup_steps": int(cfg.bench.warmup_steps) if cfg.run_kind == "bench" else 0,
         "note": "parity: VRAM and step time include the phase-tracing copies; bench (A5) measures training only",
         "failures": failures, "ranks": report["resources"]}, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["checkpoints"]:
        (output_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest(
            ADAPTERS[cfg.implementation], config, output_dir, report["checkpoints"]), indent=2), encoding="utf-8")
    report["artifacts"] = sorted(p.name for p in output_dir.iterdir()) + ["run_report.json", "verification_summary.md"]
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                                encoding="utf-8")
    (output_dir / "verification_summary.md").write_text(render_summary(report), encoding="utf-8")
    if error is not None:
        raise error
    print(f"Saved run artifacts in {output_dir}")
    return report


@hydra.main(version_base="1.3", config_path="../../conf", config_name="moco_imagenet_parity")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
