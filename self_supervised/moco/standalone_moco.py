"""Single-device MoCo v2-style mechanics for two-view image contrastive learning.

Adapted from the fixed reference facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
(``moco/builder.py``, Copyright (c) Facebook, Inc. and its affiliates, CC-BY-NC 4.0).
Differences from the reference: one device only (``concat_all_gather`` is the identity and the DDP
batch shuffle is not implemented), the backbone is passed in and exposes ``feature_dim``, the MLP
projector is a separate module, the InfoNCE loss is computed here, and invalid states raise.

One ``forward`` call keeps the reference order:
query forward -> key EMA with the not-yet-updated query -> key forward ->
logits against a snapshot of the queue -> enqueue the current keys.
The caller then runs backward and the optimizer step on ``query_parameters()`` only.
"""

import copy
import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

UNIT_NORM_ATOL = 1e-4


class CifarResNet18(nn.Module):
    """Randomly initialized ResNet18 for 32x32 inputs: 3x3 stride-1 stem, no max-pool, no classifier.

    Returns the 512-d global-average-pooled feature.
    """

    feature_dim = 512

    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.net = net

    def forward(self, images):
        return self.net(images)


def projection_head(feature_dim: int, dim: int) -> nn.Sequential:
    """MoCo v2 MLP head: Linear(feature_dim, feature_dim) -> ReLU -> Linear(feature_dim, dim)."""
    return nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, dim))


def moco_logits(q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor, temperature: float):
    """Return (logits [B, 1+K] already divided by T, positive cosine [B, 1], negative cosine [B, K])."""
    l_pos = (q * k).sum(dim=1, keepdim=True)
    l_neg = q @ queue
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    return logits, l_pos, l_neg


def _require_finite(**tensors: torch.Tensor) -> None:
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"non-finite {name}")


@dataclass
class MoCoOutput:
    loss: torch.Tensor
    logits: torch.Tensor           # [B, 1+K], divided by T; column 0 is the positive
    targets: torch.Tensor          # [B], all zeros
    q: torch.Tensor                # [B, dim], unit norm
    k: torch.Tensor                # [B, dim], unit norm, no autograd graph
    z_q: torch.Tensor              # query projector output before L2 normalization
    z_k: torch.Tensor              # key projector output before L2 normalization
    positive_cosine: torch.Tensor  # [B, 1], detached
    negative_cosine: torch.Tensor  # [B, K], detached, against the queue before this step's enqueue


class StandaloneMoCo(nn.Module):
    """Query/key backbone + projector, key EMA, FIFO queue of random unit keys, InfoNCE."""

    def __init__(self, backbone: nn.Module, dim: int = 128, K: int = 1024, m: float = 0.999, T: float = 0.2):
        super().__init__()
        if dim < 1 or K < 1:
            raise ValueError(f"dim and K must be positive, got dim={dim}, K={K}")
        if not 0.0 <= m <= 1.0:
            raise ValueError(f"momentum m must be in [0, 1], got {m}")
        if T <= 0:
            raise ValueError(f"temperature T must be positive, got {T}")
        self.feature_dim = int(backbone.feature_dim)
        self.dim, self.K, self.m, self.T = dim, K, m, T

        self.backbone_q = backbone
        self.projector_q = projection_head(self.feature_dim, dim)
        # Identical initial state (parameters and BN buffers) in separate storage.
        self.backbone_k = copy.deepcopy(self.backbone_q)
        self.projector_k = copy.deepcopy(self.projector_q)
        for param in self.key_parameters():
            param.requires_grad = False

        # K random unit columns from the start: no empty queue and no key-only warm-up.
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("real_key_count", torch.zeros(1, dtype=torch.long))

    def query_parameters(self) -> list[nn.Parameter]:
        """Optimizer targets: query backbone + query projector."""
        return list(itertools.chain(self.backbone_q.parameters(), self.projector_q.parameters()))

    def key_parameters(self) -> list[nn.Parameter]:
        return list(itertools.chain(self.backbone_k.parameters(), self.projector_k.parameters()))

    @property
    def random_key_remaining(self) -> int:
        return self.K - int(self.real_key_count)

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """theta_k <- m * theta_k + (1 - m) * theta_q over parameters only; BN buffers are not touched."""
        for param_q, param_k in zip(self.query_parameters(), self.key_parameters(), strict=True):
            param_k.copy_(param_k * self.m + param_q.detach() * (1.0 - self.m))

    @torch.no_grad()
    def check_queue_state(self, batch_size: int | None = None) -> None:
        """Raise unless the queue holds K finite unit columns and pointer/counters are consistent."""
        if tuple(self.queue.shape) != (self.dim, self.K):
            raise RuntimeError(f"queue shape {tuple(self.queue.shape)} != {(self.dim, self.K)}")
        _require_finite(queue=self.queue)
        norm_error = float((self.queue.norm(dim=0) - 1.0).abs().max())
        if norm_error > UNIT_NORM_ATOL:
            raise RuntimeError(f"queue columns are not unit norm (max error {norm_error:.3g})")
        ptr, real = int(self.queue_ptr), int(self.real_key_count)
        if not 0 <= ptr < self.K or not 0 <= real <= self.K:
            raise RuntimeError(f"queue_ptr={ptr} or real_key_count={real} outside [0, K={self.K}]")
        # FIFO starts at column 0, so until the queue is full the real keys are columns [0, ptr).
        if real < self.K and ptr != real:
            raise RuntimeError(f"queue_ptr={ptr} disagrees with real_key_count={real}")
        if batch_size is not None and ptr % batch_size:
            raise RuntimeError(f"queue_ptr={ptr} is not a multiple of batch size {batch_size}")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Overwrite the oldest batch_size columns at queue_ptr (all_gather is the identity on one device)."""
        if keys.ndim != 2 or keys.shape[1] != self.dim:
            raise ValueError(f"keys must be [B, {self.dim}], got {tuple(keys.shape)}")
        if keys.dtype != self.queue.dtype:
            raise TypeError(f"keys dtype {keys.dtype} != queue dtype {self.queue.dtype}")
        batch_size = keys.shape[0]
        if batch_size < 1 or self.K % batch_size:
            raise ValueError(f"queue size K={self.K} must be a multiple of batch size {batch_size}")
        _require_finite(keys=keys)
        if float((keys.norm(dim=1) - 1.0).abs().max()) > UNIT_NORM_ATOL:
            raise ValueError("keys must be L2-normalized before enqueue")
        self.check_queue_state(batch_size)
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.detach().T
        self.queue_ptr[0] = (ptr + batch_size) % self.K
        self.real_key_count[0] = min(self.K, int(self.real_key_count) + batch_size)

    def _project(self, backbone: nn.Module, projector: nn.Module, images: torch.Tensor) -> torch.Tensor:
        features = backbone(images)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"backbone output must be [B, {self.feature_dim}], got {tuple(features.shape)}")
        return projector(features)

    def encode_query(self, images: torch.Tensor) -> torch.Tensor:
        """Normalized query feature without EMA or enqueue (for evaluation and checkpoint comparison)."""
        return F.normalize(self._project(self.backbone_q, self.projector_q, images), dim=1)

    @torch.no_grad()
    def encode_key(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self._project(self.backbone_k, self.projector_k, images), dim=1)

    def _check_views(self, im_q: torch.Tensor, im_k: torch.Tensor) -> None:
        if im_q.ndim != 4 or im_q.shape != im_k.shape:
            raise ValueError(f"im_q and im_k must share one [B, C, H, W] shape, "
                             f"got {tuple(im_q.shape)} and {tuple(im_k.shape)}")
        if im_q.dtype != self.queue.dtype or im_k.dtype != self.queue.dtype:
            raise TypeError(f"views must be {self.queue.dtype}, got {im_q.dtype} / {im_k.dtype}")
        if self.K % im_q.shape[0]:
            raise ValueError(f"queue size K={self.K} must be a multiple of batch size {im_q.shape[0]}")
        _require_finite(im_q=im_q, im_k=im_k)

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> MoCoOutput:
        # Validate before any state changes (EMA, BN buffers, queue).
        self._check_views(im_q, im_k)
        self.check_queue_state(im_q.shape[0])

        z_q = self._project(self.backbone_q, self.projector_q, im_q)
        q = F.normalize(z_q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            # Key BN layers stay in the module's train/eval mode; no_grad does not stop running-stat updates.
            z_k = self._project(self.backbone_k, self.projector_k, im_k)
            k = F.normalize(z_k, dim=1)
        _require_finite(q=q, k=k)

        queue_snapshot = self.queue.clone().detach()
        logits, l_pos, l_neg = moco_logits(q, k, queue_snapshot, self.T)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, targets)
        _require_finite(logits=logits, loss=loss)

        # Current keys become negatives only from the next step; the logits above use the snapshot.
        self._dequeue_and_enqueue(k)
        return MoCoOutput(
            loss=loss, logits=logits, targets=targets, q=q, k=k, z_q=z_q, z_k=z_k,
            positive_cosine=l_pos.detach(), negative_cosine=l_neg.detach(),
        )
