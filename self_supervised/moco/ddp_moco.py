"""MoCo v2 under torch.distributed: ImageNet ResNet50, shuffle-BN and cross-rank key gather (Spec A, 2026-10-17).

Re-implements the multi-GPU parts of the fixed reference facebookresearch/moco@a419ef282fc9885015674e77f58c59447206a27f
(``moco/builder.py``; the LICENSE file and the file header say MIT, the README says CC-BY-NC 4.0). The reference
source is not copied here; it is kept as an independent clone and compared by ``scripts/parity``.

One ``forward`` call keeps the reference order:
query forward -> key EMA with the not-yet-updated query -> gather key images from every rank -> permute them with
one permutation drawn by rank 0 from the CPU global RNG (``torch.randperm``) and broadcast -> key forward on this
rank's shuffled slice -> L2 normalize -> restore the original order -> logits against the queue before this step ->
enqueue the keys of all ranks in rank order. BatchNorm stays plain ``nn.BatchNorm2d`` (no SyncBatchNorm).
The caller wraps the module in DistributedDataParallel and steps an optimizer over ``query_parameters()`` only.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50

from .standalone_moco import MoCoOutput, StandaloneMoCo, _require_finite, moco_logits


class ImageNetResNet50(nn.Module):
    """Randomly initialized torchvision ResNet50 (7x7 stride-2 stem, max-pool) without classifier.

    Returns the 2048-d global-average-pooled feature.
    """

    feature_dim = 2048

    def __init__(self):
        super().__init__()
        net = resnet50(weights=None)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, images):
        return self.net(images)


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate ``tensor`` from every rank along dim 0 in rank order (no autograd)."""
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def _check_on_all_ranks(device: torch.device, check, *args, **kwargs) -> None:
    """Run ``check`` on every rank; if it raises on any rank, raise on every rank.

    A rank that raised alone would leave the other ranks waiting in the next collective.
    """
    error = None
    try:
        check(*args, **kwargs)
    except (ValueError, TypeError, RuntimeError, FloatingPointError) as exc:
        error = exc
    flags = concat_all_gather(torch.tensor([error is not None], dtype=torch.int32, device=device))
    if error is not None:
        raise error
    failed = flags.nonzero().flatten().tolist()
    if failed:
        raise RuntimeError(f"rank(s) {failed} failed the check {check.__name__}")


class DistributedMoCo(StandaloneMoCo):
    """StandaloneMoCo with the reference DDP mechanics: shuffle-BN key forward and enqueue of all ranks' keys.

    Needs an initialized default process group; every rank calls ``forward`` with the same local batch size.
    The queue receives ``world_size * local_batch`` keys per step, so K must be a multiple of the global batch.
    """

    @torch.no_grad()
    def _batch_shuffle(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this rank's slice of the globally permuted key images and the inverse permutation."""
        gathered = concat_all_gather(images)
        world_size, rank = dist.get_world_size(), dist.get_rank()
        idx_shuffle = torch.randperm(gathered.shape[0]).to(images.device)
        dist.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        return gathered[idx_shuffle.view(world_size, -1)[rank]], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, features: torch.Tensor, idx_unshuffle: torch.Tensor) -> torch.Tensor:
        gathered = concat_all_gather(features)
        return gathered[idx_unshuffle.view(dist.get_world_size(), -1)[dist.get_rank()]]

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Enqueue the keys of every rank (rank order), so the queue and pointer stay identical across ranks."""
        super()._dequeue_and_enqueue(concat_all_gather(keys))

    def _check_step_inputs(self, im_q: torch.Tensor, im_k: torch.Tensor) -> None:
        self._check_views(im_q, im_k)
        global_batch = im_q.shape[0] * dist.get_world_size()
        if self.K % global_batch:
            raise ValueError(f"queue size K={self.K} must be a multiple of the global batch size {global_batch}")
        self.check_queue_state(global_batch)

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> MoCoOutput:
        device = im_q.device
        # Validate before any state changes (EMA, BN buffers, queue); equal local batches keep all_gather valid.
        batch_sizes = concat_all_gather(torch.tensor([im_q.shape[0]], device=device))
        if not bool((batch_sizes == batch_sizes[0]).all()):
            raise ValueError(f"local batch sizes differ across ranks: {batch_sizes.tolist()}")
        _check_on_all_ranks(device, self._check_step_inputs, im_q, im_k)

        z_q = self._project(self.backbone_q, self.projector_q, im_q)
        q = F.normalize(z_q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            # Key BN layers stay in the module's train/eval mode and see only this rank's shuffled slice.
            shuffled, idx_unshuffle = self._batch_shuffle(im_k)
            z_k_shuffled = self._project(self.backbone_k, self.projector_k, shuffled)
            k = self._batch_unshuffle(F.normalize(z_k_shuffled, dim=1), idx_unshuffle)
            z_k = self._batch_unshuffle(z_k_shuffled, idx_unshuffle)
        _check_on_all_ranks(device, _require_finite, q=q, k=k)

        queue_snapshot = self.queue.clone().detach()
        logits, l_pos, l_neg = moco_logits(q, k, queue_snapshot, self.T)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, targets)
        _check_on_all_ranks(device, _require_finite, logits=logits, loss=loss)

        # Current keys become negatives only from the next step; the logits above use the snapshot.
        self._dequeue_and_enqueue(k)
        return MoCoOutput(
            loss=loss, logits=logits, targets=targets, q=q, k=k, z_q=z_q, z_k=z_k,
            positive_cosine=l_pos.detach(), negative_cosine=l_neg.detach(),
        )
