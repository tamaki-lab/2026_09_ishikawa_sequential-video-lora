"""Gate A1 (self side): DistributedMoCo on 2 CPU ranks (gloo) against an independent float64 reference.

The reference re-derives the fixed facebookresearch/moco builder semantics in one process: EMA over parameters,
global key gather, one rank-0 permutation, per-rank key BN on the shuffled slice, unshuffle, logits against the
queue before the step, enqueue of all ranks' keys in rank order, and DDP-averaged query gradients.
"""

import copy
import socket
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.models import resnet50

from self_supervised.moco import DistributedMoCo, ImageNetResNet50, StandaloneMoCo
from self_supervised.moco import ddp_moco

WORLD_SIZE, LOCAL_BATCH, DIM, K, M, T = 2, 4, 16, 16, 0.9, 0.2
SHUFFLE_SEED = 1234
TOL = {"rtol": 1e-9, "atol": 1e-12}


class TinyBackbone(nn.Module):
    """Small conv + BatchNorm backbone exposing ``feature_dim``."""

    feature_dim = 8

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, images):
        return self.net(images)


def build(cls, queue_size=K):
    torch.manual_seed(0)
    return cls(TinyBackbone(), dim=DIM, K=queue_size, m=M, T=T).to(torch.float64)


def rank_views(step, rank, batch_size=LOCAL_BATCH):
    generator = torch.Generator().manual_seed(100 * step + rank)
    return tuple(torch.randn(batch_size, 3, 6, 6, generator=generator, dtype=torch.float64) for _ in range(2))


def shuffle_seed(step, rank):
    return SHUFFLE_SEED + 10 * step + rank  # rank 1 draws a different permutation that rank 0's broadcast replaces


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _entry(rank, scenario, port, out_dir, kwargs):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=WORLD_SIZE, rank=rank,
                            timeout=timedelta(seconds=60))
    try:
        torch.save(scenario(rank, **kwargs), Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def run_ranks(tmp_path, scenario, **kwargs):
    mp.spawn(_entry, args=(scenario, _free_port(), str(tmp_path), kwargs), nprocs=WORLD_SIZE, join=True)
    return [torch.load(tmp_path / f"rank{rank}.pt", weights_only=False) for rank in range(WORLD_SIZE)]


def _named_bn_buffers(model):
    return {name: buffer.clone() for name, buffer in model.named_buffers() if "queue" not in name
            and name != "real_key_count"}


def scenario_train(rank, steps):
    model = build(DistributedMoCo)
    ddp = DistributedDataParallel(model)  # broadcast_buffers=True as in the reference main_moco.py
    optimizer = torch.optim.SGD(model.query_parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    at_forward = []
    model.register_forward_pre_hook(lambda module, args: at_forward.append(
        {"buffers": _named_bn_buffers(module), "queue": module.queue.clone(), "ptr": int(module.queue_ptr)}))
    model.train()
    records = []
    for step in range(1, steps + 1):
        im_q, im_k = rank_views(step, rank)
        torch.manual_seed(shuffle_seed(step, rank))
        out = ddp(im_q, im_k)
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        grads = [p.grad.clone() for p in model.query_parameters()]
        key_after_ema = [p.clone() for p in model.key_parameters()]
        optimizer.step()
        records.append({
            "q": out.q.detach().clone(), "k": out.k.clone(), "logits": out.logits.detach().clone(),
            "loss": out.loss.detach().clone(), "targets": out.targets.clone(), "grads": grads,
            "key_after_ema": key_after_ema, "queue": model.queue.clone(), "ptr": int(model.queue_ptr),
            "real_key_count": int(model.real_key_count), "buffers_after_step": _named_bn_buffers(model),
            "at_forward": at_forward[-1], "query_after_sgd": [p.detach().clone() for p in model.query_parameters()],
        })
    return records


def reference_first_step():
    """Single-process float64 derivation of step 1 for both ranks from the shared initial state."""
    initial = build(StandaloneMoCo)
    views = [rank_views(1, rank) for rank in range(WORLD_SIZE)]
    perm = torch.randperm(WORLD_SIZE * LOCAL_BATCH, generator=torch.Generator().manual_seed(shuffle_seed(1, 0)))
    global_im_k = torch.cat([v[1] for v in views])
    queue0 = initial.queue.clone()
    pairs = zip(initial.query_parameters(), initial.key_parameters())
    ema = [k.detach() * M + q.detach() * (1.0 - M) for q, k in pairs]

    models, queries, shuffled_keys = [], [], []
    for rank in range(WORLD_SIZE):
        model = copy.deepcopy(initial)
        model.train()
        with torch.no_grad():
            for param, value in zip(model.key_parameters(), ema):
                param.copy_(value)
        queries.append(F.normalize(model.projector_q(model.backbone_q(views[rank][0])), dim=1))
        with torch.no_grad():
            rows = perm.view(WORLD_SIZE, -1)[rank]  # shuffled position i holds global sample perm[i]
            shuffled_keys.append(F.normalize(model.projector_k(model.backbone_k(global_im_k[rows])), dim=1))
        models.append(model)
    keys = torch.empty(WORLD_SIZE * LOCAL_BATCH, DIM, dtype=torch.float64)
    keys[perm] = torch.cat(shuffled_keys)

    per_rank, grads = [], []
    for rank, (model, q) in enumerate(zip(models, queries)):
        k = keys[rank * LOCAL_BATCH:(rank + 1) * LOCAL_BATCH]
        logits = torch.cat([(q * k).sum(dim=1, keepdim=True), q @ queue0], dim=1) / T
        loss = F.cross_entropy(logits, torch.zeros(LOCAL_BATCH, dtype=torch.long))
        grads.append(torch.autograd.grad(loss, model.query_parameters()))
        per_rank.append({"q": q.detach(), "k": k, "logits": logits.detach(), "loss": loss.detach(),
                         "buffers": _named_bn_buffers(model)})
    queue_after = queue0.clone()
    queue_after[:, :WORLD_SIZE * LOCAL_BATCH] = keys.T
    mean_grads = [(a + b) / 2 for a, b in zip(*grads)]
    return per_rank, mean_grads, ema, queue0, queue_after


def test_imagenet_resnet50_uses_random_init_standard_stem_and_2048d_feature():
    with patch.object(ddp_moco, "resnet50", wraps=resnet50) as factory:
        backbone = ImageNetResNet50()
    factory.assert_called_once_with(weights=None)
    conv = backbone.net.conv1
    assert (conv.kernel_size, conv.stride, conv.padding) == ((7, 7), (2, 2), (3, 3))
    assert isinstance(backbone.net.maxpool, nn.MaxPool2d) and isinstance(backbone.net.fc, nn.Identity)
    assert backbone(torch.randn(2, 3, 64, 64)).shape == (2, 2048) and backbone.feature_dim == 2048
    projector = DistributedMoCo(backbone).projector_q
    assert [(layer.in_features, layer.out_features) for layer in projector if isinstance(layer, nn.Linear)] == \
        [(2048, 2048), (2048, 128)]


def test_a1_first_ddp_step_matches_the_independent_shuffle_bn_reference(tmp_path):
    ranks = run_ranks(tmp_path, scenario_train, steps=1)
    per_rank, mean_grads, ema, queue0, queue_after = reference_first_step()

    for rank, (record,) in enumerate(ranks):
        expected = per_rank[rank]
        for name in ("q", "k", "logits", "loss"):
            torch.testing.assert_close(record[name], expected[name], **TOL, msg=f"rank {rank} {name}")
        assert record["targets"].dtype == torch.long and record["targets"].tolist() == [0] * LOCAL_BATCH
        for got, want in zip(record["grads"], mean_grads):  # DDP all-reduce averages the two ranks
            torch.testing.assert_close(got, want, **TOL)
        for got, want in zip(record["key_after_ema"], ema):  # EMA over parameters with the pre-SGD query
            torch.testing.assert_close(got, want, **TOL)
        # Logits use the queue before this step's enqueue; afterwards all ranks hold the same global keys.
        assert torch.equal(record["at_forward"]["queue"], queue0)
        torch.testing.assert_close(record["queue"], queue_after, **TOL)
        assert record["ptr"] == record["real_key_count"] == WORLD_SIZE * LOCAL_BATCH
        # BN running stats: query BN saw this rank's im_q, key BN saw this rank's shuffled slice.
        for name, buffer in expected["buffers"].items():
            torch.testing.assert_close(record["buffers_after_step"][name], buffer, **TOL, msg=name)
    assert torch.equal(ranks[0][0]["queue"], ranks[1][0]["queue"])
    for a, b in zip(ranks[0][0]["query_after_sgd"], ranks[1][0]["query_after_sgd"]):
        assert torch.equal(a, b)


def test_a1_queue_wraps_identically_and_ddp_rebroadcasts_rank0_buffers(tmp_path):
    ranks = run_ranks(tmp_path, scenario_train, steps=3)
    global_batch = WORLD_SIZE * LOCAL_BATCH
    assert [r["ptr"] for r in ranks[0]] == [r["ptr"] for r in ranks[1]] == [8, 0, 8]
    assert [r["real_key_count"] for r in ranks[0]] == [8, 16, 16]
    for step in range(3):
        assert torch.equal(ranks[0][step]["queue"], ranks[1][step]["queue"])
        keys = torch.cat([ranks[rank][step]["k"] for rank in range(WORLD_SIZE)])
        start = (step * global_batch) % K
        assert torch.equal(ranks[0][step]["queue"][:, start:start + global_batch], keys.T)  # rank order
        # The positive column pairs q with its own key; the negatives are the queue before this step.
        record = ranks[0][step]
        expected = record["q"] @ record["at_forward"]["queue"] / T
        torch.testing.assert_close(record["logits"][:, 1:], expected, **TOL)
    # Step 3 overwrote the oldest (step 1) keys and kept the step-2 keys.
    assert torch.equal(ranks[0][2]["queue"][:, 8:], ranks[0][1]["queue"][:, 8:])

    # Local BN statistics differ after a step; DDP's forward broadcast installs rank 0's on every rank.
    rank0_end, rank1_end = ranks[0][0]["buffers_after_step"], ranks[1][0]["buffers_after_step"]
    running = [name for name in rank0_end if "running_" in name]
    assert running and all(not torch.equal(rank0_end[name], rank1_end[name]) for name in running)
    for rank in range(WORLD_SIZE):
        at_step2 = ranks[rank][1]["at_forward"]["buffers"]
        assert all(torch.equal(at_step2[name], rank0_end[name]) for name in rank0_end)


def scenario_invalid(rank, case):
    queue_size = 12 if case == "global_batch" else K
    model = build(DistributedMoCo, queue_size=queue_size)
    ddp = DistributedDataParallel(model)
    key_before = [p.clone() for p in model.key_parameters()]
    im_q, im_k = rank_views(1, rank, batch_size=2 if case == "batch_mismatch" and rank == 1 else LOCAL_BATCH)
    if case == "non_finite" and rank == 1:
        im_k = im_k.clone()
        im_k[0, 0, 0, 0] = float("nan")
    try:
        ddp(im_q, im_k)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return {"error": type(error).__name__, "message": str(error), "ptr": int(model.queue_ptr),
                "key_unchanged": all(torch.equal(a, b) for a, b in zip(key_before, model.key_parameters()))}
    return {"error": None}


@pytest.mark.parametrize("case, errors", [
    ("batch_mismatch", ["ValueError", "ValueError"]),
    ("non_finite", ["RuntimeError", "FloatingPointError"]),
    ("global_batch", ["ValueError", "ValueError"]),
])
def test_a1_invalid_step_raises_on_every_rank_before_any_state_change(tmp_path, case, errors):
    ranks = run_ranks(tmp_path, scenario_invalid, case=case)
    assert [r["error"] for r in ranks] == errors, ranks
    assert all(r["ptr"] == 0 and r["key_unchanged"] for r in ranks)
    if case == "non_finite":
        assert "rank(s) [1]" in ranks[0]["message"]
    if case == "global_batch":
        assert "global batch size 8" in ranks[0]["message"]
