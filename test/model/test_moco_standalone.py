"""Gate A: MoCo v2-style mechanics (Q/K init, logits, InfoNCE, queue, EMA, step order, BN, invalid input)."""

import copy
import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from self_supervised.moco import CifarResNet18, StandaloneMoCo, moco_logits
from self_supervised.moco import standalone_moco


class TinyBackbone(nn.Module):
    """Small conv + BatchNorm backbone exposing ``feature_dim`` like CifarResNet18."""

    feature_dim = 8

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, images):
        return self.net(images)


def tiny_moco(dim=16, K=8, m=0.9, T=0.2, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return StandaloneMoCo(TinyBackbone(), dim=dim, K=K, m=m, T=T).to(dtype)


def views(batch_size=4, seed=1, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(batch_size, 3, 8, 8, generator=generator, dtype=dtype) for _ in range(2))


def unit_keys(batch_size, dim, seed):
    return F.normalize(torch.randn(batch_size, dim, generator=torch.Generator().manual_seed(seed)), dim=1)


def float64_ema(m, key_before, query_before):
    return [k.double() * m + q.double() * (1.0 - m) for k, q in zip(key_before, query_before)]


def test_cifar_resnet18_uses_random_init_small_stem_and_512d_feature():
    with patch.object(standalone_moco, "resnet18", wraps=resnet18) as factory:
        backbone = CifarResNet18()
    factory.assert_called_once_with(weights=None)
    conv = backbone.net.conv1
    assert (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding) == \
        (3, 64, (3, 3), (1, 1), (1, 1))
    assert conv.bias is None
    assert isinstance(backbone.net.maxpool, nn.Identity) and isinstance(backbone.net.fc, nn.Identity)
    assert backbone(torch.randn(2, 3, 32, 32)).shape == (2, 512) and backbone.feature_dim == 512

    projector = StandaloneMoCo(backbone).projector_q
    assert [type(layer) for layer in projector] == [nn.Linear, nn.ReLU, nn.Linear]
    assert (projector[0].in_features, projector[0].out_features) == (512, 512)
    assert (projector[2].in_features, projector[2].out_features) == (512, 128)


@pytest.mark.parametrize("make_model", [tiny_moco, lambda: StandaloneMoCo(CifarResNet18())])
def test_a01_query_key_start_identical_in_separate_storage_and_key_is_frozen(make_model):
    model = make_model()
    for query, key in ((model.backbone_q, model.backbone_k), (model.projector_q, model.projector_k)):
        query_state, key_state = query.state_dict(), key.state_dict()
        assert query_state.keys() == key_state.keys()
        for name in query_state:  # parameters and BN buffers
            assert torch.equal(query_state[name], key_state[name])
            assert query_state[name].untyped_storage().data_ptr() != key_state[name].untyped_storage().data_ptr()
    assert all(not p.requires_grad for p in model.key_parameters())
    assert all(p.requires_grad for p in model.query_parameters())
    assert len(model.query_parameters()) == len(model.key_parameters())

    optimizer = torch.optim.SGD(model.query_parameters(), lr=0.1)
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert set(optimizer_ids) == {id(p) for p in model.query_parameters()}
    assert not set(optimizer_ids) & {id(p) for p in model.key_parameters()}


def test_forward_features_have_expected_shape_norm_and_autograd_boundary():
    model = tiny_moco()
    out = model(*views())
    assert out.q.shape == out.k.shape == (4, 16) and out.logits.shape == (4, 9)
    torch.testing.assert_close(out.q.norm(dim=1), torch.ones(4))
    torch.testing.assert_close(out.k.norm(dim=1), torch.ones(4))
    assert out.q.grad_fn is not None and out.loss.grad_fn is not None
    assert out.k.grad_fn is None and not out.k.requires_grad
    assert not out.z_k.requires_grad and out.z_q.requires_grad
    assert not torch.allclose(out.z_q.norm(dim=1), torch.ones(4))  # z_q/z_k are the pre-L2 projections
    assert all(torch.isfinite(t).all() for t in (out.q, out.k, out.logits, out.loss))
    assert not model.queue.requires_grad


def test_a03_positive_and_negative_logits_match_hand_computation():
    generator = torch.Generator().manual_seed(0)
    q = F.normalize(torch.randn(3, 4, generator=generator, dtype=torch.float64), dim=1)
    k = F.normalize(torch.randn(3, 4, generator=generator, dtype=torch.float64), dim=1)
    queue = F.normalize(torch.randn(4, 5, generator=generator, dtype=torch.float64), dim=0)
    logits, l_pos, l_neg = moco_logits(q, k, queue, 0.2)

    assert logits.shape == (3, 6) and l_pos.shape == (3, 1) and l_neg.shape == (3, 5)
    for i in range(3):
        positive = sum(a * b for a, b in zip(q[i].tolist(), k[i].tolist()))
        assert math.isclose(logits[i, 0].item(), positive / 0.2, rel_tol=1e-12, abs_tol=1e-12)
        for j in range(5):
            negative = sum(a * b for a, b in zip(q[i].tolist(), queue[:, j].tolist()))
            assert math.isclose(logits[i, 1 + j].item(), negative / 0.2, rel_tol=1e-12, abs_tol=1e-12)

    out = tiny_moco()(*views())
    assert out.targets.dtype == torch.long and out.targets.tolist() == [0, 0, 0, 0]


def test_a04_info_nce_matches_independent_float64_reference():
    model = tiny_moco(dtype=torch.float64)
    queue_before = model.queue.clone()
    out = model(*views(dtype=torch.float64))
    q, k = out.q.detach(), out.k

    expected = 0.0
    for i in range(4):
        positive = sum(a * b for a, b in zip(q[i].tolist(), k[i].tolist())) / model.T
        negatives = [sum(a * b for a, b in zip(q[i].tolist(), queue_before[:, j].tolist())) / model.T
                     for j in range(model.K)]
        log_sum_exp = math.log(sum(math.exp(value) for value in [positive, *negatives]))
        expected += (log_sum_exp - positive) / 4
    assert math.isclose(out.loss.item(), expected, rel_tol=1e-8, abs_tol=1e-10)
    reference = F.cross_entropy(out.logits.detach(), torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(out.loss.detach(), reference, rtol=1e-8, atol=1e-10)


def test_a05_initial_queue_holds_k_random_unit_columns_reproducibly():
    def build(seed):
        torch.manual_seed(seed)
        return StandaloneMoCo(TinyBackbone(), dim=128, K=1024)

    model = build(0)
    assert model.queue.shape == (128, 1024) and torch.isfinite(model.queue).all()
    torch.testing.assert_close(model.queue.norm(dim=0), torch.ones(1024), rtol=0, atol=1e-6)
    assert (int(model.queue_ptr), int(model.real_key_count), model.random_key_remaining) == (0, 0, 1024)
    assert {"queue", "queue_ptr", "real_key_count"} <= dict(model.named_buffers()).keys()
    assert torch.equal(build(0).queue, model.queue)
    assert not torch.equal(build(1).queue, model.queue)
    # No warm-up: the very first step already uses all K random columns as negatives.
    out = model(*views(batch_size=32))
    assert out.logits.shape == (32, 1025)


def test_a06_fifo_k8_b4_wraparound_overwrites_the_oldest_keys():
    model = tiny_moco(dim=16, K=8)
    initial = model.queue.clone()
    a, b, c = (unit_keys(4, 16, seed) for seed in (1, 2, 3))

    model._dequeue_and_enqueue(a)
    assert torch.equal(model.queue[:, :4], a.T) and torch.equal(model.queue[:, 4:], initial[:, 4:])
    assert (int(model.queue_ptr), int(model.real_key_count), model.random_key_remaining) == (4, 4, 4)
    model._dequeue_and_enqueue(b)
    assert torch.equal(model.queue[:, :4], a.T) and torch.equal(model.queue[:, 4:], b.T)
    assert (int(model.queue_ptr), int(model.real_key_count), model.random_key_remaining) == (0, 8, 0)
    model._dequeue_and_enqueue(c)
    assert torch.equal(model.queue[:, :4], c.T) and torch.equal(model.queue[:, 4:], b.T)
    assert (int(model.queue_ptr), int(model.real_key_count), model.random_key_remaining) == (4, 8, 0)


def test_a06_fifo_k1024_b32_pointer_and_real_key_count_transitions():
    torch.manual_seed(0)
    model = StandaloneMoCo(TinyBackbone(), dim=128, K=1024)
    initial = model.queue.clone()
    for n in range(1, 34):
        keys = unit_keys(32, 128, n)
        start = ((n - 1) * 32) % 1024
        model._dequeue_and_enqueue(keys)
        assert torch.equal(model.queue[:, start:start + 32], keys.T)
        assert int(model.queue_ptr) == (n * 32) % 1024
        assert int(model.real_key_count) == min(1024, n * 32)
        if n == 20:
            assert (int(model.queue_ptr), int(model.real_key_count), model.random_key_remaining) == (640, 640, 384)
            assert torch.equal(model.queue[:, 640:], initial[:, 640:])
        if n == 32:
            assert (int(model.queue_ptr), int(model.real_key_count), model.random_key_remaining) == (0, 1024, 0)
    model.check_queue_state(32)


def test_a07_current_keys_enter_the_queue_only_after_the_logits():
    model = tiny_moco()
    reference = copy.deepcopy(model)
    queue_before = model.queue.clone()
    im_q, im_k = views()

    out = model(im_q, im_k)
    q = out.q.detach()
    torch.testing.assert_close(out.logits[:, 1:].detach(), q @ queue_before / model.T)
    torch.testing.assert_close(out.negative_cosine, q @ queue_before)
    torch.testing.assert_close(out.logits[:, 0].detach(), (q * out.k).sum(dim=1) / model.T)
    # This step's keys are in the queue for the next step, not among this step's negatives.
    assert torch.equal(model.queue[:, :4], out.k.T)
    assert not torch.allclose(out.logits[:, 1:].detach(), q @ model.queue / model.T)

    out.loss.backward()  # no autograd version-counter error despite the in-place enqueue

    q_ref = F.normalize(reference.projector_q(reference.backbone_q(im_q)), dim=1)
    with torch.no_grad():
        reference._momentum_update_key_encoder()
        k_ref = F.normalize(reference.projector_k(reference.backbone_k(im_k)), dim=1)
    logits_ref = torch.cat([(q_ref * k_ref).sum(dim=1, keepdim=True), q_ref @ queue_before], dim=1) / reference.T
    loss_ref = F.cross_entropy(logits_ref, torch.zeros(4, dtype=torch.long))
    loss_ref.backward()
    torch.testing.assert_close(out.loss.detach(), loss_ref.detach())
    for param, param_ref in zip(model.query_parameters(), reference.query_parameters()):
        torch.testing.assert_close(param.grad, param_ref.grad)


def test_a08_ema_matches_independent_formula_and_skips_bn_buffers():
    model = tiny_moco(m=0.9)
    with torch.no_grad():
        for param in model.key_parameters():
            param.add_(torch.randn_like(param))
        for buffer in model.backbone_q.buffers():
            buffer.add_(3)
    key_before = [p.detach().clone() for p in model.key_parameters()]
    query_before = [p.detach().clone() for p in model.query_parameters()]
    key_buffers_before = {name: b.clone() for name, b in model.backbone_k.named_buffers()}

    model._momentum_update_key_encoder()

    for param, expected in zip(model.key_parameters(), float64_ema(0.9, key_before, query_before)):
        torch.testing.assert_close(param.double(), expected, rtol=1e-6, atol=1e-7)
    for param, before in zip(model.query_parameters(), query_before):
        assert torch.equal(param, before)
    for name, buffer in model.backbone_k.named_buffers():
        assert torch.equal(buffer, key_buffers_before[name])
    key_parameter_names = {name for name, _ in model.backbone_k.named_parameters()}
    assert {"net.1.weight", "net.1.bias"} <= key_parameter_names  # BN affine parameters follow the EMA


def test_a08_a09_step_order_and_updates_follow_the_reference(monkeypatch):
    model = tiny_moco(m=0.9)
    optimizer = torch.optim.SGD(model.query_parameters(), lr=0.5, momentum=0.9)
    events = []
    model.backbone_q.register_forward_pre_hook(lambda *_: events.append("query_forward"))
    model.backbone_k.register_forward_pre_hook(lambda *_: events.append("key_forward"))
    optimizer.register_step_pre_hook(lambda *_: events.append("optimizer_step"))
    original_ema, original_enqueue = model._momentum_update_key_encoder, model._dequeue_and_enqueue
    original_logits = standalone_moco.moco_logits

    def ema():
        events.append("ema")
        original_ema()

    def enqueue(keys):
        events.append("enqueue")
        original_enqueue(keys)

    def logits(*args):
        events.append("logits")
        return original_logits(*args)

    monkeypatch.setattr(model, "_momentum_update_key_encoder", ema)
    monkeypatch.setattr(model, "_dequeue_and_enqueue", enqueue)
    monkeypatch.setattr(standalone_moco, "moco_logits", logits)

    initial_key = [p.detach().clone() for p in model.key_parameters()]
    for step in range(3):
        query_before = [p.detach().clone() for p in model.query_parameters()]
        key_before = [p.detach().clone() for p in model.key_parameters()]
        events.clear()
        out = model(*views(seed=step))
        out.loss.register_hook(lambda grad: events.append("backward"))
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()

        assert sum(p.grad is not None for p in model.key_parameters()) == 0
        grads = [p.grad for p in model.query_parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
        optimizer.step()

        assert events == ["query_forward", "ema", "key_forward", "logits", "enqueue", "backward", "optimizer_step"]
        # The EMA used the query of the previous optimizer step, not this step's update.
        for param, expected in zip(model.key_parameters(), float64_ema(0.9, key_before, query_before)):
            torch.testing.assert_close(param.double(), expected, rtol=1e-6, atol=1e-7)
        assert any(not torch.equal(p, before) for p, before in zip(model.query_parameters(), query_before))
        if step == 0:  # Q == K initially, so the first EMA keeps the key values
            for param, before in zip(model.key_parameters(), initial_key):
                torch.testing.assert_close(param, before)


def test_a10_key_batchnorm_buffers_follow_the_key_forward_only():
    model = tiny_moco(m=0.9)
    with torch.no_grad():
        for param in model.query_parameters():
            param.add_(0.1 * torch.randn_like(param))  # the EMA now changes the key weights
    reference = copy.deepcopy(model)
    im_q, im_k = views()

    model.train()
    model(im_q, im_k)

    reference.train()
    with torch.no_grad():
        reference.backbone_q(im_q)
        reference._momentum_update_key_encoder()
        reference.backbone_k(im_k)
    for module, module_ref in ((model.backbone_q, reference.backbone_q), (model.backbone_k, reference.backbone_k)):
        buffers_ref = dict(module_ref.named_buffers())
        for name, buffer in module.named_buffers():
            assert torch.equal(buffer, buffers_ref[name]), name
    bn_q, bn_k = model.backbone_q.net[1], model.backbone_k.net[1]
    assert int(bn_k.num_batches_tracked) == 1
    assert not torch.equal(bn_k.running_mean, torch.zeros(8))  # no_grad did not stop the running-stat update
    assert not torch.equal(bn_k.running_mean, bn_q.running_mean)  # not copied from the query branch


def test_a11_invalid_views_raise_before_any_state_change():
    model = tiny_moco()
    before = copy.deepcopy(model.state_dict())
    im_q, im_k = views()
    cases = [
        (ValueError, lambda: model(im_q, im_k[:2])),
        (ValueError, lambda: model(im_q[0], im_k[0])),
        (TypeError, lambda: model(im_q.double(), im_k.double())),
        (FloatingPointError, lambda: model(torch.full_like(im_q, float("nan")), im_k)),
        (ValueError, lambda: model(*views(batch_size=3))),  # K=8 is not a multiple of 3
    ]
    for error, call in cases:
        with pytest.raises(error):
            call()
        assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("corrupt, error", [
    (lambda m: m.queue_ptr.fill_(3), RuntimeError),
    (lambda m: m.queue_ptr.fill_(8), RuntimeError),
    (lambda m: m.queue_ptr.fill_(4), RuntimeError),  # disagrees with real_key_count=0
    (lambda m: m.real_key_count.fill_(9), RuntimeError),
    (lambda m: m.queue.mul_(2), RuntimeError),
    (lambda m: m.queue[0, 0].fill_(float("nan")), FloatingPointError),
])
def test_a11_corrupted_queue_state_raises(corrupt, error):
    model = tiny_moco()
    with torch.no_grad():
        corrupt(model)
    with pytest.raises(error):
        model(*views())
    with pytest.raises(error):
        model._dequeue_and_enqueue(unit_keys(4, 16, 0))


def test_a11_invalid_keys_construction_and_features_raise():
    model = tiny_moco()
    queue = model.queue.clone()
    nan_keys = unit_keys(4, 16, 0)
    nan_keys[0, 0] = float("nan")
    for error, keys in ((ValueError, unit_keys(4, 15, 0)), (TypeError, unit_keys(4, 16, 0).double()),
                        (ValueError, unit_keys(3, 16, 0)), (FloatingPointError, nan_keys),
                        (ValueError, 2 * unit_keys(4, 16, 0))):
        with pytest.raises(error):
            model._dequeue_and_enqueue(keys)
    assert torch.equal(model.queue, queue) and int(model.queue_ptr) == 0

    for kwargs in ({"dim": 0}, {"K": 0}, {"m": 1.5}, {"T": 0.0}):
        with pytest.raises(ValueError):
            StandaloneMoCo(TinyBackbone(), **kwargs)

    class WrongFeatureDim(TinyBackbone):
        feature_dim = 6

    with pytest.raises(ValueError, match="backbone output"):
        StandaloneMoCo(WrongFeatureDim(), dim=16, K=8)(*views())


def test_a11_non_finite_features_fail_before_enqueue():
    model = tiny_moco()
    with torch.no_grad():
        model.backbone_q.net[0].weight.fill_(float("nan"))
    queue = model.queue.clone()
    with pytest.raises(FloatingPointError, match="non-finite"):
        model(*views())
    assert torch.equal(model.queue, queue) and int(model.queue_ptr) == 0 and int(model.real_key_count) == 0


def test_same_seed_gives_identical_loss_ema_and_queue_on_cpu():
    def train(seed):
        model = tiny_moco(seed=seed)
        optimizer = torch.optim.SGD(model.query_parameters(), lr=0.1, momentum=0.9)
        losses = []
        for step in range(4):
            out = model(*views(seed=10 + step))
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            losses.append(out.loss.item())
        return losses, model.state_dict()

    losses_a, state_a = train(0)
    losses_b, state_b = train(0)
    assert losses_a == losses_b
    assert state_a.keys() == state_b.keys() and all(torch.equal(state_a[k], state_b[k]) for k in state_a)
