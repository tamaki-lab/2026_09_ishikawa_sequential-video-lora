import pytest
import torch
from torch.nn import functional as F

from self_supervised.moco import MetadataQueue, ViTLoRAMoCo
from self_supervised.moco.vit_lora_moco import lora_parameters
from test_vit_lora_frame_encoder import encoder


@pytest.fixture
def moco(encoder):
    return ViTLoRAMoCo(encoder)


def unit_key(index):
    return F.one_hot(torch.tensor(index), num_classes=128).float()


def test_projector_counts_batched_shape_normalization_and_independent_initial_state(moco):
    assert len(moco.queue) == 0 and moco.queue.capacity == 4096
    assert moco.momentum == 0.999 and moco.temperature == 0.07
    for query, key in ((moco.query_encoder, moco.key_encoder), (moco.query_projector, moco.key_projector)):
        query_parameters = dict(query.named_parameters())
        key_parameters = dict(key.named_parameters())
        assert query_parameters.keys() == key_parameters.keys()
        for name, parameter in query_parameters.items():
            assert torch.equal(parameter, key_parameters[name])
            assert parameter.data_ptr() != key_parameters[name].data_ptr()
        assert all(not p.requires_grad for p in key.parameters())
    assert sum(p.numel() for p in moco.query_projector.parameters()) == 689_024
    assert len(list(moco.query_projector.parameters())) == 4
    assert all(p.requires_grad for p in moco.query_projector.parameters())
    assert moco.query_projector(torch.randn(3, 768)).shape == (3, 128)
    clip = torch.randn(768, requires_grad=True)
    query, key = moco.project_query(clip), moco.project_key(clip)
    torch.testing.assert_close(query, key)
    for value in (query, key):
        assert value.shape == (128,) and torch.isfinite(value).all()
        torch.testing.assert_close(value.norm(), torch.tensor(1.))
    assert query.requires_grad and not key.requires_grad
    expected = list(lora_parameters(moco.query_encoder).values()) + list(moco.query_projector.parameters())
    optimized = moco.query_parameters()
    assert len(optimized) == 52 and sum(p.numel() for p in optimized) == 983_936
    assert {id(p) for p in optimized} == {id(p) for p in expected}


def test_queue_empty_fifo_overflow_detach_and_metadata_alignment():
    queue = MetadataQueue(capacity=3)
    assert queue.entries == ()
    key = unit_key(0).requires_grad_()
    queue.enqueue(key, 'a', 0)
    with torch.no_grad():
        key.copy_(unit_key(10))
    assert torch.equal(queue.entries[0].key, unit_key(0))
    for i in range(1, 6):
        queue.enqueue(unit_key(i), f'video-{i}', i * 100)
    assert len(queue) == 3
    assert [entry.sequence_id for entry in queue.entries] == ['video-3', 'video-4', 'video-5']
    for i, entry in zip(range(3, 6), queue.entries):
        assert entry.sequence_index == i * 100
        assert torch.equal(entry.key, unit_key(i))
        assert not entry.key.requires_grad and entry.key.grad_fn is None


def test_queue_masks_same_sequence_at_any_distance_and_fails_without_negatives():
    queue = MetadataQueue()
    with pytest.raises(RuntimeError, match='No valid'):
        queue.negatives('a')
    for index in (0, 1, 10000):
        queue.enqueue(unit_key(0), 'a', index)
    with pytest.raises(RuntimeError, match='No valid'):
        queue.negatives('a')
    queue.enqueue(unit_key(1), 'b', 0)
    assert [entry.sequence_id for entry in queue.negatives('a')] == ['b']
    assert len(queue.negatives('b')) == 3
    assert len(queue) == 4


@pytest.mark.parametrize('bad_key', [torch.ones(127), torch.zeros(128), torch.ones(128),
                                    torch.full((128,), float('nan')), torch.full((128,), float('inf'))])
def test_queue_rejects_nonfinite_wrong_shape_or_unnormalized_keys(bad_key):
    queue = MetadataQueue()
    with pytest.raises(RuntimeError):
        queue.enqueue(bad_key, 'video', 0)
    assert len(queue) == 0


def test_infonce_positive_zero_temperature_detached_keys_and_old_queue(moco):
    raw_query = (unit_key(0) + unit_key(1) * 0.25).requires_grad_()
    query = F.normalize(raw_query, dim=-1)
    positive = unit_key(0).requires_grad_()
    negative = unit_key(1).requires_grad_()
    moco.queue.enqueue(unit_key(2), 'current', 999)
    moco.queue.enqueue(negative, 'past-a', 0)
    moco.queue.enqueue(unit_key(3), 'past-b', 8)
    loss, logits, entries = moco.contrastive_loss(query, positive, 'current')
    expected = torch.stack((query[0], query[1], query[3])).unsqueeze(0) / 0.07
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(loss, -F.log_softmax(expected, dim=1)[0, 0])
    assert logits.shape == (1, 3) and loss.ndim == 0 and torch.isfinite(loss)
    assert [entry.sequence_id for entry in entries] == ['past-a', 'past-b']
    assert len(moco.queue) == 3
    loss.backward()
    assert torch.isfinite(raw_query.grad).all() and raw_query.grad.count_nonzero() > 0
    assert positive.grad is None and negative.grad is None
    with pytest.raises(RuntimeError, match='No valid'):
        moco.queue = MetadataQueue()
        moco.contrastive_loss(query, positive, 'current')


def test_ema_matches_formula_changes_only_adapters_and_projector(moco):
    before = {name: p.detach().clone() for name, p in moco.key_encoder.named_parameters()}
    before_projector = {name: p.detach().clone() for name, p in moco.key_projector.named_parameters()}
    with torch.no_grad():
        for p in moco.query_parameters():
            p.add_(0.5)
    moco.update_key()
    lora = lora_parameters(moco.key_encoder)
    query = dict(moco.query_encoder.named_parameters())
    for name, key in moco.key_encoder.named_parameters():
        if name in lora:
            torch.testing.assert_close(key, 0.999 * before[name] + 0.001 * query[name], rtol=1e-6, atol=1e-8)
            assert not torch.equal(key, before[name])
        else:
            assert torch.equal(key, before[name])
        assert key.grad is None and not key.requires_grad
    query_projector = dict(moco.query_projector.named_parameters())
    for name, key in moco.key_projector.named_parameters():
        torch.testing.assert_close(key, 0.999 * before_projector[name] + 0.001 * query_projector[name],
                                   rtol=1e-6, atol=1e-8)
        assert not torch.equal(key, before_projector[name]) and key.grad is None
