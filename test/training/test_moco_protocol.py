"""Independent streaming protocol, view and old-queue negative contracts."""

from dataclasses import FrozenInstanceError, fields, replace
from itertools import product
from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F

from integration.sequential_moco import make_two_views
from self_supervised.moco.negative_selection import select_negatives
from self_supervised.moco.vit_lora_moco import MetadataQueue, QueueEntry, ViTLoRAMoCo
from test_activitynet_vit_clip_feature import make_sample
from training.moco_audit import audit_views
from training.moco_protocol import StreamingMoCoProtocol, STAGE6A_PROTOCOL, STAGE6B_PROTOCOL


def unit_key(index):
    return F.one_hot(torch.tensor(index), num_classes=128).float()


@pytest.mark.parametrize('stream_mode,key_transform,negative_policy', tuple(product(
    ('round_robin', 'strict_single'),
    ('horizontal_flip', 'gbr_horizontal_flip'),
    ('different_sequence', 'all_past'),
)))
def test_axes_are_independent_and_warmup_matches_active_sources(stream_mode, key_transform, negative_policy):
    protocol = StreamingMoCoProtocol(stream_mode, key_transform, negative_policy)
    assert (protocol.stream_mode, protocol.key_transform, protocol.negative_policy) == (
        stream_mode, key_transform, negative_policy
    )
    assert protocol.source_count == protocol.warmup_count == (4 if stream_mode == 'round_robin' else 1)


def test_presets_defaults_and_immutability():
    assert STAGE6A_PROTOCOL == StreamingMoCoProtocol('round_robin', 'horizontal_flip', 'different_sequence')
    assert STAGE6B_PROTOCOL == StreamingMoCoProtocol('strict_single', 'gbr_horizontal_flip', 'all_past')
    assert StreamingMoCoProtocol() == STAGE6A_PROTOCOL
    assert replace(STAGE6B_PROTOCOL, key_transform='horizontal_flip') == StreamingMoCoProtocol(
        'strict_single', 'horizontal_flip', 'all_past'
    )
    with pytest.raises(FrozenInstanceError):
        STAGE6B_PROTOCOL.stream_mode = 'round_robin'


@pytest.mark.parametrize('axis', ['stream_mode', 'key_transform', 'negative_policy'])
def test_protocol_rejects_unknown_axes(axis):
    with pytest.raises(ValueError, match=axis):
        StreamingMoCoProtocol(**{axis: 'unknown'})


@pytest.mark.parametrize('transform', ['horizontal_flip', 'gbr_horizontal_flip'])
def test_views_transform_all_valid_rgb_frames_and_preserve_padding_and_every_metadata_field(transform):
    sample = make_sample()
    sample.frames[:2] = torch.arange(24).reshape(2, 3, 2, 2)
    before = sample.frames.clone()
    query, key = make_two_views(sample, transform)
    assert query is sample
    assert torch.equal(query.frames, before) and torch.equal(sample.frames, before)
    assert key.frames.data_ptr() != sample.frames.data_ptr()
    channels = (0, 1, 2) if transform == 'horizontal_flip' else (1, 2, 0)
    for output_channel, input_channel in enumerate(channels):
        assert torch.equal(key.frames[:2, output_channel], before[:2, input_channel].flip(-1))
    if transform == 'gbr_horizontal_flip':
        assert key.frames[0, :, 0, :].tolist() == [[5, 4], [9, 8], [1, 0]]
    assert torch.equal(key.frames[2:], before[2:])
    assert (key.frames.shape, key.frames.dtype, key.frames.device) == (
        before.shape, before.dtype, before.device
    )
    for field in fields(sample):
        if field.name != 'frames':
            assert getattr(query, field.name) is getattr(sample, field.name)
            assert getattr(key, field.name) is getattr(sample, field.name)
    audit_views(sample, query, key, transform)


def test_view_default_is_stage5_horizontal_flip_and_unknown_transform_fails():
    sample = make_sample()
    sample.frames[:2] = torch.arange(24).reshape(2, 3, 2, 2)
    query, key = make_two_views(sample)
    assert torch.equal(key.frames[:2], sample.frames[:2].flip(-1))
    audit_views(sample, query, key)
    with pytest.raises(ValueError, match='key_transform'):
        make_two_views(sample, 'unknown')
    with pytest.raises(ValueError, match='key_transform'):
        audit_views(sample, query, key, 'unknown')


@pytest.mark.parametrize('corruption', ['query', 'key', 'padding', 'source_id', 'reference'])
def test_view_audit_rejects_transform_padding_and_metadata_changes(corruption):
    sample = make_sample()
    sample.frames[:2] = torch.arange(24).reshape(2, 3, 2, 2)
    query, key = make_two_views(sample, 'gbr_horizontal_flip')
    if corruption == 'query':
        query = replace(query, frames=query.frames.clone() + 1)
    elif corruption in ('key', 'padding'):
        key.frames[0 if corruption == 'key' else 2, 0, 0, 0] += 1
    elif corruption == 'source_id':
        key = replace(key, source_id='wrong-source')
    else:
        key = replace(key, evaluation_reference=object())
    with pytest.raises(RuntimeError):
        audit_views(sample, query, key, 'gbr_horizontal_flip')


def test_negative_selection_preserves_old_entries_order_and_policy_independence():
    queue = MetadataQueue()
    for i, sequence in enumerate(('a', 'b', 'a', 'c')):
        queue.enqueue(unit_key(i), sequence, i)
    old_entries = queue.entries
    different = select_negatives(old_entries, 'a')
    all_past = select_negatives(old_entries, 'a', 'all_past')
    assert different == (old_entries[1], old_entries[3])
    assert all(entry is old for entry, old in zip(all_past, old_entries))
    assert queue.negatives('a') == different
    queue.enqueue(unit_key(4), 'a', 4)
    assert len(all_past) == 4 and len(queue) == 5
    assert all(entry.sequence_index < 4 for entry in all_past)


@pytest.mark.parametrize('policy', ['different_sequence', 'all_past'])
def test_negative_selection_rejects_empty_queue(policy):
    with pytest.raises(RuntimeError, match='No valid'):
        select_negatives((), 'a', policy)


def test_strict_single_different_sequence_fails_clearly_but_all_past_includes_same_video():
    protocol = StreamingMoCoProtocol('strict_single', 'horizontal_flip', 'different_sequence')
    queue = MetadataQueue()
    queue.enqueue(unit_key(0), 'a', 0)
    with pytest.raises(RuntimeError, match='No valid different-sequence negatives'):
        select_negatives(queue.entries, 'a', protocol.negative_policy)
    assert [entry.sequence_index for entry in select_negatives(queue.entries, 'a', 'all_past')] == [0]
    queue.enqueue(unit_key(1), 'a', 1)
    assert [entry.sequence_index for entry in select_negatives(queue.entries, 'a', 'all_past')] == [0, 1]
    with pytest.raises(ValueError, match='negative_policy'):
        select_negatives(queue.entries, 'a', 'unknown')


def test_default_capacity_retains_only_4096_most_recent_keys_with_fifo_metadata():
    queue = MetadataQueue()
    assert queue.capacity == 4096
    keys = [unit_key(i) for i in range(128)]
    for index in range(4100):
        queue.enqueue(keys[index % 128], 'same-video', index)
    entries = select_negatives(queue.entries, 'same-video', 'all_past')
    assert len(queue) == len(entries) == 4096
    assert [entry.sequence_index for entry in entries] == list(range(4, 4100))
    assert all(entry.sequence_id == 'same-video' for entry in entries)
    for entry in entries:
        assert torch.equal(entry.key, keys[entry.sequence_index % 128])
        assert not entry.key.requires_grad and entry.key.grad_fn is None


def test_infonce_uses_only_explicit_negatives_and_detaches_all_keys():
    moco = ViTLoRAMoCo(torch.nn.Identity())
    moco.queue.negatives = Mock(side_effect=AssertionError('explicit selection must be used'))
    raw_query = (unit_key(0) + .25 * unit_key(1)).requires_grad_()
    query = F.normalize(raw_query, dim=-1)
    positive = unit_key(0).requires_grad_()
    negative = unit_key(1).requires_grad_()
    entries = (QueueEntry(negative, 'same-video', 0),)
    loss, logits, selected = moco.contrastive_loss(query, positive, 'same-video', negatives=entries)
    expected = torch.stack((query[0], query[1])).unsqueeze(0) / .07
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(loss, -F.log_softmax(expected, dim=1)[0, 0])
    assert selected[0] is entries[0] and len(moco.queue) == 0
    loss.backward()
    assert torch.isfinite(raw_query.grad).all() and raw_query.grad.count_nonzero() > 0
    assert positive.grad is None and negative.grad is None
    with pytest.raises(RuntimeError, match='No valid negatives'):
        moco.contrastive_loss(query, positive, 'same-video', negatives=())
