"""Shared protocol scheduling and strict-online update contracts on CPU."""

from dataclasses import replace
from itertools import product

import pytest
import torch

from self_supervised.moco import ViTLoRAMoCo
from self_supervised.moco.vit_lora_moco import MetadataQueue
from test_activitynet_vit_clip_feature import RecordingProcessor
from test_moco_multistep_canary import reader, records, small_moco, sources
from test_vit_lora_frame_encoder import encoder
import training.moco_canary as canary
from training.moco_protocol import StreamingMoCoProtocol, STAGE6B_PROTOCOL


def test_strict_order_has_no_prefetch_and_closes_on_early_stop(sources, reader):
    source = replace(sources[0], start_frame=11)
    with canary.ordered_samples((source,), stream_mode='strict_single') as samples:
        assert reader['reads'] == []
        for index in range(4):
            sample = next(samples)
            assert (sample.sequence_id, sample.sequence_index, sample.is_first) == ('a', index, index == 0)
            assert reader['reads'] == [('a', i) for i in range(index + 1)]
            assert sample.frame_indices.tolist() == list(range(11 + index * 16, 27 + index * 16))
            assert sample.timestamps.tolist() == list(reversed(range(16)))
    assert reader['opened'] == reader['closed'] == ['a']


@pytest.mark.parametrize('ending', ['eof', 'decode', 'consumer', 'interrupt'])
def test_strict_reader_closes_on_every_exit(sources, reader, ending):
    reader['eof'] = ('a', 2) if ending == 'eof' else None
    reader['failure'] = ('a', 2) if ending == 'decode' else None

    def consume():
        with canary.ordered_samples(sources[:1], stream_mode='strict_single') as samples:
            for sample in samples:
                if sample.sequence_index == 2 and ending == 'consumer':
                    raise ValueError('consumer failure')
                if sample.sequence_index == 2 and ending == 'interrupt':
                    raise KeyboardInterrupt()
                if sample.is_last:
                    assert sample.valid_mask.sum() == 3

    if ending == 'eof':
        consume()
    else:
        error = {'decode': RuntimeError, 'consumer': ValueError, 'interrupt': KeyboardInterrupt}[ending]
        with pytest.raises(error):
            consume()
    assert reader['reads'] == [('a', 0), ('a', 1), ('a', 2)]
    assert reader['opened'] == reader['closed'] == ['a']


def test_stage6b_key_only_warmup_past_negatives_and_update_order(
    small_moco, sources, reader, monkeypatch, capsys,
):
    moco = small_moco
    events, positive_keys, optimizers = [], [], []
    initial = {name: p.detach().clone() for name, p in moco.named_parameters()}
    optimizer_factory = torch.optim.AdamW
    ema, enqueue, loss_fn = moco.update_key, moco.queue.enqueue, moco.contrastive_loss
    moco.query_encoder.register_forward_pre_hook(lambda *_: events.append('query'))
    moco.key_encoder.register_forward_pre_hook(lambda *_: events.append('key'))

    def optimizer(parameters, **kwargs):
        assert kwargs == {'lr': 1e-3, 'weight_decay': 0.0}
        assert events == ['key', 'enqueue']
        assert all(p.grad is None and torch.equal(p, initial[name]) for name, p in moco.named_parameters())
        result = optimizer_factory(parameters, **kwargs)
        optimized = [p for group in result.param_groups for p in group['params']]
        assert {id(p) for p in optimized} == {id(p) for p in moco.query_parameters()}
        optimizers.append(result)
        step = result.step

        def update():
            events.append('optimizer')
            return step()
        result.step = update
        return result

    def update_key():
        events.append('ema')
        old = {name: p.detach().clone() for name, p in moco.named_parameters() if name.startswith('key_')}
        ema()
        for name, p in moco.named_parameters():
            if name in old and ('.lora_' in name or name.startswith('key_projector.')):
                query = dict(moco.named_parameters())[name.replace('key_', 'query_', 1)]
                torch.testing.assert_close(p, old[name] * .999 + query.detach() * .001, rtol=1e-6, atol=1e-8)

    def compute_loss(query, key, sequence_id, *, negatives):
        events.append('loss')
        index = len(positive_keys) + 1
        old_entries = moco.queue.entries
        assert sequence_id == 'a'
        assert [(entry.sequence_id, entry.sequence_index) for entry in negatives] == [('a', i) for i in range(index)]
        assert len(old_entries) == index
        assert all(entry is old for entry, old in zip(negatives, old_entries))
        assert all(entry.sequence_index < index for entry in negatives)
        positive_keys.append(key.clone())
        loss, logits, selected = loss_fn(query, key, sequence_id, negatives=negatives)
        assert logits.shape == (1, index + 1)
        assert torch.isfinite(loss) and torch.isfinite(logits).all()
        loss.register_hook(lambda grad: events.append('backward'))
        return loss, logits, selected

    def append(key, sequence_id, sequence_index):
        events.append('enqueue')
        assert not key.requires_grad and key.grad_fn is None
        if sequence_index:
            assert torch.equal(key, positive_keys[-1])
        enqueue(key, sequence_id, sequence_index)

    monkeypatch.setattr(torch.optim, 'AdamW', optimizer)
    monkeypatch.setattr(moco, 'update_key', update_key)
    monkeypatch.setattr(moco, 'contrastive_loss', compute_loss)
    monkeypatch.setattr(moco.queue, 'enqueue', append)
    result = canary.run_streaming_moco(
        moco, RecordingProcessor(), sources[:1], torch.device('cpu'), 2, protocol=STAGE6B_PROTOCOL,
    )
    assert events == ['key', 'enqueue'] + ['query', 'key', 'loss', 'backward', 'optimizer', 'ema', 'enqueue'] * 2
    assert reader['reads'] == [('a', 0), ('a', 1), ('a', 2)]
    assert [(entry.sequence_id, entry.sequence_index) for entry in moco.queue.entries] == reader['reads']
    assert reader['opened'] == reader['closed'] == ['a']
    assert len(optimizers) == 1
    assert all(int(state['step']) == 2 for state in optimizers[0].state.values())
    assert result['training_steps'] == 2 and result['queue_count'] == 3
    assert result['all_parameters_finite'] and result['all_queue_keys_finite']
    for side in ('query', 'key'):
        assert result['changed_tensors'][f'{side} base'] == 0
        assert torch.equal(getattr(moco, f'{side}_encoder').base.weight, initial[f'{side}_encoder.base.weight'])
        for kind in ('LoRA', 'Projector'):
            assert result['changed_tensors'][f'{side} {kind}'] > 0
    rows = records(capsys)
    assert len([row for row in rows if row['event'] == 'warmup']) == 1
    steps = [row for row in rows if row['event'] == 'step']
    assert len(steps) == 2
    for index, row in enumerate(steps):
        assert row['step'] == index and row['sequence_index'] == index + 1
        assert row['valid_negative_count'] == index + 1 and row['queue_count'] == index + 2
        assert row['queue_unique_sequence_id_count'] == 1
        assert row['gradients']['query base gradient count'] == row['gradients']['key gradient count'] == 0
        for kind in ('LoRA', 'Projector'):
            assert row['gradients'][f'query {kind}']['finite'] > 0
            assert row['gradients'][f'query {kind}']['nonzero'] > 0
            assert row['gradients'][f'query {kind}']['norm'] > 0


@pytest.mark.parametrize('stream_mode,key_transform,negative_policy', tuple(product(
    ('round_robin', 'strict_single'),
    ('horizontal_flip', 'gbr_horizontal_flip'),
    ('different_sequence', 'all_past'),
)))
def test_engine_axes_are_independent(
    small_moco, sources, reader, capsys, stream_mode, key_transform, negative_policy,
):
    protocol = StreamingMoCoProtocol(
        stream_mode=stream_mode, key_transform=key_transform, negative_policy=negative_policy,
    )

    class CaptureProcessor(RecordingProcessor):
        def __init__(self):
            self.calls = []

        def __call__(self, *, images, return_tensors):
            self.calls.append(images.clone())
            return super().__call__(images=images, return_tensors=return_tensors)

    processor = CaptureProcessor()
    impossible = stream_mode == 'strict_single' and negative_policy == 'different_sequence'
    if impossible:
        # Independent axes are accepted; an impossible negative set must not
        # silently switch stream mode, admit the current key or change policy.
        with pytest.raises(RuntimeError, match='[Nn]o .*negative|[Ee]mpty.*negative|negative.*empty'):
            canary.run_streaming_moco(
                small_moco, processor, sources[:protocol.source_count], torch.device('cpu'), 2, protocol=protocol,
            )
        assert reader['reads'] == [('a', 0), ('a', 1)]
        assert [(entry.sequence_id, entry.sequence_index) for entry in small_moco.queue.entries] == [('a', 0)]
    else:
        result = canary.run_streaming_moco(
            small_moco, processor, sources[:protocol.source_count], torch.device('cpu'), 2, protocol=protocol,
        )
        assert result['training_steps'] == 2
        assert result['queue_count'] == protocol.warmup_count + 2
        expected = [('a', 0), ('a', 1), ('a', 2)] if stream_mode == 'strict_single' else [
            ('a', 0), ('b', 0), ('c', 0), ('d', 0), ('a', 1), ('b', 1),
        ]
        assert reader['reads'] == expected
    assert sorted(reader['closed']) == sorted(reader['opened']) == list('abcd'[:protocol.source_count])
    # Inspect the actual frames delivered to the processor in each combination.
    calls = iter(processor.calls)
    for position, (video, index) in enumerate(reader['reads']):
        raw = ((torch.arange(16 * 12).reshape(16, 3, 2, 2) + ord(video) + index * 3) % 251).to(torch.uint8)
        if position >= protocol.warmup_count:
            assert torch.equal(next(calls), raw)
        transformed = raw if key_transform == 'horizontal_flip' else raw.roll(-1, dims=1)
        assert torch.equal(next(calls), transformed.flip(-1))
    assert next(calls, None) is None
    rows = records(capsys)
    steps = [row for row in rows if row['event'] == 'step']
    assert len(steps) == (0 if impossible else 2)
    if not impossible:
        expected_counts = [3, 4] if negative_policy == 'different_sequence' else [protocol.warmup_count, protocol.warmup_count + 1]
        assert [row['valid_negative_count'] for row in steps] == expected_counts


def test_shared_engine_fifo_audit_accepts_capacity_overflow(small_moco, sources, reader, monkeypatch, capsys):
    # Reduce only this test's capacity to exercise eviction in a short CPU run.
    small_moco.queue = MetadataQueue(capacity=3)
    seen = []
    original = small_moco.contrastive_loss

    def loss(query, key, sequence_id, *, negatives):
        current_index = len(seen) + 1
        expected = list(range(max(0, current_index - 3), current_index))
        assert [entry.sequence_index for entry in negatives] == expected
        assert all(entry.sequence_id == 'a' for entry in negatives)
        seen.append(expected)
        return original(query, key, sequence_id, negatives=negatives)

    monkeypatch.setattr(small_moco, 'contrastive_loss', loss)
    result = canary.run_streaming_moco(
        small_moco, RecordingProcessor(), sources[:1], torch.device('cpu'), 5, protocol=STAGE6B_PROTOCOL,
    )
    assert result['training_steps'] == 5 and result['queue_count'] == 3
    assert seen == [[0], [0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]]
    assert [entry.sequence_index for entry in small_moco.queue.entries] == [3, 4, 5]
    assert reader['reads'] == [('a', i) for i in range(6)]
    assert reader['closed'] == ['a']
    steps = [row for row in records(capsys) if row['event'] == 'step']
    assert [row['queue_count'] for row in steps] == [2, 3, 3, 3, 3]


@pytest.mark.parametrize('terminal,target,success', [(0, 2, False), (2, 3, False), (2, 2, True)])
def test_strict_eof_does_not_claim_unreached_target(small_moco, sources, reader, capsys, terminal, target, success):
    reader['eof'] = ('a', terminal)
    if success:
        result = canary.run_streaming_moco(
            small_moco, RecordingProcessor(), sources[:1], torch.device('cpu'), target, protocol=STAGE6B_PROTOCOL,
        )
        assert result['training_steps'] == target
    else:
        with pytest.raises(RuntimeError, match='EOF after'):
            canary.run_streaming_moco(
                small_moco, RecordingProcessor(), sources[:1], torch.device('cpu'), target, protocol=STAGE6B_PROTOCOL,
            )
    assert reader['reads'] == [('a', i) for i in range(terminal + 1)]
    assert reader['closed'] == ['a']
    rows = records(capsys)
    assert any(row['event'] == 'complete' for row in rows) == success
    if not success:
        assert rows[-1]['event'] == 'stopped' and rows[-1]['training_steps'] == terminal


@pytest.mark.parametrize('failure', ['query_forward', 'key_forward', 'loss', 'optimizer', 'ema', 'enqueue'])
def test_strict_training_exception_closes_reader(small_moco, sources, reader, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')

    if failure == 'query_forward':
        monkeypatch.setattr(small_moco.query_encoder, 'forward', fail)
    elif failure == 'key_forward':
        original = small_moco.key_encoder.forward
        calls = 0

        def forward(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                fail()
            return original(*args, **kwargs)
        monkeypatch.setattr(small_moco.key_encoder, 'forward', forward)
    elif failure == 'loss':
        monkeypatch.setattr(small_moco, 'contrastive_loss', fail)
    elif failure == 'optimizer':
        monkeypatch.setattr(torch.optim.AdamW, 'step', fail)
    elif failure == 'ema':
        monkeypatch.setattr(small_moco, 'update_key', fail)
    else:
        original = small_moco.queue.enqueue

        def enqueue(key, sequence_id, sequence_index):
            if sequence_index:
                fail()
            original(key, sequence_id, sequence_index)
        monkeypatch.setattr(small_moco.queue, 'enqueue', enqueue)
    with pytest.raises(RuntimeError, match='injected failure'):
        canary.run_streaming_moco(
            small_moco, RecordingProcessor(), sources[:1], torch.device('cpu'), 2, protocol=STAGE6B_PROTOCOL,
        )
    assert reader['reads'] == [('a', 0), ('a', 1)]
    assert reader['opened'] == reader['closed'] == ['a']


def test_stage6b_real_vit_peft_integration(encoder, sources, reader, capsys):
    # Real 12-layer, width-768 ViT/PEFT; the existing fixture reduces MLP width.
    moco = ViTLoRAMoCo(encoder)
    result = canary.run_streaming_moco(
        moco, RecordingProcessor(), sources[:1], torch.device('cpu'), 1, protocol=STAGE6B_PROTOCOL,
    )
    assert result['training_steps'] == 1 and result['queue_count'] == 2
    assert moco.queue.capacity == 4096
    assert reader['reads'] == [('a', 0), ('a', 1)]
    assert reader['opened'] == reader['closed'] == ['a']
    rows = records(capsys)
    optimizer = next(row for row in rows if row['event'] == 'optimizer')
    assert optimizer['parameters'] == 983_936 and optimizer['tensors'] == 52
    step = next(row for row in rows if row['event'] == 'step')
    assert step['valid_negative_count'] == 1
    assert step['gradients']['query LoRA']['finite'] == 48
    assert step['gradients']['query LoRA']['nonzero'] > 0
    assert step['gradients']['query base gradient count'] == step['gradients']['key gradient count'] == 0
