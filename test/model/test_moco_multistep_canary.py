"""Stage 6A scheduling, lifecycle and update contracts without real video data."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import sequential_loader as sl
import torch
from torch import nn

from self_supervised.moco import ViTLoRAMoCo
from self_supervised.moco.vit_lora_moco import lora_parameters
from test_activitynet_vit_clip_feature import RecordingProcessor
from test_vit_lora_frame_encoder import encoder
import training.moco_canary as canary
import smoke_activitynet_vit_lora_moco_multistep as smoke


@pytest.fixture
def sources():
    return tuple(sl.SequenceSource(
        sequence_id=video, source_id=video, source=Path(f'{video}.mp4'),
        start_frame=0, stop_frame=None, evaluation_reference=object(),
    ) for video in ('a', 'b', 'c', 'd'))


@pytest.fixture
def reader(monkeypatch):
    state = {'opened': [], 'closed': [], 'reads': [], 'eof': None, 'failure': None}

    class Reader:
        @contextmanager
        def open(self, source):
            self.source = source
            state['opened'].append(source.sequence_id)
            try:
                yield self
            finally:
                state['closed'].append(source.sequence_id)

        def read(self, chunk):
            location = (self.source.sequence_id, chunk.chunk_index)
            state['reads'].append(location)
            if state['failure'] == location:
                raise RuntimeError('decode failure')
            terminal = state['eof'] == location
            count = 3 if terminal else 16
            frames = ((torch.arange(count * 12).reshape(count, 3, 2, 2)
                       + ord(self.source.sequence_id) + chunk.chunk_index * 3) % 251).to(torch.uint8)
            return sl.DecodedChunk(
                frames=frames, frame_indices=torch.arange(chunk.start_frame, chunk.start_frame + count),
                # Raw PTS can go backwards; chronology is defined by decode index.
                timestamps=torch.arange(count, dtype=torch.float64).flip(0), reached_eof=terminal,
            )

    monkeypatch.setattr(sl, 'SequentialVideoReader', Reader)
    return state


class SmallEncoder(nn.Module):
    """Cheap gradient-bearing stand-in; real ViT/PEFT is covered separately below."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(3, 768, bias=False).requires_grad_(False)
        self.adapter = nn.Module()
        self.adapter.lora_A = nn.Linear(3, 8, bias=False)
        self.adapter.lora_B = nn.Linear(8, 768, bias=False)
        nn.init.zeros_(self.adapter.lora_B.weight)

    def forward(self, pixels):
        colors = pixels[:, :, 0, 0] / 255
        return self.base(colors) + self.adapter.lora_B(self.adapter.lora_A(colors))


@pytest.fixture
def small_moco(monkeypatch):
    moco = ViTLoRAMoCo(SmallEncoder())

    def initial(model):
        # Only substitute the ViT-specific architecture audit. The orchestration,
        # real MoCo loss/queue/EMA, optimizer and all run-time audits stay intact.
        assert len(model.queue) == 0
        for query, key in ((model.query_encoder, model.key_encoder), (model.query_projector, model.key_projector)):
            for a, b in zip(query.parameters(), key.parameters()):
                assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
            assert all(not p.requires_grad for p in key.parameters())
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert {id(p) for p in trainable} == {id(p) for p in model.query_parameters()}

    monkeypatch.setattr(canary, 'audit_initial_state', initial)
    return moco


def records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]


def test_round_robin_no_prefetch_and_early_consumer_close(sources, reader):
    with canary.round_robin_samples(sources) as samples:
        assert reader['reads'] == []
        for expected in [('a', 0), ('b', 0), ('c', 0), ('d', 0), ('a', 1), ('b', 1)]:
            sample = next(samples)
            assert (sample.sequence_id, sample.sequence_index) == expected
            assert reader['reads'][-1] == expected
            assert len(reader['reads']) == 4 * expected[1] + 'abcd'.index(expected[0]) + 1
            assert sample.frame_indices.tolist() == list(range(expected[1] * 16, (expected[1] + 1) * 16))
            assert sample.timestamps.tolist() == list(reversed(range(16)))
    assert sorted(reader['closed']) == list('abcd')


@pytest.mark.parametrize('ending', ['eof', 'decode', 'consumer', 'interrupt'])
def test_all_open_readers_close_on_every_exit(sources, reader, ending):
    reader['eof'] = ('b', 1) if ending == 'eof' else None
    reader['failure'] = ('b', 1) if ending == 'decode' else None

    def consume():
        with canary.round_robin_samples(sources) as samples:
            for index, sample in enumerate(samples):
                if index == 4 and ending == 'consumer':
                    raise ValueError('consumer failure')
                if index == 4 and ending == 'interrupt':
                    raise KeyboardInterrupt()
                if sample.is_last:
                    assert sample.valid_mask.sum() == 3
    if ending == 'eof':
        consume()
        assert reader['reads'][-1] == ('b', 1)
    else:
        error = {'decode': RuntimeError, 'consumer': ValueError, 'interrupt': KeyboardInterrupt}[ending]
        with pytest.raises(error):
            consume()
    assert sorted(reader['closed']) == list('abcd')


@pytest.mark.parametrize('failure', ['identity', 'index', 'frames'])
def test_scheduler_rejects_corrupt_order_before_yield(sources, reader, monkeypatch, failure):
    stream = sl.sequential_sample_stream

    @contextmanager
    def corrupt(loader):
        with stream(loader) as items:
            def generate():
                sample = next(items)
                if failure == 'identity':
                    sample = replace(sample, sequence_id='wrong')
                elif failure == 'index':
                    sample = replace(sample, sequence_index=1, is_first=False)
                else:
                    sample = replace(sample, frame_indices=sample.frame_indices + 1)
                yield sample
            yield generate()

    monkeypatch.setattr(sl, 'sequential_sample_stream', corrupt)
    with pytest.raises(RuntimeError, match='order|indices'):
        with canary.round_robin_samples(sources) as samples:
            next(samples)
    assert reader['closed'] == ['a']


@pytest.mark.parametrize('max_steps', [10, 100])
def test_multistep_real_moco_semantics_with_small_encoder(small_moco, sources, reader, monkeypatch, capsys, max_steps):
    moco = small_moco
    events, positive_keys, optimizers = [], [], []
    initial = {name: p.detach().clone() for name, p in moco.named_parameters()}
    optimizer_factory = torch.optim.AdamW
    ema, enqueue, loss_fn = moco.update_key, moco.queue.enqueue, moco.contrastive_loss
    moco.query_encoder.register_forward_pre_hook(lambda *_: events.append('query'))
    moco.key_encoder.register_forward_pre_hook(lambda *_: events.append('key'))

    def optimizer(parameters, **kwargs):
        assert kwargs == {'lr': 1e-3, 'weight_decay': 0.0}
        assert events == ['key', 'enqueue'] * 4
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

    def compute_loss(query, key, sequence_id):
        events.append('loss')
        step = len(positive_keys)
        assert len(moco.queue) == 4 + step
        positive_keys.append(key.clone())
        loss, logits, negatives = loss_fn(query, key, sequence_id)
        assert len(negatives) >= 3 and all(entry.sequence_id != sequence_id for entry in negatives)
        assert all(entry is existing for entry, existing in zip(negatives, moco.queue.negatives(sequence_id)))
        loss.register_hook(lambda grad: events.append('backward'))
        return loss, logits, negatives

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
    result = canary.run_canary(moco, RecordingProcessor(), sources, torch.device('cpu'), max_steps)
    expected_metadata = [(video, i // 4) for i, video in enumerate('abcd' * ((max_steps + 7) // 4))][:4 + max_steps]
    assert reader['reads'] == expected_metadata
    assert [(entry.sequence_id, entry.sequence_index) for entry in moco.queue.entries] == expected_metadata
    assert events == ['key', 'enqueue'] * 4 + ['query', 'key', 'loss', 'backward', 'optimizer', 'ema', 'enqueue'] * max_steps
    assert len(optimizers) == 1
    assert all(int(state['step']) == max_steps for state in optimizers[0].state.values())
    assert result['training_steps'] == max_steps and result['queue_count'] == 4 + max_steps
    assert result['changed_tensors']['query base'] == result['changed_tensors']['key base'] == 0
    for side in ('query', 'key'):
        base = getattr(moco, f'{side}_encoder').base
        assert torch.equal(base.weight, initial[f'{side}_encoder.base.weight'])
    assert sorted(reader['closed']) == list('abcd')
    steps = [row for row in records(capsys) if row['event'] == 'step']
    assert len(steps) == max_steps
    for index, row in enumerate(steps):
        assert row['step'] == index and row['queue_count'] == 5 + index
        assert row['queue_unique_sequence_id_count'] == 4 and row['valid_negative_count'] >= 3
        for kind in ('LoRA', 'Projector'):
            assert row['gradients'][f'query {kind}']['finite'] > 0
            assert row['gradients'][f'query {kind}']['nonzero'] > 0
            assert row['gradients'][f'query {kind}']['norm'] > 0


def test_real_vit_peft_and_public_loader_integration(encoder, sources, reader, capsys):
    # Actual 12-layer / width-768 PEFT fixture; reduced MLP only, no mocked forward.
    moco = ViTLoRAMoCo(encoder)
    result = canary.run_canary(moco, RecordingProcessor(), sources, torch.device('cpu'), 1)
    assert result['training_steps'] == 1 and result['queue_count'] == 5
    assert reader['reads'] == [('a', 0), ('b', 0), ('c', 0), ('d', 0), ('a', 1)]
    assert sorted(reader['closed']) == list('abcd')
    rows = records(capsys)
    optimizer = next(row for row in rows if row['event'] == 'optimizer')
    assert optimizer['parameters'] == 983_936 and optimizer['tensors'] == 52
    step = next(row for row in rows if row['event'] == 'step')
    assert step['gradients']['query LoRA']['finite'] == 48


@pytest.mark.parametrize('terminal,target,success', [(('a', 0), 10, False), (('b', 1), 10, False), (('b', 1), 2, True)])
def test_eof_never_claims_unreached_target(small_moco, sources, reader, terminal, target, success, capsys):
    reader['eof'] = terminal
    if success:
        assert canary.run_canary(small_moco, RecordingProcessor(), sources, torch.device('cpu'), target)['training_steps'] == target
    else:
        with pytest.raises(RuntimeError, match='EOF after'):
            canary.run_canary(small_moco, RecordingProcessor(), sources, torch.device('cpu'), target)
    assert reader['reads'][-1] == terminal
    assert sorted(reader['closed']) == sorted(reader['opened'])
    rows = records(capsys)
    assert any(row['event'] == 'complete' for row in rows) == success
    if not success:
        assert rows[-1]['event'] == 'stopped'
        assert rows[-1]['training_steps'] == (0 if terminal[1] == 0 else 2)


@pytest.mark.parametrize('failure', ['query_forward', 'key_forward', 'loss', 'optimizer', 'ema', 'enqueue'])
def test_training_exception_closes_four_readers(small_moco, sources, reader, monkeypatch, failure):
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
            if calls == 5:
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
        canary.run_canary(small_moco, RecordingProcessor(), sources, torch.device('cpu'), 10)
    assert sorted(reader['closed']) == list('abcd')


@pytest.mark.parametrize('failure', ['zero_lora', 'zero_projector', 'nan_gradient', 'base_gradient', 'key_gradient',
                                    'base_update', 'bad_ema', 'nan_parameter', 'same_sequence', 'current_positive',
                                    'queue_metadata', 'queue_key', 'nan_loss', 'nan_logits', 'optimizer_set', 'no_update'])
def test_audits_stop_invalid_training(small_moco, sources, reader, monkeypatch, failure):
    moco = small_moco
    if failure in ('zero_lora', 'zero_projector', 'nan_gradient', 'base_gradient', 'key_gradient'):
        for p in moco.query_parameters():
            def hook(grad):
                if failure == 'base_gradient':
                    moco.query_encoder.base.weight.grad = torch.ones_like(moco.query_encoder.base.weight)
                elif failure == 'key_gradient':
                    next(moco.key_projector.parameters()).grad = torch.ones_like(next(moco.key_projector.parameters()))
                return torch.full_like(grad, float('nan')) if failure == 'nan_gradient' else grad
            p.register_hook(hook)
        if failure.startswith('zero_'):
            parameters = lora_parameters(moco.query_encoder).values() if failure == 'zero_lora' else moco.query_projector.parameters()
            for p in parameters:
                p.register_hook(lambda grad: torch.zeros_like(grad))
    elif failure in ('base_update', 'nan_parameter', 'no_update'):
        original = torch.optim.AdamW.step

        def step(optimizer):
            if failure == 'no_update':
                return
            original(optimizer)
            with torch.no_grad():
                if failure == 'base_update':
                    moco.query_encoder.base.weight.add_(1)
                else:
                    next(moco.query_projector.parameters()).fill_(float('nan'))
        monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    elif failure == 'bad_ema':
        monkeypatch.setattr(moco, 'update_key', lambda: None)
    elif failure == 'optimizer_set':
        original = torch.optim.AdamW
        monkeypatch.setattr(torch.optim, 'AdamW', lambda parameters, **kwargs:
                            original(list(parameters)[1:], **kwargs))
    else:
        original = moco.contrastive_loss

        def loss(query, key, sequence_id):
            value, logits, negatives = original(query, key, sequence_id)
            if failure == 'same_sequence':
                negatives = moco.queue.entries
            elif failure == 'current_positive':
                moco.queue.enqueue(key, sequence_id, 1)
            elif failure == 'queue_metadata':
                moco.queue._entries[0] = replace(moco.queue.entries[0], sequence_index=999)
            elif failure == 'queue_key':
                moco.queue.entries[0].key.add_(1)
            elif failure == 'nan_loss':
                value = value * float('nan')
            elif failure == 'nan_logits':
                logits = logits * float('nan')
            return value, logits, negatives
        monkeypatch.setattr(moco, 'contrastive_loss', loss)
    with pytest.raises(RuntimeError):
        canary.run_canary(moco, RecordingProcessor(), sources, torch.device('cpu'), 1)
    assert sorted(reader['closed']) == list('abcd')


@pytest.mark.parametrize('max_steps', [0, -1, 4093])
def test_cli_rejects_invalid_steps_before_provenance(monkeypatch, max_steps):
    audit = Mock()
    monkeypatch.setattr(smoke, 'audit_provenance', audit)
    monkeypatch.setattr('sys.argv', ['smoke', '/unused', '--max-steps', str(max_steps)])
    with pytest.raises(SystemExit) as error:
        smoke.main()
    assert error.value.code != 0
    audit.assert_not_called()


@pytest.mark.parametrize('source_count,duplicate', [(3, False), (4, True)])
def test_cli_rejects_invalid_sources_before_model(sources, monkeypatch, source_count, duplicate):
    chosen = sources[:source_count] if not duplicate else (sources[0],) * 4
    monkeypatch.setattr(smoke, 'audit_provenance', lambda: None)
    monkeypatch.setattr(smoke, 'git_output', Mock())
    monkeypatch.setattr('sys.argv', ['smoke', '/unused'])
    adapter = Mock()
    adapter.sequence_sources.return_value = chosen
    monkeypatch.setattr(sl, 'ActivityNetAdapter', Mock(return_value=adapter))
    model = Mock()
    monkeypatch.setattr(smoke, 'ViTLoRAFrameEncoder', model)
    with pytest.raises(ValueError, match='four distinct'):
        smoke.main()
    model.assert_not_called()


def test_cli_first_four_sources_default_steps_and_fresh_runs(sources, monkeypatch):
    extra = replace(sources[0], sequence_id='unused')
    adapter = Mock()
    adapter.sequence_sources.return_value = (*sources, extra)
    factory = Mock(return_value=adapter)
    monkeypatch.setattr(sl, 'ActivityNetAdapter', factory)
    monkeypatch.setattr(smoke, 'audit_provenance', Mock())
    git = Mock()
    monkeypatch.setattr(smoke, 'git_output', git)
    monkeypatch.setattr(smoke.AutoImageProcessor, 'from_pretrained', Mock(return_value=RecordingProcessor()))
    monkeypatch.setattr(smoke, 'ViTLoRAFrameEncoder', Mock(side_effect=lambda _: SmallEncoder()))
    run = Mock()
    monkeypatch.setattr(smoke, 'run_canary', run)
    for options in ([], ['--max-steps', '100']):
        monkeypatch.setattr('sys.argv', ['smoke', '/unused', *options])
        smoke.main()
    assert [call.args[-1] for call in run.call_args_list] == [10, 100]
    assert all(call.args[2] == sources for call in run.call_args_list)
    first, second = [call.args[0] for call in run.call_args_list]
    assert first is not second and first.queue is not second.queue
    assert len(first.queue) == len(second.queue) == 0
    assert all(call.args[1:] == ('merge-base', '--is-ancestor', smoke.BASE_COMMIT, 'HEAD') for call in git.call_args_list)
    factory.assert_called_with(dataset_root=Path('/unused'))
    adapter.sequence_sources.assert_called_with('training')
