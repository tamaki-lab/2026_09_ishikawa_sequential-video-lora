from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
import sequential_loader as sl
import torch

from sequential_moco_bridge import make_two_views, encode_query_view, encode_key_view
from test_activitynet_vit_clip_feature import RecordingProcessor, make_sample
from test_vit_lora_frame_encoder import encoder
from test_vit_lora_moco import moco, unit_key
from scripts.smoke import smoke_activitynet_vit_lora_moco_one_step as smoke


def asymmetric_sample():
    sample = make_sample()
    sample.frames[:2, :, :, 1] += 10
    return sample


def test_two_views_preserve_source_and_metadata_and_skip_padding(moco):
    sample = asymmetric_sample()
    frames = sample.frames.clone()
    metadata = {name: getattr(sample, name).clone() for name in ('frame_indices', 'timestamps', 'valid_mask')}
    query, key = make_two_views(sample)
    assert torch.equal(query.frames, frames)
    assert torch.equal(key.frames[:2], frames[:2].flip(-1))
    assert torch.equal(key.frames[2:], frames[2:])
    for view in (query, key):
        assert view.sequence_id == sample.sequence_id and view.sequence_index == sample.sequence_index
        assert view.evaluation_reference is sample.evaluation_reference
        for name, value in metadata.items():
            torch.testing.assert_close(getattr(view, name), value, rtol=0, atol=0, equal_nan=True)
    query_processor, key_processor = RecordingProcessor(), RecordingProcessor()
    query_result = encode_query_view(query, query_processor, moco, torch.device('cpu'))
    key_result = encode_key_view(key, key_processor, moco, torch.device('cpu'))
    assert torch.equal(query_processor.images, frames[:2])
    assert torch.equal(key_processor.images, frames[:2].flip(-1))
    assert torch.equal(sample.frames, frames)
    for result in (query_result, key_result):
        assert result[0].shape == (2, 3, 224, 224)
        assert result[1].shape == (16, 768) and result[1][2:].count_nonzero() == 0
        assert result[2].shape == (768,) and result[3].shape == (128,)
        assert all(torch.isfinite(value).all() for value in result)
    assert query_result[-1].requires_grad
    assert all(not value.requires_grad for value in key_result)


def test_real_peft_one_step_gradient_update_ema_and_enqueue_order(moco, monkeypatch, capsys):
    sample = asymmetric_sample()
    query_view, key_view = make_two_views(sample)
    smoke.audit_initial_state(moco)
    processor = RecordingProcessor()
    moco.queue.enqueue(unit_key(0), 'past-video', 0)
    query = encode_query_view(query_view, processor, moco, torch.device('cpu'))
    key = encode_key_view(key_view, processor, moco, torch.device('cpu'))
    query[1].retain_grad()
    query[2].retain_grad()
    initial_key = key[-1].clone()
    before = {name: p.detach().clone() for name, p in moco.named_parameters()}
    events = []
    adamw_step, update_key, enqueue, compute_loss = (
        torch.optim.AdamW.step, moco.update_key, moco.queue.enqueue, moco.contrastive_loss
    )

    def step(optimizer):
        events.append('optimizer')
        optimized = [p for group in optimizer.param_groups for p in group['params']]
        assert len(optimized) == 52
        assert {id(p) for p in optimized} == {id(p) for p in moco.parameters() if p.requires_grad}
        assert all(group['lr'] == 1e-3 and group['weight_decay'] == 0 for group in optimizer.param_groups)
        return adamw_step(optimizer)

    def ema():
        events.append('ema')
        assert len(moco.queue) == 1
        update_key()

    def append(value, sequence_id, sequence_index):
        events.append('enqueue')
        assert torch.equal(value, initial_key)
        enqueue(value, sequence_id, sequence_index)

    def loss(*args):
        events.append('loss')
        assert len(moco.queue) == 1 and moco.queue.entries[0].sequence_id == 'past-video'
        return compute_loss(*args)

    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    monkeypatch.setattr(moco, 'update_key', ema)
    monkeypatch.setattr(moco.queue, 'enqueue', append)
    monkeypatch.setattr(moco, 'contrastive_loss', loss)
    result = smoke.run_one_step(moco, query[-1], key[-1], sample)
    assert result.ndim == 0 and torch.isfinite(result)
    assert events == ['loss', 'optimizer', 'ema', 'enqueue']
    assert query[1].grad[2:].count_nonzero() == 0
    torch.testing.assert_close(query[1].grad[:2], (query[2].grad / 2).expand(2, -1))
    groups = smoke.parameter_groups(moco)
    for label, parameters in groups.items():
        changed = [name for name, p in parameters.items() if not torch.equal(before[name], p)]
        if label.endswith('base'):
            assert not changed and all(p.grad is None for p in parameters.values())
        else:
            assert changed
        if label.startswith('key'):
            assert all(p.grad is None for p in parameters.values())
        if label in ('query LoRA', 'query Projector'):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values())
            assert any(p.grad.count_nonzero() for p in parameters.values())
    for label in ('key LoRA', 'key Projector'):
        for name, p in groups[label].items():
            updated_query = dict(moco.named_parameters())[name.replace('key_', 'query_', 1)]
            torch.testing.assert_close(p, 0.999 * before[name] + 0.001 * updated_query, rtol=1e-6, atol=1e-8)
    assert len(moco.queue) == 2
    assert [entry.sequence_id for entry in moco.queue.entries] == ['past-video', sample.sequence_id]
    assert torch.equal(moco.queue.entries[-1].key, initial_key)
    assert not moco.queue.entries[-1].key.requires_grad
    assert all(torch.isfinite(p).all() for p in moco.parameters())
    assert 'all parameters finite: True' in capsys.readouterr().out


@pytest.mark.parametrize('failure', ['base_trainable', 'key_trainable', 'projector_frozen',
                                    'projector_shape', 'key_initial_state', 'lora_config', 'shared_storage'])
def test_initial_audit_rejects_wrong_contract(moco, failure):
    if failure == 'base_trainable':
        next(p for p in moco.query_encoder.parameters() if not p.requires_grad).requires_grad_(True)
    elif failure == 'key_trainable':
        next(moco.key_encoder.parameters()).requires_grad_(True)
    elif failure == 'projector_frozen':
        next(moco.query_projector.parameters()).requires_grad_(False)
    elif failure == 'projector_shape':
        moco.query_projector[2] = torch.nn.Linear(768, 127)
    elif failure == 'key_initial_state':
        with torch.no_grad():
            next(moco.key_projector.parameters()).add_(1)
    elif failure == 'lora_config':
        moco.key_encoder.vit.peft_config['default'].lora_alpha = 16
    else:
        moco.key_projector[0].weight = torch.nn.Parameter(moco.query_projector[0].weight.detach(), requires_grad=False)
    with pytest.raises(RuntimeError):
        smoke.audit_initial_state(moco)


@pytest.mark.parametrize('failure', ['zero_gradient', 'nonfinite_gradient', 'no_update',
                                    'base_update', 'bad_ema', 'key_base_ema', 'nonfinite_update'])
def test_step_audit_rejects_invalid_gradients_updates_and_ema(moco, monkeypatch, failure):
    sample = asymmetric_sample()
    query = encode_query_view(sample, RecordingProcessor(), moco, torch.device('cpu'))[-1]
    key = moco.project_key(torch.randn(768))
    moco.queue.enqueue(unit_key(0), 'past-video', 0)
    if failure in ('zero_gradient', 'nonfinite_gradient'):
        value = 0. if failure == 'zero_gradient' else float('nan')
        for p in moco.query_parameters():
            p.register_hook(lambda grad: torch.full_like(grad, value))
    elif failure in ('bad_ema', 'key_base_ema'):
        original_ema = moco.update_key

        def bad_ema():
            if failure == 'key_base_ema':
                original_ema()
                with torch.no_grad():
                    next(moco.key_encoder.parameters()).add_(1)

        monkeypatch.setattr(moco, 'update_key', bad_ema)
    else:
        original_step = torch.optim.AdamW.step

        def bad_step(optimizer):
            if failure == 'no_update':
                return
            original_step(optimizer)
            with torch.no_grad():
                if failure == 'base_update':
                    next(p for p in moco.query_encoder.parameters() if not p.requires_grad).add_(1)
                else:
                    next(moco.query_projector.parameters()).fill_(float('nan'))

        monkeypatch.setattr(torch.optim.AdamW, 'step', bad_step)
    with pytest.raises(RuntimeError):
        smoke.run_one_step(moco, query, key, sample)
    assert len(moco.queue) == 1


def valid_git_output(root, *args):
    is_loader = root.name.endswith('sequential_loader')
    if args == ('branch', '--show-current'):
        return smoke.LOADER_BRANCH if is_loader else smoke.EXPECTED_BRANCH
    if args == ('rev-parse', 'HEAD'):
        return smoke.LOADER_COMMIT if is_loader else smoke.BASE_COMMIT
    if args == ('remote', 'get-url', 'origin'):
        return 'git@github.com:tamaki-lab/2026_09_ishikawa_sequential-video-lora.git'
    return ''


@pytest.mark.parametrize('failure', [None, 'decode-a', 'decode-b', 'forward'])
def test_smoke_two_independent_sources_first_chunks_and_reader_cleanup(encoder, monkeypatch, capsys, failure):
    sources = tuple(sl.SequenceSource(
        sequence_id=video, source_id=video, source=Path(f'{video}.mp4'),
        start_frame=0, stop_frame=None, evaluation_reference=object(),
    ) for video in ('a', 'b', 'unused'))
    opened, closed, requests = [], [], []

    class Reader:
        @contextmanager
        def open(self, source):
            self.source = source
            opened.append(source.sequence_id)
            try:
                yield self
            finally:
                closed.append(source.sequence_id)

        def read(self, chunk):
            requests.append((self.source.sequence_id, chunk.start_frame, chunk.valid_length))
            if failure == f'decode-{self.source.sequence_id}':
                raise RuntimeError('decode failure')
            sample = asymmetric_sample()
            frames = sample.frames[:2].repeat(8, 1, 1, 1)
            if self.source.sequence_id == 'b':
                frames += 30
            return sl.DecodedChunk(frames=frames, frame_indices=torch.arange(16),
                                   timestamps=torch.arange(16, dtype=torch.float64), reached_eof=False)

    def forward_hook(module, inputs):
        assert closed == ['a', 'b']
        if failure == 'forward':
            raise RuntimeError('forward failure')

    encoder.register_forward_pre_hook(forward_hook)
    monkeypatch.setattr(smoke, 'git_output', valid_git_output)
    monkeypatch.setattr('sys.argv', ['smoke', '/unused/ActivityNet'])
    adapter = Mock()
    adapter.sequence_sources.return_value = sources
    factory = Mock(return_value=adapter)
    monkeypatch.setattr(sl, 'ActivityNetAdapter', factory)
    monkeypatch.setattr(sl, 'SequentialVideoReader', Reader)
    monkeypatch.setattr(smoke.AutoImageProcessor, 'from_pretrained', Mock(return_value=RecordingProcessor()))
    monkeypatch.setattr(smoke, 'ViTLoRAFrameEncoder', Mock(return_value=encoder))
    if failure:
        with pytest.raises(RuntimeError, match='sequential iteration failed' if failure.startswith('decode') else 'forward failure'):
            smoke.main()
        assert opened == closed == (['a'] if failure == 'decode-a' else ['a', 'b'])
    else:
        smoke.main()
        assert opened == closed == ['a', 'b']
        assert requests == [('a', 0, 16), ('b', 0, 16)]
        factory.assert_called_once_with(dataset_root=Path('/unused/ActivityNet'))
        adapter.sequence_sources.assert_called_once_with('training')
        output = capsys.readouterr().out
        assert 'queue count after warm-up: 1' in output
        assert "negative sequence_ids: ['a']" in output
        assert 'queue count after training enqueue: 2' in output
        assert 'Stage 5 one-step mechanics smoke: PASS' in output


@pytest.mark.parametrize('failure', ['branch', 'origin', 'ancestry', 'loader_branch', 'loader_revision',
                                    'dirty_loader', 'transformers', 'peft'])
def test_provenance_rejected_before_dataset_read(monkeypatch, failure):
    def git_output(root, *args):
        is_loader = root.name.endswith('sequential_loader')
        if args == ('branch', '--show-current') and failure == ('loader_branch' if is_loader else 'branch'):
            return 'wrong'
        if args == ('rev-parse', 'HEAD') and is_loader and failure == 'loader_revision':
            return 'wrong'
        if args[0] == 'remote' and failure == 'origin':
            return 'wrong'
        if args[0] == 'merge-base' and failure == 'ancestry':
            raise RuntimeError('wrong ancestor')
        if args[0] == 'status' and failure == 'dirty_loader':
            return ' M source.py'
        return valid_git_output(root, *args)

    monkeypatch.setattr(smoke, 'git_output', git_output)
    monkeypatch.setattr('sys.argv', ['smoke', '/unused/ActivityNet'])
    if failure in ('transformers', 'peft'):
        monkeypatch.setattr(getattr(smoke, failure), '__version__', '0.0.0')
    factory = Mock()
    monkeypatch.setattr(sl, 'ActivityNetAdapter', factory)
    with pytest.raises(RuntimeError):
        smoke.main()
    factory.assert_not_called()
