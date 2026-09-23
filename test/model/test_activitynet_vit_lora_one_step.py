from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest
import sequential_loader as sl
import torch

from model.aggregators import MaskedMeanClipAggregator
from sequential_vit_bridge import encode_chunk
from test_vit_lora_frame_encoder import encoder
from test_activitynet_vit_clip_feature import RecordingProcessor, make_sample
import smoke_activitynet_vit_lora_one_step as smoke


def test_one_step_through_valid_scatter_and_masked_mean(encoder, monkeypatch, capsys):
    sample = make_sample()
    frames = sample.frames.clone()
    indices = sample.frame_indices.clone()
    timestamps = sample.timestamps.clone()
    processor = RecordingProcessor()
    pixels, valid_features, features = encode_chunk(sample, processor, encoder, torch.device('cpu'))
    features.retain_grad()
    clip = MaskedMeanClipAggregator()(features, sample.valid_mask)
    assert pixels.shape == (2, 3, 224, 224)
    assert valid_features.shape == (2, 768)
    assert features.shape == (16, 768)
    assert processor.images[:, 0, 0, 0].tolist() == [3, 7]
    assert torch.count_nonzero(features[2:]) == 0
    before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    adamw = torch.optim.AdamW
    optimizers = []

    def make_optimizer(parameters, **kwargs):
        assert kwargs == {'lr': 1.0e-3, 'weight_decay': 0.0}
        optimizer = adamw(parameters, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, 'AdamW', make_optimizer)
    smoke.run_one_step(encoder, clip)
    assert len(optimizers) == 1
    optimized = [p for group in optimizers[0].param_groups for p in group['params']]
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    assert len(optimized) == len(trainable) == 48
    assert {id(p) for p in optimized} == {id(p) for p in trainable}
    assert torch.count_nonzero(features.grad[2:]) == 0
    assert torch.count_nonzero(features.grad[:2]) > 0
    torch.testing.assert_close(features.grad[:2], (clip.detach() / 768).expand(2, -1))
    assert all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
    assert any(torch.count_nonzero(p.grad) > 0 for p in trainable if p.grad is not None)
    assert all(p.grad is None and torch.equal(before[name], p)
               for name, p in encoder.named_parameters() if not p.requires_grad)
    changed = [name for name, p in encoder.named_parameters() if not torch.equal(before[name], p)]
    assert changed and all('.lora_B.' in name for name in changed)
    assert all(torch.isfinite(p).all() for p in encoder.parameters())
    assert torch.equal(sample.frames, frames)
    assert torch.equal(sample.frame_indices, indices)
    torch.testing.assert_close(sample.timestamps, timestamps, rtol=0, atol=0, equal_nan=True)
    output = capsys.readouterr().out
    assert 'base changed parameter count: 0' in output
    assert 'loss finite: True' in output


@pytest.mark.parametrize('failure', ['base_trainable', 'adapter_frozen', 'missing_target', 'pooler'])
def test_parameter_audit_rejects_contract_violation_before_optimizer(encoder, monkeypatch, failure):
    if failure == 'base_trainable':
        next(p for p in encoder.parameters() if not p.requires_grad).requires_grad_(True)
    elif failure == 'adapter_frozen':
        next(p for p in encoder.parameters() if p.requires_grad).requires_grad_(False)
    elif failure == 'missing_target':
        target = next(module for module in encoder.vit.modules() if hasattr(module, 'q_proj'))
        target.q_proj = target.q_proj.base_layer
    else:
        encoder.vit.get_base_model().pooler = torch.nn.Identity()
    optimizer = Mock()
    monkeypatch.setattr(torch.optim, 'AdamW', optimizer)
    with pytest.raises(RuntimeError):
        smoke.run_one_step(encoder, torch.ones(768, requires_grad=True))
    optimizer.assert_not_called()


@pytest.mark.parametrize('failure', ['zero_gradient', 'nonfinite_gradient', 'no_update', 'base_update', 'nonfinite_update'])
def test_one_step_rejects_invalid_gradients_or_updates(encoder, monkeypatch, failure):
    clip = encoder(torch.ones(1, 3, 224, 224))[0]
    if failure in ('zero_gradient', 'nonfinite_gradient'):
        value = 0.0 if failure == 'zero_gradient' else float('nan')
        for p in encoder.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.full_like(grad, value))
    else:
        step = torch.optim.AdamW.step

        def bad_step(optimizer):
            if failure == 'no_update':
                return
            step(optimizer)
            with torch.no_grad():
                if failure == 'base_update':
                    next(p for p in encoder.parameters() if not p.requires_grad).add_(1)
                else:
                    next(p for p in encoder.parameters() if p.requires_grad).fill_(float('nan'))

        monkeypatch.setattr(torch.optim.AdamW, 'step', bad_step)
    with pytest.raises(RuntimeError):
        smoke.run_one_step(encoder, clip)


@pytest.mark.parametrize('failure', [None, 'decode', 'forward'])
def test_smoke_strict_loader_one_chunk_and_reader_cleanup(encoder, monkeypatch, capsys, failure):
    closed, requests = [], []
    source = sl.SequenceSource(
        sequence_id='video', source_id='video', source=Path('unused.mp4'),
        start_frame=0, stop_frame=None, evaluation_reference=object(),
    )

    class Reader:
        @contextmanager
        def open(self, received_source):
            assert received_source is source
            try:
                yield self
            finally:
                closed.append(True)

        def read(self, chunk):
            requests.append(chunk)
            if failure == 'decode':
                raise RuntimeError('decode failure')
            return sl.DecodedChunk(
                frames=make_sample().frames[:2], frame_indices=torch.tensor([0, 1]),
                timestamps=torch.tensor([1., 0.], dtype=torch.float64), reached_eof=True,
            )

    def git_output(root, *args):
        is_loader = root.name.endswith('sequential_loader')
        if args == ('branch', '--show-current'):
            return smoke.LOADER_BRANCH if is_loader else smoke.EXPECTED_BRANCH
        if args == ('rev-parse', 'HEAD'):
            return smoke.LOADER_COMMIT if is_loader else smoke.BASE_COMMIT
        return ''

    def before_forward(module, inputs):
        assert closed == [True]
        assert torch.is_grad_enabled() and module.training
        if failure == 'forward':
            raise RuntimeError('forward failure')

    encoder.register_forward_pre_hook(before_forward)
    monkeypatch.setattr(smoke, 'git_output', git_output)
    monkeypatch.setattr('sys.argv', ['smoke_activitynet_vit_lora_one_step.py', '/unused/ActivityNet'])
    adapter = Mock()
    adapter.sequence_sources.return_value = (source,)
    adapter_factory = Mock(return_value=adapter)
    monkeypatch.setattr(sl, 'ActivityNetAdapter', adapter_factory)
    monkeypatch.setattr(sl, 'SequentialVideoReader', Reader)
    processor = Mock(return_value=RecordingProcessor())
    monkeypatch.setattr(smoke.AutoImageProcessor, 'from_pretrained', processor)
    make_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(smoke, 'ViTLoRAFrameEncoder', make_encoder)
    if failure:
        message = 'sequential iteration failed' if failure == 'decode' else 'forward failure'
        with pytest.raises(RuntimeError, match=message) as error:
            smoke.main()
        if failure == 'decode':
            assert str(error.value.__cause__) == 'decode failure'
        assert closed == [True]
        if failure == 'decode':
            make_encoder.assert_not_called()
        return
    smoke.main()
    assert closed == [True] and len(requests) == 1
    assert requests[0].start_frame == 0 and requests[0].valid_length == 16
    adapter_factory.assert_called_once_with(dataset_root=Path('/unused/ActivityNet'))
    adapter.sequence_sources.assert_called_once_with('training')
    processor.assert_called_once_with(smoke.CHECKPOINT_ID)
    make_encoder.assert_called_once_with(smoke.CHECKPOINT_ID)
    output = capsys.readouterr().out
    assert 'valid count / T: 2/16' in output
    assert 'padding rows zero: True' in output
    assert 'base changed parameter count: 0' in output
    assert 'all parameters finite: True' in output


@pytest.mark.parametrize('failure', ['branch', 'loader_branch', 'loader_revision', 'dirty_loader', 'transformers', 'peft'])
def test_smoke_rejects_wrong_provenance_before_reading_data(monkeypatch, failure):
    def git_output(root, *args):
        is_loader = root.name.endswith('sequential_loader')
        if args == ('branch', '--show-current'):
            if failure == ('loader_branch' if is_loader else 'branch'):
                return 'wrong'
            return smoke.LOADER_BRANCH if is_loader else smoke.EXPECTED_BRANCH
        if args == ('rev-parse', 'HEAD'):
            return 'wrong' if failure == 'loader_revision' else smoke.LOADER_COMMIT
        if args[0] == 'status' and failure == 'dirty_loader':
            return ' M sequential_loader/adapters/activitynet.py'
        return ''

    monkeypatch.setattr(smoke, 'git_output', git_output)
    monkeypatch.setattr('sys.argv', ['smoke_activitynet_vit_lora_one_step.py', '/unused/ActivityNet'])
    if failure in ('transformers', 'peft'):
        monkeypatch.setattr(getattr(smoke, failure), '__version__', '0.0.0')
    adapter = Mock()
    monkeypatch.setattr(sl, 'ActivityNetAdapter', adapter)
    with pytest.raises(RuntimeError):
        smoke.main()
    adapter.assert_not_called()
