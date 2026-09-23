from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import sequential_loader as sl
import torch
from torch import nn

from model import MaskedMeanClipAggregator
from sequential_vit_bridge import encode_chunk
import smoke_activitynet_vit_clip_feature as smoke


class RecordingProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == 'pt'
        self.images = images.clone()
        return {'pixel_values': images[:, :, :1, :1].float().expand(-1, 3, 224, 224).contiguous()}


def make_sample():
    frames = torch.full((16, 3, 2, 2), 200, dtype=torch.uint8)
    frames[0].fill_(3)
    frames[1].fill_(7)
    return sl.SequentialSample(
        frames=frames,
        frame_indices=torch.tensor([0, 1] + [-1] * 14),
        timestamps=torch.tensor([1., 0.] + [float('nan')] * 14, dtype=torch.float64),
        valid_mask=torch.tensor([True, True] + [False] * 14),
        sequence_id='video', sequence_index=0, is_first=True, is_last=True,
        source_id='video', evaluation_reference=object(),
    )


def test_shared_bridge_skips_padding_preserves_metadata_and_aggregates():
    sample = make_sample()
    indices = sample.frame_indices.clone()
    timestamps = sample.timestamps.clone()
    frames = sample.frames.clone()
    reference = sample.evaluation_reference
    processor = RecordingProcessor()
    encoder = Mock(side_effect=lambda pixels: pixels[:, 0, 0, 0, None].expand(-1, 768))
    pixels, valid_features, features = encode_chunk(sample, processor, encoder, torch.device('cpu'))
    output = MaskedMeanClipAggregator()(features, sample.valid_mask)
    assert processor.images[:, 0, 0, 0].tolist() == [3, 7]
    assert encoder.call_count == 1
    assert pixels.shape == (2, 3, 224, 224)
    assert valid_features.shape == (2, 768)
    assert features.shape == (16, 768)
    assert features[:, 0].tolist() == [3., 7.] + [0.] * 14
    torch.testing.assert_close(output, torch.full((768,), 5.))
    assert torch.equal(sample.frames, frames)
    assert torch.equal(sample.frame_indices, indices)
    torch.testing.assert_close(sample.timestamps, timestamps, rtol=0, atol=0, equal_nan=True)
    assert sample.evaluation_reference is reference


def test_bridge_rejects_all_invalid_before_processor_or_encoder():
    processor, encoder = Mock(), Mock()
    with pytest.raises(RuntimeError, match='no valid frames'):
        encode_chunk(SimpleNamespace(valid_mask=torch.zeros(16, dtype=torch.bool)),
                     processor, encoder, torch.device('cpu'))
    processor.assert_not_called()
    encoder.assert_not_called()


@pytest.mark.parametrize('kind', ['pixel_shape', 'feature_shape', 'nonfinite'])
def test_bridge_rejects_invalid_processor_or_encoder_output(kind):
    processor = RecordingProcessor()
    encoder = Mock(return_value=torch.zeros(2, 768))
    if kind == 'pixel_shape':
        processor = Mock(return_value={'pixel_values': torch.zeros(16, 3, 224, 224)})
    elif kind == 'feature_shape':
        encoder.return_value = torch.zeros(2, 767)
    else:
        encoder.return_value.fill_(float('nan'))
    with pytest.raises(RuntimeError):
        encode_chunk(make_sample(), processor, encoder, torch.device('cpu'))


@pytest.mark.parametrize('decode_error', [False, True])
def test_smoke_uses_strict_loader_closes_reader_and_runs_frozen_eval(monkeypatch, capsys, decode_error):
    closed = []
    requests = []
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
            if decode_error:
                raise RuntimeError('decode failure')
            return sl.DecodedChunk(
                frames=make_sample().frames[:2], frame_indices=torch.tensor([0, 1]),
                timestamps=torch.tensor([1., 0.], dtype=torch.float64), reached_eof=True,
            )

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.vit.weight.fill_(2.)
            self.vit.requires_grad_(False)
            self.calls = 0

        def forward(self, pixels):
            self.calls += 1
            assert closed == [True]
            assert not self.training and not self.vit.training
            assert not torch.is_grad_enabled()
            assert pixels.shape == (2, 3, 224, 224)
            return (pixels[:, 0, 0, 0, None] * self.vit.weight).expand(-1, 768)

    def git_output(root, *args):
        if args == ('branch', '--show-current'):
            return smoke.EXPECTED_BRANCH
        if args == ('rev-parse', 'HEAD'):
            return smoke.LOADER_COMMIT if root.name.endswith('sequential_loader') else smoke.BASE_COMMIT
        return ''

    monkeypatch.setattr(smoke, 'git_output', git_output)
    monkeypatch.setattr('sys.argv', ['smoke_activitynet_vit_clip_feature.py', '/unused/ActivityNet'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    adapter = Mock()
    adapter.sequence_sources.return_value = (source,)
    adapter_factory = Mock(return_value=adapter)
    monkeypatch.setattr(sl, 'ActivityNetAdapter', adapter_factory)
    monkeypatch.setattr(sl, 'SequentialVideoReader', Reader)
    processor = RecordingProcessor()
    load_processor = Mock(return_value=processor)
    monkeypatch.setattr(smoke.AutoImageProcessor, 'from_pretrained', load_processor)
    encoder = Encoder()
    make_encoder = Mock(return_value=encoder)
    monkeypatch.setattr(smoke, 'ViTFrameEncoder', make_encoder)
    aggregator = MaskedMeanClipAggregator()
    outputs = []

    def record_output(module, args, output):
        assert not torch.is_grad_enabled()
        outputs.append(output)

    aggregator.register_forward_hook(record_output)
    monkeypatch.setattr(smoke, 'MaskedMeanClipAggregator', lambda: aggregator)
    if decode_error:
        with pytest.raises(RuntimeError):
            smoke.main()
        assert closed == [True]
        load_processor.assert_not_called()
        make_encoder.assert_not_called()
        return

    smoke.main()
    assert closed == [True]
    assert len(requests) == 1
    assert requests[0].start_frame == 0 and requests[0].valid_length == 16
    adapter_factory.assert_called_once_with(dataset_root=Path('/unused/ActivityNet'))
    adapter.sequence_sources.assert_called_once_with('training')
    load_processor.assert_called_once_with(smoke.CHECKPOINT_ID)
    make_encoder.assert_called_once_with(smoke.CHECKPOINT_ID)
    assert encoder.calls == 1
    assert encoder.vit.weight.item() == 2. and encoder.vit.weight.grad is None
    assert len(outputs) == 1 and not outputs[0].requires_grad
    torch.testing.assert_close(outputs[0], torch.full((768,), 10.))
    log = capsys.readouterr().out
    assert 'clip_feature shape: (768,)' in log
    assert 'valid count / T: 2/16' in log
    assert 'padding rows zero: True' in log
    assert 'trainable backbone parameters: 0' in log


@pytest.mark.parametrize('failure', ['branch', 'loader_revision', 'dirty_loader'])
def test_smoke_rejects_wrong_provenance_before_reading_data(monkeypatch, failure):
    def git_output(root, *args):
        if args == ('branch', '--show-current'):
            return 'wrong' if failure == 'branch' else smoke.EXPECTED_BRANCH
        if args == ('rev-parse', 'HEAD'):
            return 'wrong' if failure == 'loader_revision' else smoke.LOADER_COMMIT
        if args[0] == 'status' and failure == 'dirty_loader':
            return ' M src/adapters/activitynet.py'
        return ''

    monkeypatch.setattr(smoke, 'git_output', git_output)
    monkeypatch.setattr('sys.argv', ['smoke_activitynet_vit_clip_feature.py', '/unused/ActivityNet'])
    adapter_factory = Mock()
    monkeypatch.setattr(sl, 'ActivityNetAdapter', adapter_factory)
    with pytest.raises(RuntimeError):
        smoke.main()
    adapter_factory.assert_not_called()
