"""Protocol selection and early validation for the common Streaming MoCo CLI."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import sequential_loader as sl
import torch
from torch import nn

from scripts.smoke import smoke_activitynet_streaming_moco as smoke
from training.moco_protocol import STAGE6A_PROTOCOL, STAGE6B_PROTOCOL


@pytest.fixture
def cli(monkeypatch):
    sources = tuple(sl.SequenceSource(
        sequence_id=video, source_id=video, source=Path(f'{video}.mp4'),
        start_frame=0, stop_frame=None, evaluation_reference=object(),
    ) for video in ('a', 'b', 'c', 'd', 'unused'))
    adapter = Mock()
    adapter.sequence_sources.return_value = sources
    factory = Mock(return_value=adapter)
    monkeypatch.setattr(sl, 'ActivityNetAdapter', factory)
    audit, git = Mock(), Mock()
    monkeypatch.setattr(smoke, 'audit_provenance', audit)
    monkeypatch.setattr(smoke, 'git_output', git)
    processor = Mock()
    monkeypatch.setattr(smoke.AutoImageProcessor, 'from_pretrained', processor)
    encoder = Mock(side_effect=lambda _: nn.Identity())
    monkeypatch.setattr(smoke, 'ViTLoRAFrameEncoder', encoder)
    run = Mock()
    monkeypatch.setattr(smoke, 'run_streaming_moco', run)
    return dict(sources=sources, adapter=adapter, factory=factory, audit=audit, git=git,
                processor=processor, encoder=encoder, run=run)


def invoke(monkeypatch, *options):
    monkeypatch.setattr('sys.argv', ['smoke', '/unused', *options])
    smoke.main()


@pytest.mark.parametrize('options,protocol', [
    ((), STAGE6B_PROTOCOL),
    (('--preset', 'stage6b'), STAGE6B_PROTOCOL),
    (('--preset', 'stage6a'), STAGE6A_PROTOCOL),
])
def test_presets_select_first_sources_and_log_resolved_protocol(cli, monkeypatch, capsys, options, protocol):
    invoke(monkeypatch, *options)
    call = cli['run'].call_args
    assert call.kwargs == {'protocol': protocol}
    assert call.args[2] == cli['sources'][:protocol.source_count]
    assert call.args[3] == torch.device('cpu')
    assert call.args[4] == 10
    assert call.args[0].training
    cli['factory'].assert_called_once_with(dataset_root=Path('/unused'))
    cli['adapter'].sequence_sources.assert_called_once_with('training')
    cli['processor'].assert_called_once_with(smoke.CHECKPOINT_ID)
    cli['encoder'].assert_called_once_with(smoke.CHECKPOINT_ID)
    output = capsys.readouterr().out
    for axis in ('stream_mode', 'key_transform', 'negative_policy', 'source_count', 'warmup_count'):
        assert f'{axis}: {getattr(protocol, axis)}' in output
    assert 'fresh state: True' in output
    assert 'Streaming MoCo 10-step mechanics: PASS' in output


@pytest.mark.parametrize('preset,axis,value', [
    ('stage6a', 'stream_mode', 'strict_single'),
    ('stage6a', 'key_transform', 'gbr_horizontal_flip'),
    ('stage6a', 'negative_policy', 'all_past'),
    ('stage6b', 'stream_mode', 'round_robin'),
    ('stage6b', 'key_transform', 'horizontal_flip'),
    ('stage6b', 'negative_policy', 'different_sequence'),
])
def test_axis_override_changes_only_selected_axis(cli, monkeypatch, preset, axis, value):
    invoke(monkeypatch, '--preset', preset, f'--{axis.replace("_", "-")}', value)
    base = STAGE6A_PROTOCOL if preset == 'stage6a' else STAGE6B_PROTOCOL
    protocol = replace(base, **{axis: value})
    assert cli['run'].call_args.kwargs == {'protocol': protocol}
    assert cli['run'].call_args.args[2] == cli['sources'][:protocol.source_count]


def test_all_axes_can_override_a_preset_together(cli, monkeypatch):
    invoke(monkeypatch, '--preset', 'stage6a', '--stream-mode', 'strict_single',
           '--key-transform', 'gbr_horizontal_flip', '--negative-policy', 'all_past')
    assert cli['run'].call_args.kwargs == {'protocol': STAGE6B_PROTOCOL}


def test_fresh_runs_and_steps_beyond_queue_capacity(cli, monkeypatch):
    for options in ((), ('--max-steps', '4097')):
        invoke(monkeypatch, *options)
    calls = cli['run'].call_args_list
    assert [call.args[4] for call in calls] == [10, 4097]
    first, second = [call.args[0] for call in calls]
    assert first is not second and first.queue is not second.queue
    assert len(first.queue) == len(second.queue) == 0
    assert all(parameter.grad is None for model in (first, second) for parameter in model.parameters())
    assert cli['audit'].call_count == 2
    assert cli['git'].call_count == 2
    assert all(call.args[1:] == ('merge-base', '--is-ancestor', smoke.BASE_COMMIT, 'HEAD')
               for call in cli['git'].call_args_list)
    assert smoke.BASE_COMMIT == '4835b5736f0b1dcc9962cbeffd85e880010311ea'


@pytest.mark.parametrize('options', [
    ('--max-steps', '0'), ('--max-steps', '-1'), ('--max-steps', '1.5'),
    ('--preset', 'unknown'), ('--stream-mode', 'unknown'),
    ('--key-transform', 'unknown'), ('--negative-policy', 'unknown'),
])
def test_invalid_options_fail_before_provenance_or_dataset(cli, monkeypatch, options):
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, *options)
    assert error.value.code != 0
    cli['audit'].assert_not_called()
    cli['factory'].assert_not_called()
    cli['processor'].assert_not_called()
    cli['encoder'].assert_not_called()


@pytest.mark.parametrize('options,count,duplicate', [
    ((), 0, False),
    (('--preset', 'stage6a'), 3, False),
    (('--stream-mode', 'round_robin'), 4, True),
])
def test_invalid_source_count_or_identity_fails_before_model(cli, monkeypatch, options, count, duplicate):
    sources = cli['sources'][:count] if not duplicate else (cli['sources'][0],) * count
    cli['adapter'].sequence_sources.return_value = sources
    with pytest.raises(ValueError):
        invoke(monkeypatch, *options)
    cli['processor'].assert_not_called()
    cli['encoder'].assert_not_called()
    cli['run'].assert_not_called()


@pytest.mark.parametrize('gate', ['audit', 'git'])
def test_provenance_failure_precedes_dataset_access(cli, monkeypatch, gate):
    cli[gate].side_effect = RuntimeError('provenance mismatch')
    with pytest.raises(RuntimeError, match='provenance mismatch'):
        invoke(monkeypatch)
    cli['factory'].assert_not_called()
    cli['processor'].assert_not_called()
    cli['encoder'].assert_not_called()
    cli['run'].assert_not_called()


def test_engine_failure_is_not_reported_as_pass(cli, monkeypatch, capsys):
    cli['run'].side_effect = RuntimeError('EOF before training target')
    with pytest.raises(RuntimeError, match='EOF'):
        invoke(monkeypatch)
    assert 'PASS' not in capsys.readouterr().out
