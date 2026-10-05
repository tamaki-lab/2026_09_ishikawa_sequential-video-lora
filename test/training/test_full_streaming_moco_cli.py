"""Full-dataset MoCo CLI: explicit fresh / resume and the source-count gate."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import sequential_loader as sl
from torch import nn

from scripts.moco import train_full_streaming_moco as cli_module


@pytest.fixture
def cli(monkeypatch, tmp_path):
    sources = tuple(sl.SequenceSource(sequence_id=f'v{index}', source_id=f'v{index}', source=Path('x.mp4'),
                                      start_frame=0, stop_frame=None) for index in range(10_024))
    adapter = Mock()
    adapter.sequence_sources.return_value = sources
    monkeypatch.setattr(sl, 'ActivityNetAdapter', Mock(return_value=adapter))
    monkeypatch.setattr(cli_module, 'collect_provenance', lambda: {'implementation': {}})
    monkeypatch.setattr(cli_module.AutoImageProcessor, 'from_pretrained', Mock())
    monkeypatch.setattr(cli_module, 'ViTLoRAFrameEncoder', Mock(side_effect=lambda _: nn.Identity()))
    monkeypatch.setattr(cli_module, 'ViTLoRAMoCo', Mock(return_value=nn.Linear(1, 1)))
    run = Mock(return_value={'status': 'paused'})
    monkeypatch.setattr(cli_module.full, 'run_full_streaming_moco', run)

    def invoke(*options):
        monkeypatch.setattr('sys.argv', ['cli', '/anet', '--device', 'cpu', '--disable-comet',
                                         '--output-root', str(tmp_path), *options])
        cli_module.main()
    return dict(invoke=invoke, run=run, adapter=adapter, sources=sources, root=tmp_path)


def test_fresh_run_uses_all_training_sources_in_adapter_order(cli, capsys):
    cli['invoke']('--run-id', 'run1', '--stop-after-videos', '2')
    cli['adapter'].sequence_sources.assert_called_once_with('training')
    call = cli['run'].call_args
    assert call.args[2] == cli['sources'] and call.args[4] == cli['root'] / 'run1'
    assert call.kwargs['resume'] is False and call.kwargs['stop_after_videos'] == 2
    assert call.kwargs['experiment'] is None and call.kwargs['run_id'] == 'run1'
    config = json.loads(capsys.readouterr().out.split('Resolved config: ')[1].splitlines()[0])
    assert config['protocol'] == {'stream_mode': 'strict_single', 'key_transform': 'gbr_horizontal_flip',
                                  'negative_policy': 'all_past'}
    assert config['intervals'] == {'resume_videos': 100, 'snapshot_videos': 1000, 'comet_metric_updates': 100}


def test_resume_is_explicit_and_reads_saved_experiment(cli):
    (cli['root'] / 'run1').mkdir()
    (cli['root'] / 'run1' / 'run_metadata.json').write_text(json.dumps({'moco_experiment_key': None}))
    cli['invoke']('--run-id', 'run1', '--resume')
    assert cli['run'].call_args.kwargs['resume'] is True
    assert 'resumes' in json.loads((cli['root'] / 'run1' / 'run_metadata.json').read_text())


@pytest.mark.parametrize('options', [('--run-id', '../x'), ('--run-id', 'a', '--stop-after-videos', '0')])
def test_invalid_options_stop_before_work(cli, options):
    with pytest.raises(SystemExit):
        cli['invoke'](*options)
    cli['run'].assert_not_called()


def test_wrong_source_count_stops_before_model(cli):
    cli['adapter'].sequence_sources.return_value = cli['sources'][:-1]
    with pytest.raises(ValueError, match='10024'):
        cli['invoke']('--run-id', 'run1')
    cli_module.ViTLoRAMoCo.assert_not_called()
