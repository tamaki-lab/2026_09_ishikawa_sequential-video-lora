"""Full-dataset MoCo CLI: explicit fresh / resume and the source-count gate."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import sequential_loader as sl
from hydra import compose, initialize_config_dir
from torch import nn

from scripts.moco import train_full_streaming_moco as cli_module


CONFIG_ROOT = Path(__file__).resolve().parents[2] / 'conf'


@pytest.fixture
def cli(monkeypatch, tmp_path):
    events = []
    sources = tuple(sl.SequenceSource(sequence_id=f'v{index}', source_id=f'v{index}', source=Path('x.mp4'),
                                      start_frame=0, stop_frame=None) for index in range(10_024))
    adapter = Mock()
    adapter.sequence_sources.return_value = sources
    monkeypatch.setattr(sl, 'ActivityNetAdapter', Mock(return_value=adapter))
    provenance = Mock(return_value={
        'implementation': {'dirty': False}, 'sequential_loader': {}, 'versions': {},
    })
    monkeypatch.setattr(cli_module, 'collect_provenance', provenance)
    monkeypatch.setattr(cli_module, 'seed_all', lambda seed: events.append(('seed', seed)))
    monkeypatch.setattr(cli_module.AutoImageProcessor, 'from_pretrained', Mock())
    monkeypatch.setattr(
        cli_module, 'ViTLoRAFrameEncoder',
        Mock(side_effect=lambda *args, **kwargs: events.append(('model', None)) or nn.Identity()),
    )
    monkeypatch.setattr(cli_module, 'ViTLoRAMoCo', Mock(return_value=nn.Linear(1, 1)))
    run = Mock(return_value={'status': 'paused'})
    monkeypatch.setattr(cli_module.full, 'run_full_streaming_moco', run)

    def invoke(*overrides):
        with initialize_config_dir(version_base='1.3', config_dir=str(CONFIG_ROOT)):
            cfg = compose(config_name='moco_full', overrides=[
                'runtime.dataset_root=/anet', 'runtime.seed=7', 'runtime.device=cpu',
                'logging.disable_comet=true', f'runtime.output_root={tmp_path}', *overrides,
            ])
        cli_module.run(cfg)
    return dict(invoke=invoke, run=run, adapter=adapter, sources=sources, root=tmp_path,
                events=events, provenance=provenance)


def test_fresh_run_uses_all_training_sources_in_adapter_order(cli, capsys):
    cli['invoke']('runtime.run_id=run1', 'runtime.stop_after_videos=2')
    cli['adapter'].sequence_sources.assert_called_once_with('training')
    call = cli['run'].call_args
    assert call.args[2] == cli['sources'] and call.args[4] == cli['root'] / 'run1'
    assert call.kwargs['resume'] is False and call.kwargs['stop_after_videos'] == 2
    assert call.kwargs['experiment'] is None and call.kwargs['run_id'] == 'run1'
    config = json.loads(capsys.readouterr().out.split('Resolved run config: ')[1].splitlines()[0])
    assert config['protocol'] == {'stream_mode': 'strict_single', 'key_transform': 'gbr_horizontal_flip',
                                  'negative_policy': 'all_past'}
    assert config['intervals'] == {'resume_videos': 100, 'snapshot_videos': 1000, 'comet_metric_updates': 100}
    assert config['seed'] == 7 and config['device_identity'] == {'type': 'cpu'}
    assert cli['events'].index(('seed', 7)) < cli['events'].index(('model', None))
    assert cli['provenance'].call_args.kwargs['require_clean'] is True
    assert cli['provenance'].call_args.kwargs['policy']['repository'].startswith('tamaki-lab/')


def test_resume_is_explicit_and_reads_saved_experiment(cli):
    (cli['root'] / 'run1').mkdir()
    (cli['root'] / 'run1' / 'run_metadata.json').write_text(json.dumps({'moco_experiment_key': None}))
    cli['invoke']('runtime.run_id=run1', 'runtime.resume=true')
    assert cli['run'].call_args.kwargs['resume'] is True
    assert 'resumes' in json.loads((cli['root'] / 'run1' / 'run_metadata.json').read_text())


def test_scientific_override_is_explicitly_nonproduction(cli):
    cli['invoke']('runtime.run_id=custom', 'moco.optimizer.lr=0.002')
    assert cli['provenance'].call_args.kwargs['require_clean'] is False
    settings = cli['run'].call_args.kwargs['config']
    assert settings.moco.optimizer.lr == 0.002
    assert settings.production_groups_match() is False


@pytest.mark.parametrize('options', [
    ('runtime.run_id=../x',), ('runtime.run_id=a', 'runtime.stop_after_videos=0'),
    ('runtime.run_id=a', 'runtime.seed=-1'),
])
def test_invalid_options_stop_before_work(cli, options):
    with pytest.raises(ValueError):
        cli['invoke'](*options)
    cli['run'].assert_not_called()


def test_wrong_source_count_stops_before_model(cli):
    cli['adapter'].sequence_sources.return_value = cli['sources'][:-1]
    with pytest.raises(ValueError, match='10024'):
        cli['invoke']('runtime.run_id=run1')
    cli_module.ViTLoRAMoCo.assert_not_called()
