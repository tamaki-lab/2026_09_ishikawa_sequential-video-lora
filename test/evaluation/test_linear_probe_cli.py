"""End-to-end CLI plumbing on synthetic data: manifest -> audit -> features -> probe."""

import json
from pathlib import Path

import pytest
import sequential_loader as sl
import torch
from hydra import compose, initialize_config_dir

from anet_synthetic import ColorEncoder, RecordingProcessor, chunk_times
from scripts.linear_probe import build_manifest, extract_features, run_probe
from utils.artifact_io import sha256_file


PROVENANCE = {'implementation': {'repository': 'repo', 'branch': 'dev', 'commit': 'abc', 'dirty': False,
                                 'tracked_diff_sha256': 'clean'},
              'sequential_loader': {'branch': 'ActivityNet', 'commit': 'loader', 'dirty': False},
              'versions': {'python': 'test'}}
CONFIG_ROOT = Path(__file__).resolve().parents[2] / 'conf'


def entry(subset, *annotations):
    return {'subset': subset, 'annotations': [{'label': label, 'segment': [start, end]}
                                              for label, start, end in annotations]}


@pytest.fixture
def cli(dataset, monkeypatch, tmp_path):
    sources = dataset({
        'a': entry('training', ('A', 0., 3.1), ('B', 1.6, 3.4), ('A', 2., 2.5)),
        'b': entry('training', ('B', 0., 1.5)),
        'c': entry('validation', ('A', 0., 1.5), ('B', 1.6, 9.)),
    }, {'a': chunk_times(4), 'b': chunk_times(2), 'c': chunk_times(3)})

    class Adapter:
        def __init__(self, dataset_root):
            pass

        def sequence_sources(self, split):
            return sources[split]

    monkeypatch.setattr(sl, 'ActivityNetAdapter', Adapter)
    for module in (build_manifest, extract_features, run_probe):
        monkeypatch.setattr(module, 'collect_provenance', lambda **kwargs: PROVENANCE)
    monkeypatch.setattr(extract_features.AutoImageProcessor, 'from_pretrained', lambda _: RecordingProcessor())

    def encoder(condition, snapshot, manifest_metadata, encoder_config, production, expected_source_count):
        info = {'type': condition, 'base_model': 'base', 'base_fingerprint': 'shared-base'}
        if condition == 'moco_query_lora_final':
            info['snapshot'] = {'final': True, 'run_id': 'r', 'files': {}}
        return ColorEncoder(scale=1. if condition == 'base_vit' else 2.), info
    monkeypatch.setattr(extract_features, 'build_encoder', encoder)
    root = tmp_path / 'out'

    config_names = {
        build_manifest: 'linear_probe_manifest',
        extract_features: 'linear_probe_features',
        run_probe: 'linear_probe_run',
    }

    def invoke(module, *overrides):
        common = [
            'activitynet.class_count=2', 'activitynet.expected_source_counts.training=2',
            'activitynet.expected_source_counts.validation=1', f'runtime.output_root={root}',
            'logging.disable_comet=true',
        ]
        with initialize_config_dir(version_base='1.3', config_dir=str(CONFIG_ROOT)):
            cfg = compose(config_name=config_names[module], overrides=[*common, *map(str, overrides)])
        return module.run(cfg)
    return root, invoke


def test_pipeline_gates_hashes_and_reuse(cli, capsys):
    root, invoke = cli
    manifest_args = ('runtime.dataset_root=/anet', 'runtime.manifest_id=m')
    invoke(build_manifest, 'runtime.command=build', *manifest_args)
    metadata = json.loads((root / 'manifest' / 'm' / 'metadata.json').read_text())
    assert metadata['gate']['status'] == 'PENDING_AUDIT' and not metadata['production']
    assert metadata['splits']['training']['skip_counts'] == {'A': 1}  # [2.0, 2.5] contains no whole chunk.
    with pytest.raises(FileExistsError):
        invoke(build_manifest, 'runtime.command=build', *manifest_args)
    with pytest.raises(RuntimeError, match='Gate has not passed'):
        invoke(extract_features, 'runtime.dataset_root=/anet', 'runtime.manifest_id=m',
               'runtime.condition=base_vit', 'runtime.device=cpu')
    invoke(build_manifest, 'runtime.command=audit', *manifest_args)
    metadata = json.loads((root / 'manifest' / 'm' / 'metadata.json').read_text())
    assert metadata['gate']['status'] == 'SMOKE_PASS' and metadata['gate']['reproducibility']['matches']
    assert metadata['comet']['status'] == 'disabled' and metadata['comet']['retry_needed']

    # Dataset mount paths are placement-only: identical annotation/source
    # content remains reusable after relocation.
    invoke(extract_features, 'runtime.dataset_root=/different-anet', 'runtime.manifest_id=m',
           'runtime.condition=base_vit', 'runtime.device=cpu', 'runtime.feature_id=base_vit')
    invoke(extract_features, 'runtime.dataset_root=/anet', 'runtime.manifest_id=m',
           'runtime.condition=moco_query_lora_final', 'runtime.device=cpu',
           'runtime.feature_id=moco_query_lora_final', f'runtime.snapshot={root / "snap"}')
    invoke(extract_features, 'runtime.dataset_root=/anet', 'runtime.manifest_id=m',
           'runtime.condition=base_vit', 'runtime.device=cpu', 'runtime.feature_id=base_vit')
    assert 'Reusing verified feature artifact' in capsys.readouterr().out
    base = root / 'features' / 'base_vit' / 'base_vit'
    lora = root / 'features' / 'moco_query_lora_final' / 'moco_query_lora_final'
    for directory in (base, lora):
        stored = json.loads((directory / 'metadata.json').read_text())
        assert stored['manifest']['segment_manifest_sha256'] == metadata['segment_manifest_sha256']
        value = torch.load(directory / 'training' / 'features.pt', weights_only=True)
        assert value['segment_ids'] == ['training:a:0', 'training:a:1', 'training:b:0']

    probe_args = (
        'runtime.manifest_id=m', f'runtime.base_features={base}', f'runtime.lora_features={lora}',
        'runtime.result_id=smoke', 'linear_probe.probe.epochs=1',
    )
    invoke(run_probe, *probe_args)
    result_root = root / 'results' / 'smoke'
    aggregate = json.loads((result_root / 'comparison' / 'aggregate_summary.json').read_text())
    aggregate_sidecar = json.loads((result_root / 'comparison' / 'metadata.json').read_text())
    assert not aggregate['production'] and set(aggregate['conditions']) == {'base_vit', 'moco_query_lora_final'}
    assert 'comet' not in aggregate
    assert aggregate_sidecar['comet']['status'] == 'disabled' and aggregate_sidecar['comet']['retry_needed']
    assert aggregate_sidecar['files'] == {
        name: sha256_file(result_root / 'comparison' / name)
        for name in ('aggregate_summary.json', 'aggregate_summary.csv')
    }
    for condition in aggregate['conditions'].values():
        assert condition['seeds'] == [0, 1, 2] and condition['top1_std'] is not None
    for condition in ('base_vit', 'moco_query_lora_final'):
        for seed in (0, 1, 2):
            summary = json.loads((result_root / condition / f'seed-{seed}' / 'summary.json').read_text())
            assert summary['hyperparameters']['epochs'] == 1 and summary['validation_used_for_selection'] is False
            assert summary['sample_counts'] == {'training': 3, 'validation': 2}
    invoke(run_probe, *probe_args)
    assert 'Reusing identical aggregate result' in capsys.readouterr().out
    (result_root / 'comparison' / 'aggregate_summary.csv').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='Aggregate CSV differs'):
        invoke(run_probe, *probe_args)
    (result_root / 'comparison' / 'aggregate_summary.csv').unlink()
    with pytest.raises(RuntimeError, match='Different probe result'):
        invoke(run_probe, *probe_args[:-1], 'linear_probe.probe.epochs=2')


def test_feature_definition_override_is_rejected_before_extraction(cli):
    root, invoke = cli
    with pytest.raises(ValueError, match='implemented feature definition'):
        invoke(extract_features, 'runtime.dataset_root=/anet', 'runtime.manifest_id=m',
               'runtime.condition=base_vit', 'runtime.device=cpu',
               'linear_probe.feature_definition.normalization=l2')
    assert not (root / 'features').exists()
