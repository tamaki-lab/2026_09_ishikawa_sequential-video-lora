"""End-to-end CLI plumbing on synthetic data: manifest -> audit -> features -> probe."""

import json

import pytest
import sequential_loader as sl
import torch

from anet_synthetic import ColorEncoder, RecordingProcessor, chunk_times
from evaluation import activitynet_manifest as manifest
from scripts.linear_probe import build_manifest, extract_features, run_probe
from utils.artifact_io import sha256_file


PROVENANCE = {'implementation': {'repository': 'repo', 'branch': 'dev', 'commit': 'abc', 'dirty': False,
                                 'tracked_diff_sha256': 'clean'},
              'sequential_loader': {'branch': 'ActivityNet', 'commit': 'loader', 'dirty': False},
              'versions': {'python': 'test'}}


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
    monkeypatch.setattr(manifest, 'CLASS_COUNT', 2)
    monkeypatch.setattr(manifest, 'SPLIT_COUNTS', {'training': 2, 'validation': 1})
    for module in (build_manifest, extract_features, run_probe):
        monkeypatch.setattr(module, 'collect_provenance', lambda **kwargs: PROVENANCE)
    monkeypatch.setattr(extract_features.AutoImageProcessor, 'from_pretrained', lambda _: RecordingProcessor())

    def encoder(condition, snapshot, manifest_metadata):
        info = {'type': condition, 'base_model': 'base', 'base_fingerprint': 'shared-base'}
        if condition == 'moco_query_lora_final':
            info['snapshot'] = {'path': str(snapshot), 'final': True, 'run_id': 'r'}
        return ColorEncoder(scale=1. if condition == 'base_vit' else 2.), info
    monkeypatch.setattr(extract_features, 'build_encoder', encoder)
    root = tmp_path / 'out'

    def invoke(module, *arguments):
        monkeypatch.setattr('sys.argv', ['cli', *map(str, arguments)])
        module.main()
    return root, invoke


def test_pipeline_gates_hashes_and_reuse(cli, capsys):
    root, invoke = cli
    common = ('--output-root', root, '--disable-comet')
    invoke(build_manifest, 'build', '/anet', '--manifest-id', 'm', *common)
    metadata = json.loads((root / 'manifest' / 'm' / 'metadata.json').read_text())
    assert metadata['gate']['status'] == 'PENDING_AUDIT' and metadata['production']
    assert metadata['splits']['training']['skip_counts'] == {'A': 1}  # [2.0, 2.5] contains no whole chunk.
    with pytest.raises(FileExistsError):
        invoke(build_manifest, 'build', '/anet', '--manifest-id', 'm', *common)
    with pytest.raises(RuntimeError, match='Gate has not passed'):
        invoke(extract_features, '/anet', '--manifest-id', 'm', '--condition', 'base_vit', '--device', 'cpu', *common)
    invoke(build_manifest, 'audit', '/anet', '--manifest-id', 'm', *common)
    metadata = json.loads((root / 'manifest' / 'm' / 'metadata.json').read_text())
    assert metadata['gate']['status'] == 'PASS' and metadata['gate']['reproducibility']['matches']
    assert metadata['comet']['status'] == 'disabled' and metadata['comet']['retry_needed']
    with pytest.raises(RuntimeError, match='Dataset root differs'):
        invoke(extract_features, '/different-anet', '--manifest-id', 'm', '--condition', 'base_vit',
               '--device', 'cpu', *common)

    for condition, extra in (('base_vit', ()), ('moco_query_lora_final', ('--snapshot', root / 'snap'))):
        invoke(extract_features, '/anet', '--manifest-id', 'm', '--condition', condition, '--device', 'cpu',
               '--feature-id', condition, *extra, *common)
    invoke(extract_features, '/anet', '--manifest-id', 'm', '--condition', 'base_vit', '--device', 'cpu',
           '--feature-id', 'base_vit', *common)
    assert 'Reusing verified feature artifact' in capsys.readouterr().out
    base = root / 'features' / 'base_vit' / 'base_vit'
    lora = root / 'features' / 'moco_query_lora_final' / 'moco_query_lora_final'
    for directory in (base, lora):
        stored = json.loads((directory / 'metadata.json').read_text())
        assert stored['manifest']['segment_manifest_sha256'] == metadata['segment_manifest_sha256']
        value = torch.load(directory / 'training' / 'features.pt', weights_only=True)
        assert value['segment_ids'] == ['training:a:0', 'training:a:1', 'training:b:0']

    probe_args = ('--manifest-id', 'm', '--base-features', base, '--lora-features', lora, '--smoke-epochs', '1')
    invoke(run_probe, *probe_args, *common)
    aggregate = json.loads((root / 'results' / 'comparison' / 'aggregate_summary.json').read_text())
    aggregate_sidecar = json.loads((root / 'results' / 'comparison' / 'metadata.json').read_text())
    assert not aggregate['production'] and set(aggregate['conditions']) == {'base_vit', 'moco_query_lora_final'}
    assert 'comet' not in aggregate
    assert aggregate_sidecar['comet']['status'] == 'disabled' and aggregate_sidecar['comet']['retry_needed']
    assert aggregate_sidecar['files'] == {
        name: sha256_file(root / 'results' / 'comparison' / name)
        for name in ('aggregate_summary.json', 'aggregate_summary.csv')
    }
    for condition in aggregate['conditions'].values():
        assert condition['seeds'] == [0, 1, 2] and condition['top1_std'] is not None
    for condition in ('base_vit', 'moco_query_lora_final'):
        for seed in (0, 1, 2):
            summary = json.loads((root / 'results' / condition / f'seed-{seed}' / 'summary.json').read_text())
            assert summary['hyperparameters']['epochs'] == 1 and summary['validation_used_for_selection'] is False
            assert summary['sample_counts'] == {'training': 3, 'validation': 2}
    invoke(run_probe, *probe_args, *common)
    assert 'Reusing identical aggregate result' in capsys.readouterr().out
    (root / 'results' / 'comparison' / 'aggregate_summary.csv').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='Aggregate CSV differs'):
        invoke(run_probe, *probe_args, *common)
    with pytest.raises(RuntimeError, match='Different probe result'):
        invoke(run_probe, *probe_args[:-1], '2', *common)
