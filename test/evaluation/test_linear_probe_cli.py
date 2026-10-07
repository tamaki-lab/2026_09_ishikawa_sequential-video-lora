"""End-to-end CLI plumbing on synthetic data: manifest -> audit -> features -> probe."""

import json
from pathlib import Path

import pytest
import sequential_loader as sl
import torch
from hydra import compose, initialize_config_dir

from anet_synthetic import ColorEncoder, RecordingProcessor, chunk_times
from integration.activitynet_source_selection import SourceSelectionConfig
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

    def encoder(
        condition, snapshot, manifest_metadata, encoder_config, production, expected_source_count,
        expected_protocol_version=None,
    ):
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
            'source_selection.splits.training.count=2', 'source_selection.splits.validation.count=1',
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


def test_reduced_selection_auto_separates_manifest_features_and_results(cli):
    root, invoke = cli
    def materialize(profile_id, seed):
        selection = (
            f'source_selection.id={profile_id}',
            'source_selection.strategy=sha256_rank_preserve_adapter_order_v1',
            f'source_selection.selection_seed={seed}',
            'source_selection.splits.training.count=1',
            'source_selection.splits.validation.count=1',
        )
        before = set((root / 'manifest').iterdir()) if (root / 'manifest').exists() else set()
        invoke(build_manifest, 'runtime.command=build', 'runtime.dataset_root=/anet', *selection)
        created = set((root / 'manifest').iterdir()) - before
        assert len(created) == 1
        directory = created.pop()
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert directory.name == f'lp-v1__sel-{metadata["selection_sha256"][:12]}'
        assert metadata['source_selection']['profile_id'] == profile_id
        assert metadata['splits']['training']['ordered_used_source_ids'] == metadata[
            'source_selection'
        ]['splits']['training']['ordered_selected_ids']
        invoke(build_manifest, 'runtime.command=audit', 'runtime.dataset_root=/anet', *selection)
        assert json.loads((directory / 'metadata.json').read_text())['gate']['status'] == 'SMOKE_PASS'

        feature_metadata = {}
        for condition, snapshot in (
            ('base_vit', ()),
            ('moco_query_lora_final', (f'runtime.snapshot={root / "snap"}',)),
        ):
            feature_metadata[condition] = invoke(
                extract_features, 'runtime.dataset_root=/anet', f'runtime.manifest_id={directory.name}',
                f'runtime.condition={condition}', 'runtime.device=cpu', *selection, *snapshot,
            )
        base = feature_metadata['base_vit']
        lora = feature_metadata['moco_query_lora_final']
        assert base['selection_sha256'] == lora['selection_sha256'] == metadata['selection_sha256']
        assert base['shared_feature_contract'] == lora['shared_feature_contract']
        result = invoke(
            run_probe, f'runtime.manifest_id={directory.name}',
            f'runtime.base_features={root / "features/base_vit" / base["feature_id"]}',
            f'runtime.lora_features={root / "features/moco_query_lora_final" / lora["feature_id"]}',
            'linear_probe.probe.epochs=1', *selection,
        )
        return metadata, feature_metadata, result

    first = materialize('synthetic-reduced-v1', 3)
    second = materialize('synthetic-reduced-v2', 4)
    assert first[0]['selection_sha256'] != second[0]['selection_sha256']
    for condition in ('base_vit', 'moco_query_lora_final'):
        assert first[1][condition]['feature_id'] != second[1][condition]['feature_id']
    assert first[2]['result_id'] != second[2]['result_id']


def test_snapshot_manifest_resolution_recomputes_and_validates_exact_selection(cli):
    root, invoke = cli
    selection_overrides = (
        'source_selection.id=synthetic-reduced-v1',
        'source_selection.strategy=sha256_rank_preserve_adapter_order_v1',
        'source_selection.selection_seed=3',
        'source_selection.splits.training.count=1',
        'source_selection.splits.validation.count=1',
    )
    invoke(
        build_manifest, 'runtime.command=build', 'runtime.dataset_root=/anet',
        *selection_overrides,
    )
    directory = next((root / 'manifest').iterdir())
    manifest_metadata = json.loads((directory / 'metadata.json').read_text())
    activitynet = {
        'name': 'ActivityNet', 'version': '1.3',
        'expected_source_counts': {'training': 2, 'validation': 1},
    }
    source_config = SourceSelectionConfig.from_mapping({
        'id': 'synthetic-reduced-v1',
        'strategy': 'sha256_rank_preserve_adapter_order_v1',
        'selection_seed': 3,
        'splits': {'training': {'count': 1}, 'validation': {'count': 1}},
    }, activitynet['expected_source_counts'])
    snapshot = {
        'final': True,
        'protocol_version': 'activitynet-selected-single-pass-streaming-moco/v1',
        'source_selection': manifest_metadata['source_selection'],
        'selection_sha256': manifest_metadata['selection_sha256'],
        'selection_record': manifest_metadata['selection_record'],
    }
    assert build_manifest.resolve_snapshot_manifest_id(
        '/anet', snapshot, activitynet, source_config, PROVENANCE, 'lp-v1',
    ) == directory.name

    with pytest.raises(RuntimeError, match='SHA-256'):
        build_manifest.resolve_snapshot_manifest_id(
            '/anet', {**snapshot, 'selection_sha256': '0' * 64}, activitynet,
            source_config, PROVENANCE, 'lp-v1',
        )
    changed_identity = {
        **snapshot['source_selection'],
        'splits': {
            **snapshot['source_selection']['splits'],
            'training': {
                **snapshot['source_selection']['splits']['training'],
                'ordered_selected_ids': ['different'],
            },
        },
    }
    with pytest.raises(RuntimeError, match='identity'):
        build_manifest.resolve_snapshot_manifest_id(
            '/anet', {**snapshot, 'source_selection': changed_identity}, activitynet,
            source_config, PROVENANCE, 'lp-v1',
        )
    with pytest.raises(RuntimeError, match='requires source-selection metadata'):
        build_manifest.resolve_snapshot_manifest_id(
            '/anet', {'final': True, 'protocol_version': snapshot['protocol_version']},
            activitynet, source_config, PROVENANCE, 'lp-v1',
        )
    with pytest.raises(RuntimeError, match='record is missing'):
        build_manifest.resolve_snapshot_manifest_id(
            '/anet', {key: value for key, value in snapshot.items()
                      if key != 'selection_record'},
            activitynet, source_config, PROVENANCE, 'lp-v1',
        )
    with pytest.raises(RuntimeError, match='protocol'):
        build_manifest.resolve_snapshot_manifest_id(
            '/anet', {**snapshot, 'protocol_version': 'wrong'}, activitynet,
            source_config, PROVENANCE, 'lp-v1',
        )

    full_config = SourceSelectionConfig.from_mapping({
        'id': 'activitynet-full-v1', 'strategy': 'all_sources',
        'selection_seed': None,
        'splits': {'training': {'count': 2}, 'validation': {'count': 1}},
    }, activitynet['expected_source_counts'])
    assert build_manifest.resolve_snapshot_manifest_id(
        '/anet', {
            'final': True,
            'protocol_version': 'activitynet-full-single-pass-streaming-moco/v2',
        }, activitynet, full_config, PROVENANCE, 'lp-v1',
    ) == 'lp-v1'


def test_lora_snapshot_selection_mismatch_is_rejected_even_for_smoke(monkeypatch):
    encoder = object()
    monkeypatch.setattr(extract_features, 'ViTLoRAFrameEncoder', lambda *args, **kwargs: encoder)
    monkeypatch.setattr(extract_features, 'encoder_base_fingerprint', lambda value: 'base')
    monkeypatch.setattr(extract_features, 'load_query_lora_snapshot', lambda *args: {
        'selection_sha256': 'b' * 64, 'source_selection': {'profile_id': 'other'},
    })
    manifest_metadata = {
        'selection_sha256': 'a' * 64,
        'source_selection': {'profile_id': 'expected'},
        'splits': {'training': {'ordered_used_source_sha256': 'sources'}},
    }
    encoder_config = {
        'checkpoint_id': 'base', 'feature_size': 4, 'image_size': 2, 'channels': 3, 'lora': {},
    }
    with pytest.raises(RuntimeError, match='source selections differ'):
        extract_features.build_encoder(
            'moco_query_lora_final', Path('/snapshot'), manifest_metadata, encoder_config,
            False, 1, 'activitynet-selected-single-pass-streaming-moco/v1',
        )



@pytest.fixture
def comet_calls(monkeypatch):
    """Record each new Comet experiment's (name, tags); Comet itself stays disabled."""
    calls = []
    for module in (build_manifest, extract_features, run_probe):
        def record(name, parameters, tags=(), start=module.start_experiment, **kwargs):
            calls.append((name, tags))
            return start(name, parameters, tags=tags, **kwargs)
        monkeypatch.setattr(module, 'start_experiment', record)
    return calls


@pytest.mark.parametrize('scope', ['smoke', 'production'])
def test_pipeline_comet_experiment_labels(cli, comet_calls, monkeypatch, scope):
    root, invoke = cli
    if scope == 'production':
        # Treat the synthetic config as canonical so every stage takes its production path.
        for module, name in ((build_manifest, 'science_contract'), (extract_features, 'feature_science_contract'),
                             (run_probe, 'science_contract'), (run_probe, 'feature_science_contract')):
            contract = getattr(module, name)
            monkeypatch.setattr(module, name, lambda *args, contract=contract: {**contract(*args), 'canonical': True})
        monkeypatch.setattr(run_probe, 'validate_production_snapshot_metadata', lambda *args, **kwargs: None)
        monkeypatch.setattr(build_manifest, 'approved_profile_name', lambda *args: 'synthetic-full-v1')
    manifest_args = ('runtime.dataset_root=/anet', 'runtime.manifest_id=m')
    invoke(build_manifest, 'runtime.command=build', *manifest_args)
    invoke(build_manifest, 'runtime.command=audit', *manifest_args)
    for condition, snapshot in (('base_vit', ()), ('moco_query_lora_final', (f'runtime.snapshot={root / "snap"}',))):
        invoke(extract_features, *manifest_args, f'runtime.condition={condition}', 'runtime.device=cpu',
               f'runtime.feature_id={condition}', *snapshot)
    invoke(run_probe, 'runtime.manifest_id=m', f'runtime.base_features={root / "features/base_vit/base_vit"}',
           f'runtime.lora_features={root / "features/moco_query_lora_final/moco_query_lora_final"}',
           'runtime.result_id=r', 'linear_probe.probe.epochs=1')

    probes = [(f'lp-v1__probe__{condition}__seed-{seed}',
               ('linear-probe', 'probe', condition, f'seed-{seed}', scope, 'lp-v1'))
              for condition in ('base-vit', 'moco-query-lora-final') for seed in (0, 1, 2)]
    assert comet_calls == [
        ('lp-v1__manifest__m', ('linear-probe', 'manifest', scope, 'lp-v1')),
        ('lp-v1__features__base-vit', ('linear-probe', 'features', 'base-vit', scope, 'lp-v1')),
        ('lp-v1__features__moco-query-lora-final',
         ('linear-probe', 'features', 'moco-query-lora-final', scope, 'lp-v1')),
        *probes,
        ('lp-v1__aggregate', ('linear-probe', 'aggregate', scope, 'lp-v1')),
    ]
    aggregate = json.loads((root / 'results' / 'r' / 'comparison' / 'aggregate_summary.json').read_text())
    assert aggregate['production'] is (scope == 'production')
    assert set(aggregate['probe_experiment_keys']) == {name for name, _ in probes}
