"""Raw segment features: same chunks / order for both conditions, no post-processing."""

import pytest
import torch

from anet_synthetic import ColorEncoder, RecordingProcessor, chunk_times
from evaluation import activitynet_manifest as manifest
from evaluation import segment_features as features
from integration.sequential_vit import encode_chunk
from model.aggregators import MaskedMeanClipAggregator


def entry(subset, *annotations):
    return {'subset': subset, 'annotations': [{'label': label, 'segment': [start, end]}
                                              for label, start, end in annotations]}


@pytest.fixture
def setup(dataset, monkeypatch):
    monkeypatch.setattr(manifest, 'CLASS_COUNT', 2)
    sources = dataset({
        'a': entry('training', ('A', 0., 3.1), ('B', 1.6, 3.4)),
        'b': entry('training', ('B', 0., 1.5)),
        'c': entry('validation', ('A', 0., 1.5), ('B', 1.6, 9.)),
    }, {'a': chunk_times(4), 'b': chunk_times(2), 'c': chunk_times(3)})
    index = manifest.AnnotationIndex()
    mapping = manifest.canonical_label_mapping(index, sources)
    rows, _ = manifest.build_manifest(index, sources, mapping)
    return sources, rows


def extract(sources, rows, encoder):
    return {split: features.extract_split([row for row in rows if row['split'] == split], sources[split],
                                          RecordingProcessor(), encoder, torch.device('cpu'))
            for split in manifest.SPLITS}


def test_segment_feature_is_plain_mean_of_raw_chunk_masked_means(setup, videos):
    sources, rows = setup
    encoder = ColorEncoder()
    videos['reads'].clear()
    splits = extract(sources, rows, encoder)
    # Video a needs chunks 0..1 only: the stream closes after chunk 1 without reading chunks 2..3.
    assert [read for read in videos['reads'] if read[0] == 'a'] == [('a', 0), ('a', 1)]
    assert videos['opened'] == videos['closed']
    assert splits['training']['segment_ids'] == ['training:a:0', 'training:a:1', 'training:b:0']
    assert splits['training']['labels'].tolist() == [0, 1, 1]
    # Recompute the definition independently for training:a:0 (chunks 0 and 1).
    from integration.sequential_stream import ordered_samples
    expected = []
    with ordered_samples((sources['training'][0],), stream_mode='strict_single') as samples:
        for sample in samples:
            if sample.sequence_index in (0, 1):
                with torch.no_grad():
                    frames = encode_chunk(sample, RecordingProcessor(), encoder, torch.device('cpu'))[2]
                expected.append(MaskedMeanClipAggregator()(frames, sample.valid_mask))
    torch.testing.assert_close(splits['training']['features'][0], torch.stack(expected).mean(0), rtol=0, atol=0)
    value = splits['training']['features']
    assert value.dtype == torch.float32 and not value.requires_grad and value.device.type == 'cpu'
    assert not torch.allclose(value.norm(dim=1), torch.ones(3))  # No L2 normalization.
    assert not torch.allclose(value.mean(dim=0), torch.zeros(768), atol=1e-3)  # No centering.


def test_base_and_lora_conditions_share_segment_ids_and_labels(setup):
    sources, rows = setup
    base = extract(sources, rows, ColorEncoder())
    lora = extract(sources, rows, ColorEncoder(scale=2.))
    for split in manifest.SPLITS:
        assert base[split]['segment_ids'] == lora[split]['segment_ids']
        assert torch.equal(base[split]['labels'], lora[split]['labels'])
        assert not torch.equal(base[split]['features'], lora[split]['features'])


def test_artifact_round_trip_and_rejections(setup, tmp_path):
    sources, rows = setup
    splits = extract(sources, rows, ColorEncoder())
    manifest_metadata = {'segment_manifest_sha256': 'm', 'label_mapping_sha256': 'l'}
    contract = {'manifest': dict(manifest_metadata), 'common': 'same'}
    metadata = features.write_feature_artifact(tmp_path / 'f', splits, rows, {
        'condition': 'base_vit', 'manifest': dict(manifest_metadata),
        'feature_definition': {'size': features.FEATURE_SIZE},
        'shared_feature_contract': contract,
        'shared_feature_contract_sha256': features.shared_contract_sha256(contract)})
    assert set(metadata['files']) == {'training/features.pt', 'validation/features.pt'}
    loaded, _ = features.load_features(tmp_path / 'f', rows, manifest_metadata, 'base_vit')
    assert torch.equal(loaded['training']['features'], splits['training']['features'])
    with pytest.raises(RuntimeError, match='Expected moco'):
        features.load_features(tmp_path / 'f', rows, manifest_metadata, 'moco_query_lora_final')
    with pytest.raises(RuntimeError, match='different manifest'):
        features.load_features(tmp_path / 'f', rows, {**manifest_metadata, 'segment_manifest_sha256': 'x'})
    reordered = {**splits['training'], 'segment_ids': splits['training']['segment_ids'][::-1]}
    with pytest.raises(RuntimeError, match='order or count'):
        features.verify_split(reordered, rows, 'training')
    with pytest.raises(RuntimeError, match='float32'):
        features.verify_split({**splits['training'], 'features': splits['training']['features'].double()},
                              rows, 'training')
    (tmp_path / 'f' / 'training' / 'features.pt').write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='hashes'):
        features.load_features(tmp_path / 'f', rows, manifest_metadata)


def test_shared_feature_contract_pair_validation():
    contract = {'manifest': {'segment_manifest_sha256': 'm', 'label_mapping_sha256': 'l'},
                'preprocessing': {'processor': 'same'}}
    metadata = {'shared_feature_contract': contract,
                'shared_feature_contract_sha256': features.shared_contract_sha256(contract)}
    assert features.validate_feature_pair(metadata, dict(metadata)) == contract
    changed = {**contract, 'preprocessing': {'processor': 'different'}}
    with pytest.raises(RuntimeError, match='contracts differ'):
        features.validate_feature_pair(metadata, {
            'shared_feature_contract': changed,
            'shared_feature_contract_sha256': features.shared_contract_sha256(changed),
        })
    with pytest.raises(RuntimeError, match='contract hash'):
        features.validate_shared_contract({**metadata, 'shared_feature_contract_sha256': 'tampered'})


def test_changed_timestamps_are_rejected_at_extraction(setup, videos):
    sources, rows = setup
    videos['timestamps']['a'] = chunk_times(4, offset=0.05)
    with pytest.raises(RuntimeError, match='no longer lies'):
        extract(sources, rows, ColorEncoder())


def test_only_the_implemented_feature_definition_is_accepted():
    from utils.configuration import load_config_group
    declared = load_config_group('linear_probe', 'lp_v1')['feature_definition']
    assert features.validate_feature_definition(declared) == features.FEATURE_DEFINITION
    for key, value in (('normalization', 'l2'), ('chunk_aggregation', 'max over valid frames')):
        with pytest.raises(ValueError, match='implemented feature definition'):
            features.validate_feature_definition({**declared, key: value})
