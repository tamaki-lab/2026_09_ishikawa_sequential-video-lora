"""Segment manifest: strict inclusion, ordering, label mapping and Integrity Gate."""

import pytest
import torch

from anet_synthetic import chunk_times
from evaluation import activitynet_manifest as manifest
from utils.artifact_io import canonical_json_bytes, sha256_bytes


def entry(subset, *annotations):
    return {'subset': subset, 'annotations': [{'label': label, 'segment': [start, end]}
                                              for label, start, end in annotations]}


@pytest.fixture
def two_classes(monkeypatch):
    monkeypatch.setattr(manifest, 'CLASS_COUNT', 2)


def test_closed_interval_containment_padding_and_partial_overlap(dataset, videos):
    # Chunks: 0 -> [0.0, 1.5], 1 -> [1.6, 3.1], 2 (3 frames + padding) -> [3.2, 3.4].
    sources = dataset({'v': entry('training', ('A', 1.6, 3.1), ('A', 1.6, 3.09), ('B', 0.0, 3.4),
                                  ('B', 3.2, 3.4), ('A', 3.25, 9.0))}, {'v': chunk_times(3)})
    mapping = {'A': 0, 'B': 1}
    rows, skipped = manifest.video_rows('training', sources['training'][0],
                                        manifest.AnnotationIndex().annotations(sources['training'][0]
                                                                               .evaluation_reference),
                                        mapping, manifest.video_chunk_timestamps(sources['training'][0]))
    assert [(row['annotation_index'], row['chunk_indices']) for row in rows] == [(0, [1]), (2, [0, 1, 2]), (3, [2])]
    assert skipped == ['A', 'A']  # Partial overlap of chunk 1, and of the terminal chunk.
    assert rows[0] == {
        'segment_id': 'training:v:0', 'split': 'training', 'video_id': 'v', 'annotation_index': 0, 'label': 'A',
        'label_id': 0, 'segment_start': 1.6, 'segment_end': 3.1, 'chunk_indices': [1], 'chunk_count': 1}
    assert videos['opened'] == videos['closed'] == ['v']


def test_contained_requires_valid_finite_timestamps():
    assert not manifest.contained(torch.tensor([], dtype=torch.float64), 0, 1)
    assert not manifest.contained(torch.tensor([0.5, float('nan')], dtype=torch.float64), 0, 1)
    assert manifest.contained(torch.tensor([0., 1.], dtype=torch.float64), 0, 1)


def test_duplicate_annotations_are_independent_and_order_is_split_source_index(dataset, two_classes):
    sources = dataset({
        'b': entry('training', ('B', 0., 1.5), ('B', 0., 1.5)),
        'a': entry('training', ('A', 0., 9.), ('B', 0., 1.5)),
        'c': entry('validation', ('A', 0., 1.5), ('B', 1.6, 3.1)),
    }, {'a': chunk_times(2), 'b': chunk_times(2), 'c': chunk_times(3)})
    index = manifest.AnnotationIndex()
    mapping = manifest.canonical_label_mapping(index, sources)
    assert mapping == {'A': 0, 'B': 1}
    rows, skips = manifest.build_manifest(index, sources, mapping)
    assert [row['segment_id'] for row in rows] == [
        'training:a:0', 'training:a:1', 'training:b:0', 'training:b:1', 'validation:c:0', 'validation:c:1']
    assert skips == {'training': {}, 'validation': {}}
    again, _ = manifest.build_manifest(manifest.AnnotationIndex(), sources, mapping)
    assert manifest.manifest_bytes(again) == manifest.manifest_bytes(rows)
    mapping_sha = sha256_bytes(canonical_json_bytes(mapping))
    gate = manifest.integrity_gate(rows, mapping, mapping_sha, mapping_sha)
    assert all(gate['checks'].values())
    assert manifest.gate_status(gate, True, None) == ('PENDING_AUDIT', [])
    assert manifest.gate_status(gate, True, True) == ('PASS', [])
    assert manifest.gate_status(gate, True, False) == ('FAIL', ['9_regeneration_sha256_matches'])


def test_label_mapping_requires_same_label_set(dataset, two_classes):
    sources = dataset({'a': entry('training', ('A', 0., 1.), ('B', 0., 1.)),
                       'c': entry('validation', ('A', 0., 1.), ('C', 0., 1.))},
                      {'a': chunk_times(1), 'c': chunk_times(1)})
    with pytest.raises(RuntimeError, match='same 2 labels'):
        manifest.canonical_label_mapping(manifest.AnnotationIndex(), sources)


def row(split, video, index, label_id, chunks=(0,)):
    return {'segment_id': f'{split}:{video}:{index}', 'split': split, 'video_id': video, 'annotation_index': index,
            'label': 'AB'[label_id], 'label_id': label_id, 'segment_start': 0., 'segment_end': 1.,
            'chunk_indices': list(chunks), 'chunk_count': len(chunks)}


@pytest.mark.parametrize('case,failed', [
    ('missing_class', ['3_training_sample_per_class', '1_training_labels_2']),
    ('duplicate', ['5_no_duplicate_segment_id']),
    ('chunk_count', ['7_chunk_count_consistent']),
    ('mapping', ['8_label_mapping_hash_matches']),
])
def test_gate_failures(two_classes, case, failed):
    mapping = {'A': 0, 'B': 1}
    rows = [row('training', 'a', 0, 0), row('training', 'a', 1, 1), row('validation', 'c', 0, 0),
            row('validation', 'c', 1, 1)]
    sha = metadata_sha = 'x'
    if case == 'missing_class':
        rows[1] = row('training', 'a', 1, 0)
    elif case == 'duplicate':
        rows.append(dict(rows[0]))
    elif case == 'chunk_count':
        rows[0]['chunk_count'] = 2
    else:
        metadata_sha = 'y'
    gate = manifest.integrity_gate(rows, mapping, sha, metadata_sha)
    assert sorted(name for name, passed in gate['checks'].items() if not passed) == sorted(failed)
    status, names = manifest.gate_status(gate, True, True)
    assert status == 'FAIL' and sorted(names) == sorted(failed)
    smoke = manifest.gate_status(gate, False, True)
    assert smoke[0] == ('SMOKE_PASS' if case == 'missing_class' else 'FAIL')


def test_split_overlap_fails(two_classes):
    rows = [row('training', 'a', 0, 0), row('training', 'a', 1, 1), row('validation', 'c', 0, 0),
            row('validation', 'c', 1, 1)]
    rows[2] = {**rows[2], 'segment_id': 'training:a:0'}
    assert not manifest.integrity_gate(rows, {'A': 0, 'B': 1}, 'x', 'x')['checks']['6_no_split_overlap']


def test_manifest_files_round_trip_and_reject_tampering(tmp_path):
    rows = [row('training', 'a', 0, 0)]
    mapping = {'A': 0, 'B': 1}
    metadata = {'schema': manifest.MANIFEST_SCHEMA, 'production': True,
                'segment_manifest_sha256': sha256_bytes(manifest.manifest_bytes(rows)),
                'label_mapping_sha256': sha256_bytes(canonical_json_bytes(mapping)),
                'gate': {'status': 'PENDING_AUDIT'}}
    manifest.write_manifest(tmp_path, rows, mapping, metadata)
    assert manifest.load_manifest(tmp_path, require_gate=False)[:2] == (rows, mapping)
    with pytest.raises(RuntimeError, match='Gate has not passed'):
        manifest.load_manifest(tmp_path)
    (tmp_path / manifest.MANIFEST_FILE).write_bytes(b'{}\n')
    with pytest.raises(RuntimeError, match='hashes'):
        manifest.load_manifest(tmp_path, require_gate=False)
