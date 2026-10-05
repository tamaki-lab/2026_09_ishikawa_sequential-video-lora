"""ActivityNet annotation-segment manifest, canonical labels and Integrity Gate.

One annotation instance is one sample. A chunk belongs to a segment only if it
has at least one valid frame, all valid timestamps are finite, and all of them
lie in the closed interval [segment_start, segment_end]. Raw Loader timestamps
are used as-is; padding never participates. Segments with no such chunk are
skipped and counted. Nothing here relaxes the rule, class set or split.
"""

from collections import Counter
import json
from pathlib import Path

import torch

from integration.sequential_stream import ordered_samples
from utils.artifact_io import canonical_json_bytes, read_json, sha256_file, write_bytes_atomic
from utils.artifact_io import write_json_atomic


MANIFEST_SCHEMA = 'activitynet-linear-probe-manifest/v1'
MANIFEST_FILE = 'activitynet_linear_probe_manifest.jsonl'
MAPPING_FILE = 'label_mapping.json'
SPLITS = ('training', 'validation')
SPLIT_COUNTS = {'training': 10_024, 'validation': 4_926}
CLASS_COUNT = 200
CHUNK_RULE = 'all valid timestamps finite and within closed [segment_start, segment_end]; >=1 valid frame'


class AnnotationIndex:
    """Resolve Adapter `evaluation_reference`s to v1.3 annotation entries."""

    def __init__(self):
        self._databases = {}

    def database(self, path):
        path = Path(path)
        if path not in self._databases:
            with path.open(encoding='utf-8') as file:
                annotation = json.load(file)
            if annotation.get('version') != 'VERSION 1.3':
                raise ValueError('ActivityNet annotation version must be VERSION 1.3')
            self._databases[path] = annotation['database']
        return self._databases[path]

    def annotations(self, reference):
        entry = self.database(reference.annotation_path).get(reference.video_id)
        if entry is None or entry.get('subset') != reference.subset:
            raise ValueError(f'Annotation entry missing or split differs: {reference.video_id}')
        result = []
        for index, annotation in enumerate(entry.get('annotations', [])):
            label = annotation.get('label')
            segment = annotation.get('segment')
            if not isinstance(label, str) or not label or not isinstance(segment, list) or len(segment) != 2:
                raise ValueError(f'Malformed annotation {index} for {reference.video_id}')
            start, end = (float(value) for value in segment)
            if not (torch.isfinite(torch.tensor([start, end])).all() and start <= end):
                raise ValueError(f'Invalid segment {segment} for {reference.video_id}:{index}')
            result.append({'annotation_index': index, 'label': label, 'segment_start': start, 'segment_end': end})
        return result


def canonical_label_mapping(index, sources_by_split):
    """Sort the shared 200-label set once and assign 0..199."""
    label_sets = {
        split: {item['label'] for source in sources for item in index.annotations(source.evaluation_reference)}
        for split, sources in sources_by_split.items()
    }
    if any(len(labels) != CLASS_COUNT for labels in label_sets.values()) or len(
        {frozenset(labels) for labels in label_sets.values()}
    ) != 1:
        counts = {split: len(labels) for split, labels in label_sets.items()}
        raise RuntimeError(f'Expected the same {CLASS_COUNT} labels in every split: {counts}')
    return {label: label_id for label_id, label in enumerate(sorted(label_sets[SPLITS[0]]))}


def chunk_in_segment(sample, start, end):
    return contained(sample.timestamps[sample.valid_mask], start, end)


def video_chunk_timestamps(source):
    """Valid raw timestamps of every chunk, in chronological chunk order."""
    chunks = []
    with ordered_samples((source,), stream_mode='strict_single') as samples:
        for sample in samples:
            if sample.sequence_index != len(chunks):
                raise RuntimeError('Chunk order changed while scanning timestamps')
            chunks.append(sample.timestamps[sample.valid_mask].clone())
    if not chunks:
        raise RuntimeError(f'Source produced no chunks: {source.sequence_id}')
    return chunks


def contained(timestamps, start, end):
    return bool(len(timestamps) and torch.isfinite(timestamps).all().item()
                and ((timestamps >= start) & (timestamps <= end)).all().item())


def video_rows(split, source, annotations, mapping, chunks):
    rows, skipped = [], []
    for item in annotations:
        indices = [index for index, timestamps in enumerate(chunks)
                   if contained(timestamps, item['segment_start'], item['segment_end'])]
        if not indices:
            skipped.append(item['label'])
            continue
        rows.append({
            'segment_id': f'{split}:{source.sequence_id}:{item["annotation_index"]}', 'split': split,
            'video_id': source.sequence_id, 'annotation_index': item['annotation_index'],
            'label': item['label'], 'label_id': mapping[item['label']],
            'segment_start': item['segment_start'], 'segment_end': item['segment_end'],
            'chunk_indices': indices, 'chunk_count': len(indices),
        })
    return rows, skipped


def build_manifest(index, sources_by_split, mapping, progress=None):
    """Rows in split -> Adapter source order -> annotation_index order."""
    rows, skips = [], {split: Counter() for split in sources_by_split}
    for split in SPLITS:
        for position, source in enumerate(sources_by_split[split]):
            annotations = index.annotations(source.evaluation_reference)
            video, skipped = video_rows(split, source, annotations, mapping, video_chunk_timestamps(source))
            rows.extend(video)
            skips[split].update(skipped)
            if progress:
                progress(split, position + 1, len(sources_by_split[split]))
    return rows, {split: dict(sorted(counter.items())) for split, counter in skips.items()}


def manifest_bytes(rows):
    return b''.join(canonical_json_bytes(row) + b'\n' for row in rows)


def integrity_gate(rows, mapping, mapping_sha256, metadata_mapping_sha256):
    """Checks 1-8 of the Dataset Integrity Gate. Check 9 is the audit rebuild."""
    labels = {split: Counter(row['label_id'] for row in rows if row['split'] == split) for split in SPLITS}
    ids = [row['segment_id'] for row in rows]
    split_ids = {split: {row['segment_id'] for row in rows if row['split'] == split} for split in SPLITS}
    every_class = set(mapping.values())
    checks = {
        '1_training_labels_200': len(labels['training']) == CLASS_COUNT,
        '2_validation_labels_200': len(labels['validation']) == CLASS_COUNT,
        '3_training_sample_per_class': set(labels['training']) == every_class,
        '4_validation_sample_per_class': set(labels['validation']) == every_class,
        '5_no_duplicate_segment_id': len(ids) == len(set(ids)),
        '6_no_split_overlap': not (split_ids['training'] & split_ids['validation']),
        '7_chunk_count_consistent': all(row['chunk_count'] == len(row['chunk_indices']) >= 1 for row in rows),
        '8_label_mapping_hash_matches': mapping_sha256 == metadata_mapping_sha256 and all(
            mapping[row['label']] == row['label_id'] for row in rows),
    }
    return {
        'checks': checks,
        'class_counts': {split: len(counter) for split, counter in labels.items()},
        'missing_classes': {split: sorted(every_class - set(counter)) for split, counter in labels.items()},
        'rows': {split: sum(counter.values()) for split, counter in labels.items()},
    }


def gate_status(gate, production, reproducible):
    checks = dict(gate['checks'], **{'9_regeneration_sha256_matches': reproducible})
    coverage = ('1_', '2_', '3_', '4_')
    failed = [name for name, passed in checks.items() if passed is False
              and (production or not name.startswith(coverage))]
    if failed:
        return 'FAIL', failed
    if reproducible is None:
        return 'PENDING_AUDIT', []
    return ('PASS' if production else 'SMOKE_PASS'), []


def write_manifest(directory, rows, mapping, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(directory / MAPPING_FILE, canonical_json_bytes(mapping))
    write_bytes_atomic(directory / MANIFEST_FILE, manifest_bytes(rows))
    write_json_atomic(directory / 'metadata.json', metadata)


def load_manifest(directory, require_gate=True):
    """Load and hash-verify a manifest; optionally require a passed Gate."""
    directory = Path(directory)
    metadata = read_json(directory / 'metadata.json')
    if metadata.get('schema') != MANIFEST_SCHEMA:
        raise RuntimeError(f'Unsupported manifest schema: {metadata.get("schema")}')
    if sha256_file(directory / MANIFEST_FILE) != metadata['segment_manifest_sha256'] or sha256_file(
        directory / MAPPING_FILE
    ) != metadata['label_mapping_sha256']:
        raise RuntimeError('Manifest or label mapping file differs from metadata hashes')
    if require_gate and metadata['gate']['status'] not in ('PASS', 'SMOKE_PASS'):
        raise RuntimeError(f'Manifest Integrity Gate has not passed: {metadata["gate"]["status"]}')
    if require_gate and metadata['production'] != (metadata['gate']['status'] == 'PASS'):
        raise RuntimeError('Manifest production flag and Gate status disagree')
    mapping = read_json(directory / MAPPING_FILE)
    rows = [json.loads(line) for line in (directory / MANIFEST_FILE).read_text(encoding='utf-8').splitlines()]
    return rows, mapping, metadata
