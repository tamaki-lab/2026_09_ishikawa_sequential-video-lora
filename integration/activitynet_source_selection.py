"""Deterministic ActivityNet source selection shared by MoCo and evaluation."""

from dataclasses import dataclass

from utils.artifact_io import canonical_json_bytes, sha256_bytes
from utils.configuration import load_config_group


SELECTION_SCHEMA = 'activitynet-source-selection/v1'
SHA256_RANK_STRATEGY = 'sha256_rank_preserve_adapter_order_v1'
ALL_SOURCES_STRATEGY = 'all_sources'
APPROVED_PROFILES = ('activitynet_full_v1', 'activitynet_reduced_v1')


def _uint32(name, value):
    if type(value) is not int or not 0 <= value < 2 ** 32:
        raise ValueError(f'{name} must be an integer in [0, 2**32)')
    return value


@dataclass(frozen=True)
class SourceSelectionConfig:
    id: str
    strategy: str
    selection_seed: int | None
    counts: dict

    @classmethod
    def from_mapping(cls, value, expected_counts):
        splits = value.get('splits') or {}
        if set(splits) != set(expected_counts):
            raise ValueError('source_selection.splits must define training and validation')
        counts = {}
        for split, full_count in expected_counts.items():
            item = splits[split]
            if not isinstance(item, dict) or type(item.get('count')) is not int:
                raise ValueError(f'source_selection.splits.{split}.count must be an integer')
            count = item['count']
            if not 1 <= count <= full_count:
                raise ValueError(
                    f'source_selection.splits.{split}.count must be in [1, {full_count}]'
                )
            counts[split] = count
        result = cls(
            id=str(value.get('id', '')), strategy=str(value.get('strategy', '')),
            selection_seed=value.get('selection_seed'), counts=counts,
        )
        if not result.id:
            raise ValueError('source_selection.id must be non-empty')
        if result.strategy == ALL_SOURCES_STRATEGY:
            if result.selection_seed is not None or result.counts != expected_counts:
                raise ValueError('all_sources requires selection_seed=null and every full split count')
        elif result.strategy == SHA256_RANK_STRATEGY:
            _uint32('source_selection.selection_seed', result.selection_seed)
        else:
            raise ValueError(f'Unknown source-selection strategy: {result.strategy}')
        return result

    def metadata(self):
        return {
            'id': self.id, 'strategy': self.strategy, 'selection_seed': self.selection_seed,
            'splits': {split: {'count': count} for split, count in self.counts.items()},
        }


@dataclass(frozen=True)
class SourceSelection:
    sources: dict
    identity: dict
    selection_sha256: str


def approved_profile_name(value, expected_counts):
    """Return the repository profile name only for an exact versioned preset."""
    config = SourceSelectionConfig.from_mapping(value, expected_counts)
    for name in APPROVED_PROFILES:
        try:
            candidate = SourceSelectionConfig.from_mapping(
                load_config_group('source_selection', name), expected_counts,
            )
        except ValueError:
            continue
        if config == candidate:
            return name
    return None


def _score(strategy, seed, split, sequence_id):
    values = (strategy, str(seed), split, sequence_id)
    if any('\0' in value for value in values):
        raise ValueError('Source-selection identity fields may not contain NUL')
    return sha256_bytes('\0'.join(values).encode('utf-8'))


def _ordered_ids(sources, split, expected_count):
    if len(sources) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} {split} sources, got {len(sources)}')
    ids = [source.sequence_id for source in sources]
    if any(not isinstance(sequence_id, str) or not sequence_id for sequence_id in ids):
        raise ValueError(f'{split} source IDs must be non-empty strings')
    if len(set(ids)) != len(ids):
        raise ValueError(f'{split} source IDs must be unique')
    return ids


def select_activitynet_sources(
    full_sources, config, *, expected_counts, dataset, sequential_loader, annotation_sha256,
):
    """Select a stable set, then restore Adapter order for actual processing."""
    if set(full_sources) != set(expected_counts):
        raise ValueError('Full sources must define the configured ActivityNet splits')
    if not isinstance(annotation_sha256, str) or len(annotation_sha256) != 64:
        raise ValueError('annotation_sha256 must be a full SHA-256 hex digest')
    selected, split_identity = {}, {}
    for split in expected_counts:
        sources = tuple(full_sources[split])
        ids = _ordered_ids(sources, split, expected_counts[split])
        count = config.counts[split]
        if config.strategy == ALL_SOURCES_STRATEGY:
            chosen = set(ids)
        else:
            ranked = sorted(ids, key=lambda sequence_id: (
                _score(config.strategy, config.selection_seed, split, sequence_id), sequence_id,
            ))
            chosen = set(ranked[:count])
        used = tuple(source for source in sources if source.sequence_id in chosen)
        used_ids = [source.sequence_id for source in used]
        if len(used_ids) != count:
            raise RuntimeError(f'{split} deterministic selection did not produce the requested count')
        selected[split] = used
        split_identity[split] = {
            'full_source_count': len(ids),
            'full_ordered_source_sha256': sha256_bytes(canonical_json_bytes(ids)),
            'selected_source_count': len(used_ids),
            'selected_ordered_source_sha256': sha256_bytes(canonical_json_bytes(used_ids)),
            'ordered_selected_ids': used_ids,
        }
    identity = {
        'schema': SELECTION_SCHEMA,
        'profile_id': config.id,
        'strategy': config.strategy,
        'selection_seed': config.selection_seed,
        'dataset': {'name': dataset['name'], 'version': dataset['version']},
        'splits': split_identity,
        'sequential_loader': {
            'branch': sequential_loader['branch'], 'commit': sequential_loader['commit'],
        },
        'annotation_sha256': annotation_sha256,
    }
    selection_sha256 = sha256_bytes(canonical_json_bytes(identity))
    return SourceSelection(selected, identity, selection_sha256)


def validate_selection_metadata(metadata, expected):
    if metadata.get('selection_sha256') != expected.selection_sha256:
        raise RuntimeError('Source selection SHA-256 differs from the requested selection')
    if metadata.get('source_selection') != expected.identity:
        raise RuntimeError('Source selection identity differs from the requested selection')


def validate_selection_record(metadata):
    """Validate non-hashed operational provenance stored beside stable identity."""
    selection = metadata.get('source_selection')
    record = metadata.get('selection_record')
    if selection is None:
        if record is not None:
            raise RuntimeError('Source selection record exists without selection identity')
        return
    if not isinstance(selection, dict):
        raise RuntimeError('Source selection identity is invalid')
    if not isinstance(record, dict):
        raise RuntimeError('Source selection record is missing')
    implementation = record.get('implementation')
    required_implementation = (
        'repository', 'branch', 'commit', 'dirty', 'tracked_diff_sha256',
    )
    if (
        record.get('schema') != selection.get('schema')
        or not isinstance(record.get('created'), str) or not record['created']
        or not isinstance(implementation, dict)
        or any(key not in implementation for key in required_implementation)
        or any(not isinstance(implementation[key], str) or not implementation[key]
               for key in required_implementation if key != 'dirty')
        or type(implementation.get('dirty')) is not bool
    ):
        raise RuntimeError('Source selection record is invalid')
