"""Deterministic, versioned ActivityNet source-selection identity."""

from types import SimpleNamespace

import pytest

from integration.activitynet_source_selection import (
    ALL_SOURCES_STRATEGY, SHA256_RANK_STRATEGY, SourceSelectionConfig,
    approved_profile_name, select_activitynet_sources, validate_selection_metadata,
    validate_selection_record,
)
from utils.configuration import load_config_group


EXPECTED = {'training': 8, 'validation': 5}
DATASET = {'name': 'ActivityNet', 'version': '1.3'}
LOADER = {'branch': 'ActivityNet', 'commit': 'a' * 40}
ANNOTATION_SHA = 'b' * 64


def sources():
    return {
        split: tuple(SimpleNamespace(sequence_id=f'{split}-{index}') for index in range(count))
        for split, count in EXPECTED.items()
    }


def reduced(seed=7, training=4, validation=3):
    return SourceSelectionConfig.from_mapping({
        'id': 'test-reduced-v1', 'strategy': SHA256_RANK_STRATEGY, 'selection_seed': seed,
        'splits': {'training': {'count': training}, 'validation': {'count': validation}},
    }, EXPECTED)


def select(config, full=None):
    return select_activitynet_sources(
        sources() if full is None else full, config, expected_counts=EXPECTED,
        dataset=DATASET, sequential_loader=LOADER, annotation_sha256=ANNOTATION_SHA,
    )


def test_same_inputs_reproduce_ids_hash_and_adapter_order():
    first = select(reduced())
    second = select(reduced(), sources())
    assert first.identity == second.identity
    assert first.selection_sha256 == second.selection_sha256
    for split in EXPECTED:
        selected_ids = [source.sequence_id for source in first.sources[split]]
        adapter_positions = {source.sequence_id: index for index, source in enumerate(sources()[split])}
        assert [adapter_positions[value] for value in selected_ids] == sorted(
            adapter_positions[value] for value in selected_ids
        )
        assert selected_ids == first.identity['splits'][split]['ordered_selected_ids']


def test_sha256_ranking_v1_has_a_fixed_known_result():
    result = select(reduced())
    assert {
        split: [source.sequence_id for source in selected]
        for split, selected in result.sources.items()
    } == {
        'training': ['training-1', 'training-2', 'training-4', 'training-5'],
        'validation': ['validation-0', 'validation-2', 'validation-4'],
    }
    assert result.selection_sha256 == '520e1e4dbe96da9d3728304bafafc68b4485e77061a4a49c1e6e89c64d459a62'


@pytest.mark.parametrize('config', [
    reduced(seed=8), reduced(training=5),
    SourceSelectionConfig.from_mapping({
        'id': 'test-full-v1', 'strategy': ALL_SOURCES_STRATEGY, 'selection_seed': None,
        'splits': {'training': {'count': 8}, 'validation': {'count': 5}},
    }, EXPECTED),
])
def test_seed_count_or_strategy_changes_selection_identity(config):
    assert select(config).selection_sha256 != select(reduced()).selection_sha256


def test_model_seed_is_not_an_input_to_source_selection():
    selected = [select(reduced()).selection_sha256 for _runtime_seed in (0, 1, 2 ** 32 - 1)]
    assert len(set(selected)) == 1


def test_duplicate_count_and_metadata_mismatch_are_rejected():
    full = sources()
    full['training'] = (*full['training'][:-1], full['training'][0])
    with pytest.raises(ValueError, match='unique'):
        select(reduced(), full)
    with pytest.raises(ValueError, match='Expected exactly'):
        select(reduced(), {**sources(), 'training': sources()['training'][:-1]})
    result = select(reduced())
    with pytest.raises(RuntimeError, match='SHA-256'):
        validate_selection_metadata({
            'selection_sha256': 'c' * 64, 'source_selection': result.identity,
        }, result)
    with pytest.raises(RuntimeError, match='identity'):
        validate_selection_metadata({
            'selection_sha256': result.selection_sha256, 'source_selection': {},
        }, result)


def test_selection_record_is_required_but_excluded_from_stable_hash():
    result = select(reduced())
    metadata = {
        'source_selection': result.identity,
        'selection_sha256': result.selection_sha256,
        'selection_record': {
            'schema': result.identity['schema'], 'created': '2026-10-08T00:00:00Z',
            'implementation': {
                'repository': 'repo', 'branch': 'dev', 'commit': 'abc',
                'dirty': False, 'tracked_diff_sha256': 'clean',
            },
        },
    }
    validate_selection_record(metadata)
    changed_time = {
        **metadata,
        'selection_record': {**metadata['selection_record'], 'created': 'later'},
    }
    validate_selection_record(changed_time)
    assert changed_time['selection_sha256'] == metadata['selection_sha256']
    with pytest.raises(RuntimeError, match='record is missing'):
        validate_selection_record({
            'source_selection': result.identity,
            'selection_sha256': result.selection_sha256,
        })


@pytest.mark.parametrize('value, message', [
    ({'id': '', 'strategy': SHA256_RANK_STRATEGY, 'selection_seed': 0,
      'splits': {'training': {'count': 1}, 'validation': {'count': 1}}}, 'id'),
    ({'id': 'x', 'strategy': SHA256_RANK_STRATEGY, 'selection_seed': -1,
      'splits': {'training': {'count': 1}, 'validation': {'count': 1}}}, 'selection_seed'),
    ({'id': 'x', 'strategy': SHA256_RANK_STRATEGY, 'selection_seed': 0,
      'splits': {'training': {'count': 0}, 'validation': {'count': 1}}}, 'count'),
    ({'id': 'x', 'strategy': 'unknown', 'selection_seed': 0,
      'splits': {'training': {'count': 1}, 'validation': {'count': 1}}}, 'Unknown'),
])
def test_invalid_config_is_rejected(value, message):
    with pytest.raises(ValueError, match=message):
        SourceSelectionConfig.from_mapping(value, EXPECTED)


def test_only_exact_repository_profiles_are_approved():
    expected = {'training': 10_024, 'validation': 4_926}
    full = load_config_group('source_selection', 'activitynet_full_v1')
    reduced_profile = load_config_group('source_selection', 'activitynet_reduced_v1')
    assert approved_profile_name(full, expected) == 'activitynet_full_v1'
    assert approved_profile_name(reduced_profile, expected) == 'activitynet_reduced_v1'
    reduced_profile['selection_seed'] = 1
    assert approved_profile_name(reduced_profile, expected) is None
