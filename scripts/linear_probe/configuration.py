"""Resolve the shared Hydra scientific presets used by Linear Probe jobs."""

from utils.artifact_io import canonical_json_bytes, sha256_bytes
from utils.configuration import load_config_group, plain_config


CANONICAL_PRESETS = {
    'activitynet': 'v1_3',
    'encoder': 'vit_base_patch16_224',
    'sequential': 'default',
    'moco': 'stage6b_v2',
    'provenance': 'research_v1',
    'linear_probe': 'lp_v1',
}


def _science_view(group, value):
    """Remove operational names from a scientific identity.

    Artifact destinations and Comet names live in the ``tracking`` group and
    therefore never make otherwise identical features/results incompatible.
    This projection also makes the boundary explicit if operational fields are
    added to the Linear Probe preset in the future.
    """
    value = plain_config(value)
    if group != 'linear_probe':
        return value
    return {
        'id': value['id'],
        'conditions': {
            name: {key: item for key, item in condition.items() if key != 'artifact_name'}
            for name, condition in value['conditions'].items()
        },
        'feature_definition': value['feature_definition'],
        'probe': value['probe'],
    }


def resolved_science(cfg, groups=None):
    groups = tuple(CANONICAL_PRESETS) if groups is None else tuple(groups)
    unknown = set(groups) - set(CANONICAL_PRESETS)
    if unknown:
        raise ValueError(f'Unknown scientific config groups: {sorted(unknown)}')
    return {group: _science_view(group, cfg[group]) for group in groups}


def science_contract(cfg, groups=None):
    resolved = resolved_science(cfg, groups)
    canonical = {
        group: _science_view(group, load_config_group(group, CANONICAL_PRESETS[group]))
        for group in resolved
    }
    return {
        'config': resolved,
        'sha256': sha256_bytes(canonical_json_bytes(resolved)),
        'canonical': resolved == canonical,
        'presets': {group: CANONICAL_PRESETS[group] for group in resolved},
    }


def feature_science_contract(cfg):
    """Scientific identity needed to create features, excluding probe training."""
    groups = ('activitynet', 'encoder', 'sequential', 'moco', 'provenance')
    resolved = resolved_science(cfg, groups)
    linear_probe = _science_view('linear_probe', cfg['linear_probe'])
    resolved['feature_protocol'] = {
        key: linear_probe[key] for key in ('conditions', 'feature_definition')
    }
    canonical = {
        group: load_config_group(group, CANONICAL_PRESETS[group])
        for group in groups
    }
    canonical_probe = _science_view(
        'linear_probe', load_config_group('linear_probe', CANONICAL_PRESETS['linear_probe'])
    )
    canonical['feature_protocol'] = {
        key: canonical_probe[key] for key in ('conditions', 'feature_definition')
    }
    return {
        'config': resolved,
        'sha256': sha256_bytes(canonical_json_bytes(resolved)),
        'canonical': resolved == canonical,
        'presets': {group: CANONICAL_PRESETS[group] for group in groups}
        | {'feature_protocol': CANONICAL_PRESETS['linear_probe']},
    }
