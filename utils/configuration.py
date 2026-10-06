"""Read repository Hydra configuration without leaking OmegaConf objects.

Runtime entry points use Hydra composition.  Lower-level modules and artifact
validators use this helper to read the immutable canonical group choices and
always receive ordinary Python mappings, lists and scalar values.
"""

from copy import deepcopy
from pathlib import Path
import re

from omegaconf import OmegaConf


CONFIG_ROOT = Path(__file__).resolve().parent.parent / 'conf'
_CHOICE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')


def plain_config(config, *, resolve=True, throw_on_missing=True):
    """Return a deep, JSON/checkpoint-safe container for an OmegaConf value."""
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    value = OmegaConf.to_container(config, resolve=resolve, throw_on_missing=throw_on_missing)
    if not isinstance(value, dict):
        raise TypeError('Expected a mapping configuration')
    return deepcopy(value)


def load_config_group(group, name):
    """Load ``conf/<group>/<name>.yaml`` as a plain mapping.

    Group and choice names are deliberately restricted because callers select
    repository-owned canonical contracts, not arbitrary filesystem paths.
    """
    if not all(isinstance(value, str) and _CHOICE.fullmatch(value) for value in (group, name)):
        raise ValueError('Config group and choice must be simple names')
    path = CONFIG_ROOT / group / f'{name}.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'Unknown config choice: {group}={name}')
    return plain_config(OmegaConf.load(path))


def validate_path_component(name, value):
    """Validate an identifier before it is used as one directory component."""
    if not isinstance(value, str) or not _CHOICE.fullmatch(value):
        raise ValueError(f'{name} may contain only letters, digits, ".", "_" and "-"')
    return value
