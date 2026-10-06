"""Repository, dependency and environment provenance for production artifacts."""

from datetime import datetime, timezone
import hashlib
import platform
import subprocess
from pathlib import Path

import numpy
import peft
import sequential_loader as sl
import torch
import transformers

from utils.configuration import load_config_group, plain_config


_DEFAULT_POLICY = load_config_group('provenance', 'research_v1')
_DEFAULT_ENCODER = load_config_group('encoder', 'vit_base_patch16_224')
_DEFAULT_TRACKING = load_config_group('tracking', 'default')

# Compatibility aliases for smoke scripts and old imports.  Their source of
# truth is the Hydra group YAML, rather than a second collection of literals.
REPOSITORY = _DEFAULT_POLICY['repository']
REPOSITORY_URLS = tuple(_DEFAULT_POLICY['repository_urls'])
LOADER_BRANCH = _DEFAULT_POLICY['sequential_loader']['branch']
LOADER_COMMIT = _DEFAULT_POLICY['sequential_loader']['commit']
TRANSFORMERS_VERSION = _DEFAULT_POLICY['dependencies']['transformers']
PEFT_VERSION = _DEFAULT_POLICY['dependencies']['peft']
CHECKPOINT_ID = _DEFAULT_ENCODER['checkpoint_id']
COMET_PROJECT = _DEFAULT_TRACKING['comet_project']


def git_output(repository, *args):
    return subprocess.check_output(['git', '-C', str(repository), *args], text=True).strip()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def device_identity(device):
    """Stable identity of the selected device and visible CUDA device order."""
    device = torch.device(device)
    if device.type == 'cpu':
        return {'type': 'cpu'}
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError(f'Unavailable production device: {device}')
    selected = torch.cuda.current_device() if device.index is None else device.index

    def cuda_device(index):
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, 'uuid', None)
        return {
            'index': index, 'name': properties.name, 'total_memory': int(properties.total_memory),
            'uuid': None if uuid is None else str(uuid),
        }

    visible = [cuda_device(index) for index in range(torch.cuda.device_count())]
    return {'type': 'cuda', 'selected_index': selected, 'selected': visible[selected], 'visible_devices': visible}


def collect_provenance(*, policy=None, require_clean=False):
    """Fail on an unexpected repository or pinned loader; record dirty state.

    Tracked changes in this repository are recorded rather than refused, so
    that development smoke runs remain possible; artifacts carry the flag.
    """
    policy = _DEFAULT_POLICY if policy is None else plain_config(policy)
    repository = policy['repository']
    repository_urls = tuple(policy['repository_urls'])
    loader_policy = policy['sequential_loader']
    dependencies = policy['dependencies']
    root = Path(__file__).resolve().parent.parent
    origin = git_output(root, 'remote', 'get-url', 'origin')
    if origin not in repository_urls:
        raise RuntimeError(f'Unexpected implementation repository: {origin}')
    dirty = git_output(root, 'status', '--porcelain', '--untracked-files=all')
    loader_root = Path(sl.__file__).resolve().parent.parent
    loader = {
        'branch': git_output(loader_root, 'branch', '--show-current'),
        'commit': git_output(loader_root, 'rev-parse', 'HEAD'),
        'dirty': bool(git_output(loader_root, 'status', '--porcelain', '--untracked-files=all')),
    }
    if (loader['branch'], loader['commit']) != (
        loader_policy['branch'], loader_policy['commit'],
    ) or loader['dirty']:
        raise RuntimeError(f'Unexpected or modified sequential_loader checkout: {loader}')
    if (transformers.__version__, peft.__version__) != (
        dependencies['transformers'], dependencies['peft'],
    ):
        raise RuntimeError(
            f'Requires transformers=={dependencies["transformers"]} and peft=={dependencies["peft"]}'
        )
    provenance = {
        'implementation': {
            'repository': repository,
            'origin': origin,
            'branch': git_output(root, 'branch', '--show-current'),
            'commit': git_output(root, 'rev-parse', 'HEAD'),
            'dirty': bool(dirty),
            'dirty_files': dirty.splitlines(),
            # With the commit, identifies the exact tracked source even for dirty runs.
            'tracked_diff_sha256': hashlib.sha256(
                subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--binary'])).hexdigest(),
        },
        'sequential_loader': loader,
        # Plain str: torch.__version__ is a TorchVersion, which weights_only loads reject.
        'versions': {name: str(version) for name, version in {
            'python': platform.python_version(),
            'numpy': numpy.__version__,
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'peft': peft.__version__,
            'sequential_loader': sl.__version__,
        }.items()},
    }
    if require_clean and provenance['implementation']['dirty']:
        raise RuntimeError(
            'Production execution requires a clean implementation checkout; found: '
            f'{provenance["implementation"]["dirty_files"]}'
        )
    return provenance
