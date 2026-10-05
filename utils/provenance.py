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


REPOSITORY = 'tamaki-lab/2026_09_ishikawa_sequential-video-lora'
REPOSITORY_URLS = (
    'git@github.com:tamaki-lab/2026_09_ishikawa_sequential-video-lora.git',
    'https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora.git',
)
LOADER_BRANCH = 'ActivityNet'
LOADER_COMMIT = '19a0ed7e4c00300214bc9a2fe12da8c72c0499c0'
TRANSFORMERS_VERSION = '5.17.0'
PEFT_VERSION = '0.21.0'
CHECKPOINT_ID = 'google/vit-base-patch16-224'
COMET_PROJECT = '2026-09-ishikawa-sequential-video-lora'


def git_output(repository, *args):
    return subprocess.check_output(['git', '-C', str(repository), *args], text=True).strip()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def collect_provenance():
    """Fail on an unexpected repository or pinned loader; record dirty state.

    Tracked changes in this repository are recorded rather than refused, so
    that development smoke runs remain possible; artifacts carry the flag.
    """
    root = Path(__file__).resolve().parent.parent
    origin = git_output(root, 'remote', 'get-url', 'origin')
    if origin not in REPOSITORY_URLS:
        raise RuntimeError(f'Unexpected implementation repository: {origin}')
    dirty = git_output(root, 'status', '--porcelain', '--untracked-files=no')
    loader_root = Path(sl.__file__).resolve().parent.parent
    loader = {
        'branch': git_output(loader_root, 'branch', '--show-current'),
        'commit': git_output(loader_root, 'rev-parse', 'HEAD'),
        'dirty': bool(git_output(loader_root, 'status', '--porcelain', '--untracked-files=no')),
    }
    if (loader['branch'], loader['commit']) != (LOADER_BRANCH, LOADER_COMMIT) or loader['dirty']:
        raise RuntimeError(f'Unexpected or modified sequential_loader checkout: {loader}')
    if (transformers.__version__, peft.__version__) != (TRANSFORMERS_VERSION, PEFT_VERSION):
        raise RuntimeError(f'Requires transformers=={TRANSFORMERS_VERSION} and peft=={PEFT_VERSION}')
    return {
        'implementation': {
            'repository': REPOSITORY,
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
