"""Hashing and atomic local writes shared by MoCo and Linear Probe artifacts."""

import hashlib
import json
import os
from pathlib import Path


def canonical_json_bytes(value):
    """Stable bytes for hashing: sorted keys, no whitespace, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_atomic(path, data):
    """Write to a sibling temporary file, fsync, then replace."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('wb') as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def write_json_atomic(path, value):
    write_bytes_atomic(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def read_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)
