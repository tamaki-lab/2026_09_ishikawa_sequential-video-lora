"""Resume checkpoints and Query LoRA evaluation snapshots for Streaming MoCo.

Two separate contracts:

* `resume/latest.pt` holds the complete training state at a video boundary.
  It is self-verifying (payload SHA-256), written atomically, and only used to
  continue the same run.
* `evaluation_snapshots/<name>/` holds only the Query LoRA in PEFT-native form
  for downstream evaluation. It is never used to resume MoCo training.
"""

import hashlib
import io
import random
import shutil
from pathlib import Path

import numpy
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from self_supervised.moco.vit_lora_moco import lora_parameters
from utils.artifact_io import canonical_json_bytes, read_json, sha256_bytes, sha256_file, write_bytes_atomic
from utils.artifact_io import fsync_directory, write_json_atomic


RESUME_SCHEMA = 'streaming-moco-resume/v1'
SNAPSHOT_SCHEMA = 'activitynet-moco-query-lora-snapshot/v1'
PROTOCOL_VERSION = 'activitynet-full-single-pass-streaming-moco/v1'
SNAPSHOT_FILES = ('adapter_model.safetensors', 'adapter_config.json')
# Fields that must match exactly between a checkpoint and the resuming process.
RESUME_IDENTITY = (
    'run_id', 'protocol_version', 'protocol', 'repository', 'commit', 'dependencies', 'base_model', 'base_fingerprints',
    'lora_config', 'queue_capacity', 'momentum', 'temperature', 'optimizer_config',
    'source_count', 'ordered_source_sha256',
)


def ordered_source_sha256(source_ids):
    return sha256_bytes(canonical_json_bytes(list(source_ids)))


def base_fingerprint(encoder):
    """SHA-256 of all frozen (non-LoRA) encoder tensors in name order."""
    lora = lora_parameters(encoder)
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        if name in lora or '.lora_' in name:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode() + str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lora_config(encoder):
    config = encoder.vit.peft_config['default']
    return {
        'target_modules': sorted(config.target_modules), 'r': config.r, 'lora_alpha': config.lora_alpha,
        'lora_dropout': config.lora_dropout, 'bias': config.bias,
    }


def optimizer_config(optimizer):
    group = optimizer.param_groups[0]
    return {
        'class': type(optimizer).__name__, 'lr': group['lr'], 'weight_decay': group['weight_decay'],
        'betas': list(group['betas']), 'eps': group['eps'], 'amsgrad': group['amsgrad'],
        'tensor_count': sum(len(g['params']) for g in optimizer.param_groups),
        'parameter_count': sum(p.numel() for g in optimizer.param_groups for p in g['params']),
    }


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def rng_state():
    kind, keys, position, has_gauss, cached = numpy.random.get_state()
    state = {
        'python': random.getstate(),
        'numpy': {'kind': kind, 'keys': torch.from_numpy(keys.astype(numpy.int64)), 'position': int(position),
                  'has_gauss': int(has_gauss), 'cached_gaussian': float(cached)},
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    numpy.random.set_state((
        numpy_state['kind'], numpy_state['keys'].numpy().astype(numpy.uint32), numpy_state['position'],
        numpy_state['has_gauss'], numpy_state['cached_gaussian'],
    ))
    torch.set_rng_state(state['torch_cpu'])
    if 'torch_cuda' in state:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state['torch_cuda']):
            raise RuntimeError('CUDA RNG state cannot be restored on this device set')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def training_state(moco, optimizer):
    """Every mutable MoCo training tensor, on CPU, in a deterministic order."""
    entries = moco.queue.entries
    return _cpu({
        'query_lora': lora_parameters(moco.query_encoder),
        'query_projector': moco.query_projector.state_dict(),
        'key_lora': lora_parameters(moco.key_encoder),
        'key_projector': moco.key_projector.state_dict(),
        'optimizer': optimizer.state_dict(),
        'queue': {
            'capacity': moco.queue.capacity,
            'keys': torch.stack([entry.key for entry in entries]) if entries else torch.empty(0, 128),
            'sequence_ids': [entry.sequence_id for entry in entries],
            'sequence_indices': [entry.sequence_index for entry in entries],
        },
    })


def save_resume_checkpoint(path, identity, counters, state):
    """Atomically replace `path` with a verified, self-hashed checkpoint."""
    payload = {
        'schema': RESUME_SCHEMA, 'identity': identity, 'counters': counters, 'at_video_boundary': True,
        'state': state, 'rng': rng_state(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    outer = io.BytesIO()
    torch.save({'format': RESUME_SCHEMA, 'payload_sha256': sha256_bytes(data),
                'payload': torch.frombuffer(bytearray(data), dtype=torch.uint8)}, outer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.verify')
    write_bytes_atomic(temporary, outer.getvalue())
    loaded = load_resume_checkpoint(temporary)
    if loaded['counters'] != counters or loaded['identity'] != identity:
        raise RuntimeError('Resume checkpoint verification failed after write')
    temporary.replace(path)
    fsync_directory(path.parent)
    return sha256_file(path)


def load_resume_checkpoint(path):
    """Load a complete checkpoint or raise; never partially loads."""
    try:
        outer = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:
        raise RuntimeError(f'Corrupted resume checkpoint: {path}') from error
    if not isinstance(outer, dict) or outer.get('format') != RESUME_SCHEMA:
        raise RuntimeError(f'Unsupported resume checkpoint schema: {path}')
    data = outer['payload'].numpy().tobytes()
    if sha256_bytes(data) != outer['payload_sha256']:
        raise RuntimeError(f'Resume checkpoint payload hash mismatch: {path}')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if payload.get('schema') != RESUME_SCHEMA or set(payload) != {
        'schema', 'identity', 'counters', 'at_video_boundary', 'state', 'rng',
    }:
        raise RuntimeError('Unsupported or incomplete resume checkpoint payload')
    counters = payload['counters']
    if payload['at_video_boundary'] is not True or set(counters) != {
        'processed_videos', 'next_video_index', 'global_update_step', 'next_source_id', 'final',
    } or counters['next_video_index'] != counters['processed_videos']:
        raise RuntimeError('Resume checkpoint is not at a completed video boundary')
    return payload


def validate_resume_identity(saved, current):
    mismatched = [name for name in RESUME_IDENTITY if saved.get(name) != current.get(name)]
    if mismatched or set(saved) != set(current):
        fields = mismatched or sorted(set(saved) ^ set(current))
        raise RuntimeError(f'Resume checkpoint does not match this run: {fields}')


def restore_training_state(moco, optimizer, state, device):
    for side in ('query', 'key'):
        lora = lora_parameters(getattr(moco, f'{side}_encoder'))
        saved = state[f'{side}_lora']
        if set(saved) != set(lora):
            raise RuntimeError(f'{side} LoRA tensor names differ from the checkpoint')
        with torch.no_grad():
            for name, parameter in lora.items():
                if saved[name].shape != parameter.shape:
                    raise RuntimeError(f'{side} LoRA shape differs: {name}')
                parameter.copy_(saved[name])
        getattr(moco, f'{side}_projector').load_state_dict(state[f'{side}_projector'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    queue = state['queue']
    if queue['capacity'] != moco.queue.capacity or len(moco.queue):
        raise RuntimeError('Queue capacity differs or the queue is not empty before restore')
    if not len(queue['keys']) == len(queue['sequence_ids']) == len(queue['sequence_indices']):
        raise RuntimeError('Queue keys and metadata lengths differ')
    for key, sequence_id, sequence_index in zip(queue['keys'], queue['sequence_ids'], queue['sequence_indices']):
        moco.queue.enqueue(key.to(device), sequence_id, sequence_index)


def snapshot_name(processed_videos, global_update_step, final):
    return f'videos-{processed_videos:06d}_step-{global_update_step}' + ('_final' if final else '')


def query_lora_state(encoder):
    return {name: tensor.detach().cpu() for name, tensor in get_peft_model_state_dict(encoder.vit).items()}


def write_evaluation_snapshot(snapshot_root, encoder, metadata):
    """Save only the Query LoRA via PEFT save_pretrained, verify, then publish.

    An existing directory with identical adapter weights is reused; any other
    existing content is an error rather than an overwrite.
    """
    name = snapshot_name(metadata['processed_videos'], metadata['global_update_step'], metadata['final'])
    target = Path(snapshot_root) / name
    expected = query_lora_state(encoder)
    if target.exists():
        if not all((target / file).is_file() for file in (*SNAPSHOT_FILES, 'metadata.json')):
            raise RuntimeError(f'Incomplete evaluation snapshot already exists: {target}')
        saved = load_file(target / 'adapter_model.safetensors')
        if saved.keys() != expected.keys() or any(not torch.equal(saved[k], expected[k]) for k in expected):
            raise RuntimeError(f'Different evaluation snapshot already exists: {target}')
        return target, read_json(target / 'metadata.json')
    temporary = target.with_name(f'.{name}.tmp')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    encoder.vit.save_pretrained(temporary)
    (temporary / 'README.md').unlink(missing_ok=True)
    if sorted(path.name for path in temporary.iterdir()) != sorted(SNAPSHOT_FILES):
        raise RuntimeError('PEFT snapshot must contain only adapter weights and config')
    saved = load_file(temporary / 'adapter_model.safetensors')
    if saved.keys() != expected.keys() or any(not torch.equal(saved[k], expected[k]) for k in expected):
        raise RuntimeError('Saved Query LoRA snapshot differs from the in-memory Query LoRA')
    if any('.lora_' not in key for key in saved):
        raise RuntimeError('Evaluation snapshot must contain only LoRA tensors')
    metadata = {
        'schema': SNAPSHOT_SCHEMA, **metadata,
        'files': {file: sha256_file(temporary / file) for file in SNAPSHOT_FILES},
        'local_path': str(target),
    }
    write_json_atomic(temporary / 'metadata.json', metadata)
    temporary.rename(target)
    fsync_directory(target.parent)
    return target, metadata


def load_query_lora_snapshot(encoder, snapshot):
    """Load a PEFT-native Query LoRA snapshot into a ViTLoRAFrameEncoder.

    Verifies schema, file hashes, LoRA config and that every LoRA tensor of
    the encoder is replaced by exactly the saved value.
    """
    snapshot = Path(snapshot)
    metadata = read_json(snapshot / 'metadata.json')
    if metadata.get('schema') != SNAPSHOT_SCHEMA:
        raise RuntimeError(f'Unsupported Query LoRA snapshot schema: {metadata.get("schema")}')
    actual = {file: sha256_file(snapshot / file) for file in SNAPSHOT_FILES}
    if actual != metadata['files']:
        raise RuntimeError('Query LoRA snapshot files do not match metadata hashes')
    config = read_json(snapshot / 'adapter_config.json')
    current = lora_config(encoder)
    if {key: (sorted(config[key]) if key == 'target_modules' else config[key]) for key in current} != current:
        raise RuntimeError('Query LoRA snapshot config differs from the encoder LoRA config')
    if metadata['lora_config'] != current:
        raise RuntimeError('Query LoRA snapshot metadata config differs from the encoder')
    weights = load_file(snapshot / 'adapter_model.safetensors')
    result = set_peft_model_state_dict(encoder.vit, weights)
    if result.unexpected_keys or any('.lora_' in key for key in result.missing_keys):
        raise RuntimeError('Query LoRA snapshot keys do not match the encoder')
    loaded = query_lora_state(encoder)
    if loaded.keys() != weights.keys() or any(not torch.equal(loaded[k], weights[k]) for k in weights):
        raise RuntimeError('Query LoRA snapshot was not loaded exactly')
    return metadata
