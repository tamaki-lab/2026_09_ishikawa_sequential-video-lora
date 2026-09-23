"""Verify MoCo mechanics on the first chunks of two ActivityNet training videos.

Usage:
    python smoke_activitynet_vit_lora_moco_one_step.py /path/to/ActivityNet

This performs one CPU-default engineering step, not representation evaluation.
"""

import argparse
from pathlib import Path

import peft
import sequential_loader as sl
import torch
import transformers
from transformers import AutoImageProcessor

from model.backbones.vit import ViTLoRAFrameEncoder
from self_supervised.moco import ViTLoRAMoCo
from self_supervised.moco.vit_lora_moco import lora_parameters, require_normalized
from sequential_moco_bridge import make_two_views, encode_query_view, encode_key_view
from smoke_activitynet_vit_lora_one_step import (
    CHECKPOINT_ID, EXPECTED_BRANCH, LOADER_BRANCH, LOADER_COMMIT,
    FRAMES_PER_CHUNK, audit_parameters as audit_encoder, git_output,
)


BASE_COMMIT = '3b980c8a90e8cad08e78a5ba13cbb3629ec5ae58'


def audit_provenance():
    root = Path(__file__).resolve().parent
    branch = git_output(root, 'branch', '--show-current')
    revision = git_output(root, 'rev-parse', 'HEAD')
    origin = git_output(root, 'remote', 'get-url', 'origin')
    if origin not in (
        'git@github.com:tamaki-lab/2026_09_ishikawa_sequential-video-lora.git',
        'https://github.com/tamaki-lab/2026_09_ishikawa_sequential-video-lora.git',
    ):
        raise RuntimeError(f'Unexpected implementation repository: {origin}')
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f'Expected implementation branch {EXPECTED_BRANCH}, got {branch}')
    git_output(root, 'merge-base', '--is-ancestor', BASE_COMMIT, 'HEAD')
    loader_root = Path(sl.__file__).resolve().parent.parent
    loader_branch = git_output(loader_root, 'branch', '--show-current')
    loader_revision = git_output(loader_root, 'rev-parse', 'HEAD')
    if (loader_branch, loader_revision) != (LOADER_BRANCH, LOADER_COMMIT):
        raise RuntimeError('Unexpected sequential_loader branch or revision')
    if git_output(loader_root, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('Pinned sequential_loader checkout has tracked changes')
    if transformers.__version__ != '5.17.0' or peft.__version__ != '0.21.0':
        raise RuntimeError('Stage 5 requires transformers==5.17.0 and peft==0.21.0')
    print(f'implementation origin: {origin}')
    print(f'implementation branch / HEAD: {branch} / {revision}')
    print(f'implementation base: {BASE_COMMIT}')
    print(f'sequential_loader branch / HEAD: {loader_branch} / {loader_revision}')
    print(f'checkpoint: {CHECKPOINT_ID}')
    print(f'PyTorch / Transformers / PEFT: {torch.__version__} / {transformers.__version__} / {peft.__version__}')


def parameter_groups(moco):
    groups = {}
    for side in ('query', 'key'):
        encoder = getattr(moco, f'{side}_encoder')
        lora = lora_parameters(encoder)
        groups[f'{side} base'] = {
            f'{side}_encoder.{name}': p for name, p in encoder.named_parameters() if name not in lora
        }
        groups[f'{side} LoRA'] = {f'{side}_encoder.{name}': p for name, p in lora.items()}
        groups[f'{side} Projector'] = {
            f'{side}_projector.{name}': p for name, p in getattr(moco, f'{side}_projector').named_parameters()
        }
    return groups


def audit_parameters(moco):
    audit_encoder(moco.query_encoder)
    for side in ('query', 'key'):
        encoder = getattr(moco, f'{side}_encoder')
        config = encoder.vit.peft_config['default']
        if config.target_modules != {'q_proj', 'v_proj'} or (
            config.r, config.lora_alpha, config.lora_dropout, config.bias
        ) != (8, 8, 0.0, 'none'):
            raise RuntimeError(f'Unexpected {side} LoRA configuration')
        projector = getattr(moco, f'{side}_projector')
        if len(projector) != 3 or not isinstance(projector[0], torch.nn.Linear) or not isinstance(
            projector[1], torch.nn.ReLU
        ) or not isinstance(projector[2], torch.nn.Linear):
            raise RuntimeError('Expected Linear / ReLU / Linear projector')
        if (projector[0].in_features, projector[0].out_features,
            projector[2].in_features, projector[2].out_features) != (768, 768, 768, 128):
            raise RuntimeError('Expected 768 -> 768 -> 128 projector')
    groups = parameter_groups(moco)
    for side in ('query', 'key'):
        for kind, count, tensors in (('LoRA', 294_912, 48), ('Projector', 689_024, 4)):
            parameters = groups[f'{side} {kind}']
            actual = sum(p.numel() for p in parameters.values())
            print(f'{side} {kind} params / tensors: {actual} / {len(parameters)}')
            if (actual, len(parameters)) != (count, tensors):
                raise RuntimeError(f'Unexpected {side} {kind} parameter count')
    expected = {**groups['query LoRA'], **groups['query Projector']}
    trainable = {name: p for name, p in moco.named_parameters() if p.requires_grad}
    if set(trainable) != set(expected):
        raise RuntimeError('Only Query LoRA and Query Projector may be trainable')
    if (moco.momentum, moco.temperature, moco.queue.capacity) != (0.999, 0.07, 4096):
        raise RuntimeError('Unexpected momentum, temperature or queue capacity')
    return groups


def audit_initial_state(moco):
    audit_parameters(moco)
    for query, key in ((moco.query_encoder, moco.key_encoder), (moco.query_projector, moco.key_projector)):
        query_state, key_state = query.state_dict(), key.state_dict()
        if query_state.keys() != key_state.keys() or any(
            not torch.equal(query_state[name], key_state[name]) for name in query_state
        ):
            raise RuntimeError('Query / Key initial states differ')
        key_parameters = dict(key.named_parameters())
        if any(p.data_ptr() == key_parameters[name].data_ptr() for name, p in query.named_parameters()):
            raise RuntimeError('Query / Key parameters must have independent storage')
    if len(moco.queue) != 0:
        raise RuntimeError('Queue must start empty')
    print('Query / Key initial states equal, independent storage: True')


def read_first_chunk(source):
    dataset = sl.SequentialDataset(
        sources=(source,), reader=sl.SequentialVideoReader(),
        chunk_config=sl.FixedChunkConfig(frames_per_chunk=FRAMES_PER_CHUNK),
    )
    loader = sl.build_sequential_dataloader(dataset=dataset)
    with sl.sequential_sample_stream(loader) as samples:
        sample = next(samples)
    if not isinstance(sample, sl.SequentialSample):
        raise RuntimeError('Expected SequentialSample')
    if (sample.sequence_id, sample.sequence_index, sample.is_first) != (source.sequence_id, 0, True):
        raise RuntimeError('Expected the first chunk of the selected source')
    if sample.frames.ndim != 4 or tuple(sample.frames.shape[:2]) != (16, 3):
        raise RuntimeError('Expected 16 RGB frames including padding')
    if sample.frames.device.type != 'cpu' or sample.frames.dtype != torch.uint8:
        raise RuntimeError('Expected CPU uint8 frames')
    if sample.valid_mask.shape != (16,) or sample.valid_mask.dtype != torch.bool:
        raise RuntimeError('Expected bool valid_mask [16]')
    count = int(sample.valid_mask.sum().item())
    if not count or not torch.equal(sample.frame_indices[sample.valid_mask], torch.arange(count)):
        raise RuntimeError('Expected valid contiguous frame indices starting at zero')
    return sample


def report_sample(label, sample):
    print(f'{label} sequence_id / sequence_index: {sample.sequence_id} / {sample.sequence_index}')
    print(f'{label} frame_indices: {sample.frame_indices.tolist()}')
    print(f'{label} timestamps: {sample.timestamps.tolist()}')
    print(f'{label} valid count / T: {int(sample.valid_mask.sum())}/16')


def audit_views(sample, query, key):
    for view in (query, key):
        if (view.sequence_id, view.sequence_index) != (sample.sequence_id, sample.sequence_index):
            raise RuntimeError('Two-view sequence metadata differs')
        for name in ('frame_indices', 'valid_mask', 'timestamps'):
            if not torch.allclose(getattr(view, name), getattr(sample, name), rtol=0, atol=0, equal_nan=True):
                raise RuntimeError(f'Two-view {name} differs')
    if not torch.equal(query.frames, sample.frames) or not torch.equal(
        key.frames[sample.valid_mask], sample.frames[sample.valid_mask].flip(-1)
    ) or not torch.equal(key.frames[~sample.valid_mask], sample.frames[~sample.valid_mask]):
        raise RuntimeError('Expected raw Query and valid-frame horizontal-flip Key')


def report_features(label, result, sample):
    pixels, frames, clip, projected = result
    for name, value, shape in (
        ('pixel_values', pixels, (int(sample.valid_mask.sum()), 3, 224, 224)),
        ('frame_features', frames, (16, 768)), ('clip_feature', clip, (768,)), ('projected', projected, (128,)),
    ):
        print(f'{label} {name} shape: {tuple(value.shape)}')
        if tuple(value.shape) != shape or not torch.isfinite(value).all().item():
            raise RuntimeError(f'Unexpected or nonfinite {label} {name}')
    if torch.count_nonzero(frames[~sample.valid_mask.to(frames.device)]).item():
        raise RuntimeError('Padding frame features must be zero')
    require_normalized(projected)


def run_one_step(moco, query, key, sample):
    groups = audit_parameters(moco)
    expected = {**groups['query LoRA'], **groups['query Projector']}
    optimizer = torch.optim.AdamW(moco.query_parameters(), lr=1.0e-3, weight_decay=0.0)
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    expected_ids = {id(p) for p in expected.values()}
    unexpected = [name for name, p in moco.named_parameters()
                  if any(p is optimized_p for optimized_p in optimized) and id(p) not in expected_ids]
    print(f'optimizer total params / tensors: {sum(p.numel() for p in optimized)} / {len(optimized)}')
    print(f'unexpected optimizer params: {unexpected}')
    if len(optimized) != 52 or {id(p) for p in optimized} != expected_ids:
        raise RuntimeError('Optimizer must contain exactly Query LoRA + Query Projector')
    before = {name: p.detach().cpu().clone() for name, p in moco.named_parameters()}
    queue_before = moco.queue.entries
    loss, logits, negatives = moco.contrastive_loss(query, key, sample.sequence_id)
    if any(entry.sequence_id == sample.sequence_id for entry in negatives):
        raise RuntimeError('Same-sequence key leaked into negatives')
    print(f'valid negative count: {len(negatives)}')
    print(f'negative sequence_ids: {[entry.sequence_id for entry in negatives]}')
    print(f'positive similarity: {(query @ key).item()}')
    print(f'negative logits min/max: {logits[0, 1:].min().item()} / {logits[0, 1:].max().item()}')
    print(f'temperature: {moco.temperature}')
    print(f'loss: {loss.item()}')
    print(f'loss finite: {bool(torch.isfinite(loss).item())}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for label, parameters in groups.items():
        gradients = [p.grad for p in parameters.values() if p.grad is not None]
        finite = sum(bool(torch.isfinite(grad).all().item()) for grad in gradients)
        nonzero = sum(bool(torch.count_nonzero(grad).item()) for grad in gradients)
        print(f'{label} gradient count / finite / nonzero: {len(gradients)} / {finite} / {nonzero}')
        if label in ('query LoRA', 'query Projector'):
            if finite != len(parameters) or nonzero == 0:
                raise RuntimeError(f'Expected finite nonzero {label} gradients')
        elif gradients:
            raise RuntimeError(f'Unexpected {label} gradient')

    optimizer.step()
    for label in ('key base', 'key LoRA', 'key Projector'):
        if any(not torch.equal(before[name], p.detach().cpu()) for name, p in groups[label].items()):
            raise RuntimeError('Key branch changed before EMA')
    for kind in ('LoRA', 'Projector'):
        if all(torch.equal(p, groups[f'key {kind}'][name.replace('query_', 'key_', 1)])
               for name, p in groups[f'query {kind}'].items()):
            raise RuntimeError(f'Query / Key {kind} must differ before EMA')

    moco.update_key()
    for kind in ('LoRA', 'Projector'):
        for name, p in groups[f'key {kind}'].items():
            query_p = groups[f'query {kind}'][name.replace('key_', 'query_', 1)]
            expected_value = before[name] * moco.momentum + query_p.detach().cpu() * (1. - moco.momentum)
            if not torch.allclose(p.detach().cpu(), expected_value, rtol=1e-6, atol=1e-8):
                raise RuntimeError('Key EMA does not match m=0.999 formula')
    for label, parameters in groups.items():
        changed = sum(not torch.equal(before[name], p.detach().cpu()) for name, p in parameters.items())
        print(f'{label} changed count: {changed}')
        if (label.endswith('base') and changed != 0) or (not label.endswith('base') and changed == 0):
            raise RuntimeError(f'Unexpected {label} parameter change count')
    if any(p.grad is not None for p in list(moco.key_encoder.parameters()) + list(moco.key_projector.parameters())):
        raise RuntimeError('Key branch must remain gradient-free after EMA')
    if len(moco.queue) != len(queue_before) or any(
        actual is not original for actual, original in zip(moco.queue.entries, queue_before)
    ):
        raise RuntimeError('Queue changed before the post-EMA enqueue')
    moco.queue.enqueue(key, sample.sequence_id, sample.sequence_index)
    print(f'queue count after training enqueue: {len(moco.queue)}')
    parameters_finite = all(torch.isfinite(p).all().item() for p in moco.parameters())
    keys_finite = all(torch.isfinite(entry.key).all().item() for entry in moco.queue.entries)
    print(f'all parameters finite: {parameters_finite}')
    print(f'all queue keys finite: {keys_finite}')
    if not parameters_finite or not keys_finite:
        raise RuntimeError('Nonfinite parameters or queue keys after step / EMA')
    return loss.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    audit_provenance()
    device = torch.device(args.device)
    print(f'device: {device}')
    adapter = sl.ActivityNetAdapter(dataset_root=args.dataset_root)
    sources = adapter.sequence_sources('training')
    if len(sources) < 2 or sources[0].sequence_id == sources[1].sequence_id:
        raise RuntimeError('Expected two distinct ActivityNet training sources')
    warmup_sample, training_sample = (read_first_chunk(source) for source in sources[:2])
    print('dataset / split: ActivityNet v1.3 / training')
    report_sample('warm-up', warmup_sample)
    report_sample('training', training_sample)
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    moco = ViTLoRAMoCo(ViTLoRAFrameEncoder(CHECKPOINT_ID)).to(device).train()
    audit_initial_state(moco)
    print(f'queue capacity / count before warm-up: {moco.queue.capacity} / {len(moco.queue)}')
    warmup_query, warmup_key = make_two_views(warmup_sample)
    audit_views(warmup_sample, warmup_query, warmup_key)
    warmup = encode_key_view(warmup_key, processor, moco, device)
    report_features('warm-up key', warmup, warmup_sample)
    moco.queue.enqueue(warmup[-1], warmup_sample.sequence_id, warmup_sample.sequence_index)
    print(f'queue count after warm-up: {len(moco.queue)}')
    if len(moco.queue) != 1:
        raise RuntimeError('Expected one warm-up key')
    query_view, key_view = make_two_views(training_sample)
    audit_views(training_sample, query_view, key_view)
    query = encode_query_view(query_view, processor, moco, device)
    key = encode_key_view(key_view, processor, moco, device)
    report_features('query', query, training_sample)
    report_features('key', key, training_sample)
    run_one_step(moco, query[-1], key[-1], training_sample)
    if len(moco.queue) != 2:
        raise RuntimeError('Expected two queue entries after the training step')
    print('Stage 5 one-step mechanics smoke: PASS')


if __name__ == '__main__':
    main()
