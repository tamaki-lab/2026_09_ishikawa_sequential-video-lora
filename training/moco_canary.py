"""Four chronological streams and audited, finite-length Stage 6A MoCo updates."""

from contextlib import ExitStack, contextmanager
import json

import sequential_loader as sl
import torch

from self_supervised.moco.vit_lora_moco import require_normalized
from integration.sequential_moco import make_two_views, encode_query_view, encode_key_view
from training.moco_audit import (
    audit_initial_state, audit_views, parameter_groups,
)


FRAMES_PER_CHUNK = 16
STREAM_COUNT = 4


def validate_sources(sources):
    if len(sources) != STREAM_COUNT or len({source.sequence_id for source in sources}) != STREAM_COUNT:
        raise ValueError('Expected exactly four distinct sources')


@contextmanager
def round_robin_samples(sources):
    """Yield A1/B1/C1/D1/A2/... without prefetch; close all readers on every exit.

    A final sample is consumed normally, then the whole iterator ends, even if
    the other streams still have samples. Timestamps retain their raw ordering.
    """
    validate_sources(sources)
    with ExitStack() as stack:
        streams = []
        for source in sources:
            dataset = sl.SequentialDataset(
                sources=(source,), reader=sl.SequentialVideoReader(),
                chunk_config=sl.FixedChunkConfig(frames_per_chunk=FRAMES_PER_CHUNK),
            )
            loader = sl.build_sequential_dataloader(dataset=dataset)
            streams.append(stack.enter_context(sl.sequential_sample_stream(loader)))

        def iterate():
            index = 0
            while True:
                for source, stream in zip(sources, streams):
                    sample = next(stream, None)
                    if sample is None:
                        return
                    if not isinstance(sample, sl.SequentialSample):
                        raise RuntimeError('Expected SequentialSample')
                    if (sample.sequence_id, sample.sequence_index, sample.is_first) != (
                        source.sequence_id, index, index == 0
                    ):
                        raise RuntimeError('Round-robin sequence identity or chronological chunk order changed')
                    if sample.frames.ndim != 4 or tuple(sample.frames.shape[:2]) != (16, 3):
                        raise RuntimeError('Expected 16 RGB frames including padding')
                    if sample.frames.device.type != 'cpu' or sample.frames.dtype != torch.uint8:
                        raise RuntimeError('Expected CPU uint8 frames')
                    if sample.valid_mask.shape != (16,) or sample.valid_mask.dtype != torch.bool:
                        raise RuntimeError('Expected bool valid_mask [16]')
                    count = int(sample.valid_mask.sum().item())
                    if count == 0 and sample.is_last:
                        return
                    start = source.start_frame + index * FRAMES_PER_CHUNK
                    if not count or not torch.equal(sample.valid_mask, torch.arange(16) < count) or not torch.equal(
                        sample.frame_indices[sample.valid_mask], torch.arange(start, start + count)
                    ) or (count != FRAMES_PER_CHUNK and not sample.is_last):
                        raise RuntimeError('Expected contiguous absolute frame indices and terminal-only padding')
                    yield sample
                    if sample.is_last:
                        return
                index += 1

        samples = iterate()
        try:
            yield samples
        finally:
            samples.close()


def _report(event, **values):
    print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)


def _snapshot(parameters):
    return {name: p.detach().cpu().clone() for name, p in parameters.items()}


def _changed(parameters, before):
    return sum(not torch.equal(p.detach().cpu(), before[name]) for name, p in parameters.items())


def _audit_bases(groups, before):
    for label in ('query base', 'key base'):
        if _changed(groups[label], before):
            raise RuntimeError(f'{label} changed')


def _audit_finite(moco):
    if not all(torch.isfinite(p).all().item() for p in moco.parameters()):
        raise RuntimeError('Model parameter contains NaN or Inf')


def _audit_queue(moco, expected):
    entries = moco.queue.entries
    if len(entries) != len(expected):
        raise RuntimeError('Queue count changed unexpectedly or eviction occurred')
    for entry, (sequence_id, sequence_index, key) in zip(entries, expected):
        require_normalized(entry.key)
        if entry.key.requires_grad or entry.key.grad_fn is not None:
            raise RuntimeError('Queue keys must be detached')
        if (entry.sequence_id, entry.sequence_index) != (sequence_id, sequence_index) or not torch.equal(entry.key, key):
            raise RuntimeError('Queue key / metadata alignment changed')


def _enqueue(moco, key, sample, expected):
    require_normalized(key)
    if key.requires_grad or key.grad_fn is not None:
        raise RuntimeError('Current Key must be detached')
    # Independent reference checks FIFO history, metadata and the pre-EMA key.
    expected.append((sample.sequence_id, sample.sequence_index, key.detach().clone()))
    moco.queue.enqueue(key, sample.sequence_id, sample.sequence_index)
    _audit_queue(moco, expected)


def _gradient_diagnostics(groups):
    diagnostics = {}
    for label, parameters in groups.items():
        gradients = [p.grad for p in parameters.values() if p.grad is not None]
        finite = sum(bool(torch.isfinite(grad).all().item()) for grad in gradients)
        nonzero = sum(bool(torch.count_nonzero(grad).item()) for grad in gradients)
        if label in ('query LoRA', 'query Projector'):
            if finite != len(parameters) or nonzero == 0:
                raise RuntimeError(f'Expected finite nonzero {label} gradients')
            norm = torch.stack([grad.detach().double().norm() for grad in gradients]).norm().item()
            diagnostics[label] = {'finite': finite, 'nonzero': nonzero, 'norm': norm}
        elif gradients:
            raise RuntimeError(f'Unexpected {label} gradient')
    diagnostics['query base gradient count'] = 0
    diagnostics['key gradient count'] = 0
    return diagnostics


def _train_step(moco, processor, sample, device, optimizer, groups, before, expected_queue, step):
    query_view, key_view = make_two_views(sample)
    audit_views(sample, query_view, key_view)
    optimizer.zero_grad(set_to_none=True)
    query = encode_query_view(query_view, processor, moco, device)[-1]
    key = encode_key_view(key_view, processor, moco, device)[-1]
    require_normalized(key)
    if key.requires_grad:
        raise RuntimeError('Current Key must be detached')
    _audit_queue(moco, expected_queue)
    old_entries = moco.queue.entries
    expected_negatives = tuple(entry for entry in old_entries if entry.sequence_id != sample.sequence_id)
    if len(expected_negatives) < 3:
        raise RuntimeError('Expected at least three different-sequence negatives')
    loss, logits, negatives = moco.contrastive_loss(query, key, sample.sequence_id)
    if len(negatives) != len(expected_negatives) or any(
        actual is not original for actual, original in zip(negatives, expected_negatives)
    ):
        raise RuntimeError('Negatives must exactly match the different-sequence entries in the old queue')
    if loss.ndim != 0 or tuple(logits.shape) != (1, 1 + len(negatives)) or not (
        torch.isfinite(loss).item() and torch.isfinite(logits).all().item()
    ):
        raise RuntimeError('Expected finite scalar InfoNCE loss and logits')
    key_parameters = {**groups['key LoRA'], **groups['key Projector']}
    key_before = _snapshot(key_parameters)
    loss.backward()
    gradients = _gradient_diagnostics(groups)
    optimizer.step()
    if _changed(key_parameters, key_before):
        raise RuntimeError('Key changed before EMA')
    _audit_bases(groups, before)
    moco.update_key()
    query_parameters = {**groups['query LoRA'], **groups['query Projector']}
    for name, p in key_parameters.items():
        query_name = name.replace('key_', 'query_', 1)
        query_p = query_parameters[query_name]
        expected = key_before[name] * moco.momentum + query_p.detach().cpu() * (1. - moco.momentum)
        if not torch.allclose(p.detach().cpu(), expected, rtol=1e-6, atol=1e-8):
            raise RuntimeError('Key EMA does not match the Stage 5 formula')
    _audit_bases(groups, before)
    _audit_finite(moco)
    _audit_queue(moco, expected_queue)
    _enqueue(moco, key, sample, expected_queue)
    _report(
        'step', step=step, sequence_id=sample.sequence_id, sequence_index=sample.sequence_index,
        valid_frame_count=int(sample.valid_mask.sum().item()), loss=loss.item(),
        positive_similarity=(query.detach() @ key).item(),
        negative_logits_min=logits[0, 1:].min().item(), negative_logits_max=logits[0, 1:].max().item(),
        valid_negative_count=len(negatives), queue_count=len(moco.queue),
        queue_unique_sequence_id_count=len({entry.sequence_id for entry in moco.queue.entries}),
        gradients=gradients, all_parameters_finite=True, all_queue_keys_finite=True,
    )


def run_canary(moco, processor, sources, device, max_steps=10):
    """Start fresh, warm up four keys, then perform exactly max_steps updates.

    EOF before the target raises rather than reporting a successful canary.
    Models, queues and optimizers are never resumed across calls.
    """
    validate_sources(sources)
    if type(max_steps) is not int or not 1 <= max_steps <= moco.queue.capacity - STREAM_COUNT:
        raise ValueError('max_steps must be a positive integer fitting the queue without eviction')
    audit_initial_state(moco)
    groups = parameter_groups(moco)
    _audit_finite(moco)
    if any(p.grad is not None for p in moco.parameters()):
        raise RuntimeError('Fresh model must have no gradients')
    before = _snapshot(dict(moco.named_parameters()))
    expected_queue = []
    completed = 0
    last_sample = None
    try:
        with round_robin_samples(sources) as samples:
            def take_sample():
                nonlocal last_sample
                sample = next(samples, None)
                if sample is None:
                    last = None if last_sample is None else (last_sample.sequence_id, last_sample.sequence_index)
                    raise RuntimeError(f'EOF after {completed}/{max_steps} training steps; last sample: {last}')
                last_sample = sample
                return sample

            for _ in range(STREAM_COUNT):
                sample = take_sample()
                query_view, key_view = make_two_views(sample)
                audit_views(sample, query_view, key_view)
                key = encode_key_view(key_view, processor, moco, device)[-1]
                _enqueue(moco, key, sample, expected_queue)
                _report('warmup', sequence_id=sample.sequence_id, sequence_index=sample.sequence_index,
                        valid_frame_count=int(sample.valid_mask.sum().item()), queue_count=len(moco.queue))
            if len(moco.queue) != STREAM_COUNT:
                raise RuntimeError('Expected exactly four warm-up keys')
            if _changed(dict(moco.named_parameters()), before) or any(p.grad is not None for p in moco.parameters()):
                raise RuntimeError('Warm-up must not change model parameters or create gradients')
            _report('warmup_complete', queue_count=len(moco.queue))
            optimizer = torch.optim.AdamW(moco.query_parameters(), lr=1.0e-3, weight_decay=0.0)
            optimized = [p for group in optimizer.param_groups for p in group['params']]
            trainable = {**groups['query LoRA'], **groups['query Projector']}
            if len(optimized) != len(trainable) or {id(p) for p in optimized} != {id(p) for p in trainable.values()}:
                raise RuntimeError('Optimizer must contain exactly Query LoRA + Query Projector')
            _report('optimizer', parameters=sum(p.numel() for p in optimized), tensors=len(optimized),
                    matches_query_lora_and_projector=True, lr=1e-3, weight_decay=0.0)
            for step in range(max_steps):
                sample = take_sample()
                _train_step(moco, processor, sample, device, optimizer, groups, before, expected_queue, step)
                completed += 1
        _audit_bases(groups, before)
        _audit_finite(moco)
        _audit_queue(moco, expected_queue)
        changes = {label: _changed(parameters, before) for label, parameters in groups.items()}
        if any(changes[label] == 0 for label in ('query LoRA', 'query Projector', 'key LoRA', 'key Projector')):
            raise RuntimeError('Expected Query updates and Key EMA changes from the initial state')
        result = dict(training_steps=completed, max_steps=max_steps, queue_count=len(moco.queue),
                      changed_tensors=changes, all_parameters_finite=True, all_queue_keys_finite=True)
        _report('complete', **result)
        return result
    except Exception as error:
        _report('stopped', training_steps=completed, max_steps=max_steps, reason=str(error))
        raise
