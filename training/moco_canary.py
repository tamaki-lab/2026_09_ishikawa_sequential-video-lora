"""Bounded-step Streaming MoCo canary runs (Stage 6A canary / Stage 6B smoke).

Each run starts fresh, warms up every active stream, performs a fixed number of
updates and stops. The per-chunk update lives in training.streaming_moco and
the ordered streams in integration.sequential_stream.
"""

from collections import deque

import torch

from training.moco_protocol import STAGE6A_PROTOCOL
from training.moco_config import canonical_moco_config
from integration.sequential_moco import make_two_views, encode_key_view
from integration.sequential_stream import (  # noqa: F401 - re-exported for existing callers
    FRAMES_PER_CHUNK, STREAM_COUNT, ordered_samples, validate_sources,
)
from training.moco_audit import (
    audit_initial_state, audit_views, parameter_groups,
)
from training.streaming_moco import (
    audit_bases, audit_finite, audit_queue, count_changed, cpu_snapshot, enqueue_key, report_json,
    train_step,
)


def round_robin_samples(sources, *, frames_per_chunk=FRAMES_PER_CHUNK):
    """Historical Stage 6A scheduling entry point."""
    return ordered_samples(sources, stream_mode='round_robin', frames_per_chunk=frames_per_chunk)


def run_streaming_moco(
    moco, processor, sources, device, max_steps=10, *, protocol=STAGE6A_PROTOCOL,
    optimizer_config=None, frames_per_chunk=FRAMES_PER_CHUNK,
):
    """Start fresh, warm up each active stream, then perform max_steps updates.

    EOF before the target raises rather than reporting a successful canary.
    Models, queues and optimizers are never resumed across calls.
    """
    optimizer_config = canonical_moco_config().optimizer if optimizer_config is None else optimizer_config
    validate_sources(
        sources, protocol.stream_mode, round_robin_stream_count=protocol.round_robin_stream_count,
    )
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    audit_initial_state(moco)
    groups = parameter_groups(moco)
    audit_finite(moco)
    if any(p.grad is not None for p in moco.parameters()):
        raise RuntimeError('Fresh model must have no gradients')
    before = cpu_snapshot(dict(moco.named_parameters()))
    expected_queue = deque(maxlen=moco.queue.capacity)
    completed = 0
    last_sample = None
    try:
        with ordered_samples(
            sources, protocol.stream_mode, frames_per_chunk=frames_per_chunk,
            round_robin_stream_count=protocol.round_robin_stream_count,
        ) as samples:
            def take_sample():
                nonlocal last_sample
                sample = next(samples, None)
                if sample is None:
                    last = None if last_sample is None else (last_sample.sequence_id, last_sample.sequence_index)
                    raise RuntimeError(f'EOF after {completed}/{max_steps} training steps; last sample: {last}')
                last_sample = sample
                return sample

            for _ in range(protocol.warmup_count):
                sample = take_sample()
                query_view, key_view = make_two_views(sample, protocol.key_transform)
                audit_views(sample, query_view, key_view, protocol.key_transform)
                key = encode_key_view(key_view, processor, moco, device)[-1]
                enqueue_key(moco, key, sample, expected_queue)
                report_json('warmup', sequence_id=sample.sequence_id, sequence_index=sample.sequence_index,
                            valid_frame_count=int(sample.valid_mask.sum().item()), queue_count=len(moco.queue))
            if len(moco.queue) != protocol.warmup_count:
                raise RuntimeError('Expected one warm-up key per active stream')
            if count_changed(dict(moco.named_parameters()), before) or any(
                p.grad is not None for p in moco.parameters()
            ):
                raise RuntimeError('Warm-up must not change model parameters or create gradients')
            report_json('warmup_complete', queue_count=len(moco.queue))
            optimizer = torch.optim.AdamW(
                moco.query_parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay,
            )
            optimized = [p for group in optimizer.param_groups for p in group['params']]
            trainable = {**groups['query LoRA'], **groups['query Projector']}
            if len(optimized) != len(trainable) or {id(p) for p in optimized} != {id(p) for p in trainable.values()}:
                raise RuntimeError('Optimizer must contain exactly Query LoRA + Query Projector')
            report_json('optimizer', parameters=sum(p.numel() for p in optimized), tensors=len(optimized),
                        matches_query_lora_and_projector=True, lr=optimizer_config.lr,
                        weight_decay=optimizer_config.weight_decay)
            for step in range(max_steps):
                sample = take_sample()
                train_step(moco, processor, sample, device, optimizer, groups, before, expected_queue, step, protocol)
                completed += 1
        audit_bases(groups, before)
        audit_finite(moco)
        audit_queue(moco, expected_queue)
        changes = {label: count_changed(parameters, before) for label, parameters in groups.items()}
        if any(changes[label] == 0 for label in ('query LoRA', 'query Projector', 'key LoRA', 'key Projector')):
            raise RuntimeError('Expected Query updates and Key EMA changes from the initial state')
        result = dict(training_steps=completed, max_steps=max_steps, queue_count=len(moco.queue),
                      changed_tensors=changes, all_parameters_finite=True, all_queue_keys_finite=True)
        report_json('complete', **result)
        return result
    except Exception as error:
        report_json('stopped', training_steps=completed, max_steps=max_steps, reason=str(error))
        raise


def run_canary(moco, processor, sources, device, max_steps=10):
    """Stage 6A compatibility entry point, retaining its no-eviction step bound."""
    validate_sources(sources)
    if type(max_steps) is not int or not 1 <= max_steps <= moco.queue.capacity - STREAM_COUNT:
        raise ValueError('max_steps must be a positive integer fitting the queue without eviction')
    return run_streaming_moco(moco, processor, sources, device, max_steps, protocol=STAGE6A_PROTOCOL)
