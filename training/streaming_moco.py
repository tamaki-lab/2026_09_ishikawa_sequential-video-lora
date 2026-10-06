"""Shared, audited per-chunk Streaming MoCo update.

Both the bounded canary runs (training.moco_canary) and the production
single-pass run (training.streaming_moco_full) drive this engine.
"""

import json

import torch

from self_supervised.moco.vit_lora_moco import require_normalized
from self_supervised.moco.negative_selection import select_negatives
from integration.sequential_moco import make_two_views, encode_query_view, encode_key_view
from training.moco_audit import audit_views


def report_json(event, **values):
    print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)


def cpu_snapshot(parameters):
    return {name: p.detach().cpu().clone() for name, p in parameters.items()}


def count_changed(parameters, before):
    # Snapshots may live on CPU or, for long runs, on the parameter device.
    return sum(not torch.equal(p.detach().to(before[name].device), before[name]) for name, p in parameters.items())


def audit_bases(groups, before):
    for label in ('query base', 'key base'):
        if count_changed(groups[label], before):
            raise RuntimeError(f'{label} changed')


def audit_finite(moco):
    if not all(torch.isfinite(p).all().item() for p in moco.parameters()):
        raise RuntimeError('Model parameter contains NaN or Inf')


def audit_queue(moco, expected):
    entries = moco.queue.entries
    if len(entries) != len(expected):
        raise RuntimeError('Queue count does not match the retained FIFO history')
    if not entries:
        return
    if any(entry.key.requires_grad or entry.key.grad_fn is not None for entry in entries):
        raise RuntimeError('Queue keys must be detached')
    if [(entry.sequence_id, entry.sequence_index) for entry in entries] != [
        (sequence_id, sequence_index) for sequence_id, sequence_index, _ in expected
    ]:
        raise RuntimeError('Queue key / metadata alignment changed')
    # One batched check per call; same per-key contract as require_normalized.
    keys = torch.stack([entry.key for entry in entries])
    if tuple(keys.shape[1:]) != (moco.projection_size,) or not torch.isfinite(keys).all().item():
        raise RuntimeError(f'Expected a finite projected vector [{moco.projection_size}]')
    norms = keys.norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-6):
        raise RuntimeError('Expected an L2-normalized projected vector')
    if not torch.equal(keys, torch.stack([key for _, _, key in expected])):
        raise RuntimeError('Queue key / metadata alignment changed')


def enqueue_key(moco, key, sample, expected):
    require_normalized(key, moco.projection_size)
    if key.requires_grad or key.grad_fn is not None:
        raise RuntimeError('Current Key must be detached')
    # Independent reference checks FIFO history, metadata and the pre-EMA key.
    expected.append((sample.sequence_id, sample.sequence_index, key.detach().clone()))
    moco.queue.enqueue(key, sample.sequence_id, sample.sequence_index)
    audit_queue(moco, expected)


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


def train_step(moco, processor, sample, device, optimizer, groups, before, expected_queue, step, protocol,
               report=None):
    query_view, key_view = make_two_views(sample, protocol.key_transform)
    audit_views(sample, query_view, key_view, protocol.key_transform)
    optimizer.zero_grad(set_to_none=True)
    query = encode_query_view(query_view, processor, moco, device)[-1]
    key = encode_key_view(key_view, processor, moco, device)[-1]
    require_normalized(key, moco.projection_size)
    if key.requires_grad:
        raise RuntimeError('Current Key must be detached')
    audit_queue(moco, expected_queue)
    old_entries = moco.queue.entries
    expected_negatives = select_negatives(old_entries, sample.sequence_id, protocol.negative_policy)
    loss, logits, negatives = moco.contrastive_loss(
        query, key, sample.sequence_id, negatives=expected_negatives,
    )
    if len(negatives) != len(expected_negatives) or any(
        actual is not original for actual, original in zip(negatives, expected_negatives)
    ):
        raise RuntimeError('Negatives must exactly match the selected entries in the old queue')
    if loss.ndim != 0 or tuple(logits.shape) != (1, 1 + len(negatives)) or not (
        torch.isfinite(loss).item() and torch.isfinite(logits).all().item()
    ):
        raise RuntimeError('Expected finite scalar InfoNCE loss and logits')
    key_parameters = {**groups['key LoRA'], **groups['key Projector']}
    key_before = cpu_snapshot(key_parameters)
    loss.backward()
    gradients = _gradient_diagnostics(groups)
    optimizer.step()
    if count_changed(key_parameters, key_before):
        raise RuntimeError('Key changed before EMA')
    audit_bases(groups, before)
    moco.update_key()
    query_parameters = {**groups['query LoRA'], **groups['query Projector']}
    for name, p in key_parameters.items():
        query_name = name.replace('key_', 'query_', 1)
        query_p = query_parameters[query_name]
        expected = key_before[name] * moco.momentum + query_p.detach().cpu() * (1. - moco.momentum)
        if not torch.allclose(p.detach().cpu(), expected, rtol=1e-6, atol=1e-8):
            raise RuntimeError('Key EMA does not match the Stage 5 formula')
    audit_bases(groups, before)
    audit_finite(moco)
    audit_queue(moco, expected_queue)
    enqueue_key(moco, key, sample, expected_queue)
    (report or report_json)(
        'step', step=step, sequence_id=sample.sequence_id, sequence_index=sample.sequence_index,
        valid_frame_count=int(sample.valid_mask.sum().item()), loss=loss.item(),
        positive_similarity=(query.detach() @ key).item(),
        negative_logits_min=logits[0, 1:].min().item(), negative_logits_max=logits[0, 1:].max().item(),
        valid_negative_count=len(negatives), queue_count=len(moco.queue),
        queue_unique_sequence_id_count=len({entry.sequence_id for entry in moco.queue.entries}),
        gradients=gradients, all_parameters_finite=True, all_queue_keys_finite=True,
    )
