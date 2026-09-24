"""Shared model and view audits used by sequential MoCo smoke and canary runs."""

import torch
from peft.tuners.lora import LoraLayer

from self_supervised.moco.vit_lora_moco import lora_parameters


def audit_encoder_parameters(encoder):
    """Require exactly the approved Q/V adapters before creating an optimizer."""
    targets = [name for name, module in encoder.vit.named_modules() if isinstance(module, LoraLayer)]
    q_count = sum(name.endswith('.q_proj') for name in targets)
    v_count = sum(name.endswith('.v_proj') for name in targets)
    pooler_none = encoder.vit.get_base_model().pooler is None
    print(f"pooler is None: {pooler_none}")
    print(f"LoRA target names: {targets}")
    print(f"LoRA target count: {len(targets)}")
    print(f"q_proj count: {q_count}")
    print(f"v_proj count: {v_count}")
    if not pooler_none or len(targets) != 24 or (q_count, v_count) != (12, 12):
        raise RuntimeError("Expected no pooler and exactly 12 Q / 12 V LoRA targets")

    parameters = dict(encoder.named_parameters())
    lora = {name: p for name, p in parameters.items() if '.lora_A.' in name or '.lora_B.' in name}
    base = {name: p for name, p in parameters.items() if name not in lora}
    trainable = {name: p for name, p in parameters.items() if p.requires_grad}
    unexpected = sorted(set(trainable) - set(lora))
    count = sum(p.numel() for p in trainable.values())
    print(f"total model parameters: {sum(p.numel() for p in parameters.values())}")
    print(f"trainable parameters: {count}")
    print(f"trainable tensor count: {len(trainable)}")
    print(f"unexpected trainable names: {unexpected}")
    if unexpected or set(trainable) != set(lora) or count != 294_912 or len(trainable) != 48:
        raise RuntimeError("Expected exactly 294,912 trainable parameters in 48 LoRA tensors")
    return base, lora


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
    audit_encoder_parameters(moco.query_encoder)
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
