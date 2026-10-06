"""Shared model and view audits used by sequential MoCo smoke and canary runs."""

import torch
from peft.tuners.lora import LoraLayer

from self_supervised.moco.vit_lora_moco import lora_parameters


def audit_encoder_parameters(encoder):
    """Require exactly the approved Q/V adapters before creating an optimizer."""
    config = encoder.vit.peft_config['default']
    target_modules = set(config.target_modules)
    layer_count = getattr(encoder.vit.get_base_model().config, 'num_hidden_layers', None)
    if type(layer_count) is not int or layer_count <= 0:
        raise RuntimeError('Encoder must report a positive transformer layer count')
    targets = [name for name, module in encoder.vit.named_modules() if isinstance(module, LoraLayer)]
    target_counts = {target: sum(name.endswith(f'.{target}') for name in targets) for target in target_modules}
    pooler_none = encoder.vit.get_base_model().pooler is None
    print(f"pooler is None: {pooler_none}")
    print(f"LoRA target names: {targets}")
    print(f"LoRA target count: {len(targets)}")
    print(f"LoRA target counts: {target_counts}")
    if not pooler_none or len(targets) != layer_count * len(target_modules) or any(
        count != layer_count for count in target_counts.values()
    ):
        raise RuntimeError('Expected no pooler and one configured LoRA target per transformer layer')

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
    expected_tensors = layer_count * len(target_modules) * 2
    expected_parameters = layer_count * len(target_modules) * 2 * config.r * encoder.feature_size
    if unexpected or set(trainable) != set(lora) or count != expected_parameters or len(trainable) != expected_tensors:
        raise RuntimeError(
            f'Expected exactly {expected_parameters:,} trainable parameters in {expected_tensors} LoRA tensors'
        )
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
        query_config = moco.query_encoder.vit.peft_config['default']
        if (
            set(config.target_modules), config.r, config.lora_alpha, config.lora_dropout, config.bias
        ) != (
            set(query_config.target_modules), query_config.r, query_config.lora_alpha,
            query_config.lora_dropout, query_config.bias,
        ):
            raise RuntimeError(f'{side} LoRA configuration differs from Query')
        projector = getattr(moco, f'{side}_projector')
        if len(projector) != 3 or not isinstance(projector[0], torch.nn.Linear) or not isinstance(
            projector[1], torch.nn.ReLU
        ) or not isinstance(projector[2], torch.nn.Linear):
            raise RuntimeError('Expected Linear / ReLU / Linear projector')
        if (projector[0].in_features, projector[0].out_features,
            projector[2].in_features, projector[2].out_features) != (
                moco.feature_size, moco.feature_size, moco.feature_size, moco.projection_size,
        ):
            raise RuntimeError(
                f'Expected {moco.feature_size} -> {moco.feature_size} -> {moco.projection_size} projector'
            )
    groups = parameter_groups(moco)
    for side in ('query', 'key'):
        expected_counts = {
            'LoRA': (
                sum(p.numel() for p in groups['query LoRA'].values()), len(groups['query LoRA'])
            ),
            'Projector': (
                moco.feature_size * moco.feature_size + moco.feature_size
                + moco.feature_size * moco.projection_size + moco.projection_size,
                4,
            ),
        }
        for kind, (count, tensors) in expected_counts.items():
            parameters = groups[f'{side} {kind}']
            actual = sum(p.numel() for p in parameters.values())
            print(f'{side} {kind} params / tensors: {actual} / {len(parameters)}')
            if (actual, len(parameters)) != (count, tensors):
                raise RuntimeError(f'Unexpected {side} {kind} parameter count')
    expected = {**groups['query LoRA'], **groups['query Projector']}
    trainable = {name: p for name, p in moco.named_parameters() if p.requires_grad}
    if set(trainable) != set(expected):
        raise RuntimeError('Only Query LoRA and Query Projector may be trainable')
    if moco.queue.projection_size != moco.projection_size:
        raise RuntimeError('Queue projection size differs from the MoCo projector')
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


def audit_views(sample, query, key, key_transform='horizontal_flip'):
    if key_transform not in ('horizontal_flip', 'gbr_horizontal_flip'):
        raise ValueError(f'Unknown key_transform: {key_transform}')
    for view in (query, key):
        for name in ('sequence_id', 'sequence_index', 'source_id', 'is_first', 'is_last', 'sequence_length'):
            if getattr(view, name) != getattr(sample, name):
                raise RuntimeError(f'Two-view {name} differs')
        for name in ('source_metadata', 'dataset_metadata', 'evaluation_reference'):
            if getattr(view, name) is not getattr(sample, name):
                raise RuntimeError(f'Two-view {name} differs')
        for name in ('frame_indices', 'valid_mask', 'timestamps'):
            if not torch.allclose(getattr(view, name), getattr(sample, name), rtol=0, atol=0, equal_nan=True):
                raise RuntimeError(f'Two-view {name} differs')
    expected = sample.frames[sample.valid_mask]
    if key_transform == 'gbr_horizontal_flip':
        expected = expected[:, [1, 2, 0]]
    if not torch.equal(query.frames, sample.frames) or not torch.equal(
        key.frames[sample.valid_mask], expected.flip(-1)
    ) or not torch.equal(key.frames[~sample.valid_mask], sample.frames[~sample.valid_mask]):
        raise RuntimeError(f'Expected raw Query and valid-frame {key_transform} Key')
