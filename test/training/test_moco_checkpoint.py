"""Resume payload serialization and PEFT-native Query LoRA snapshots."""

import json

import pytest
import torch
from torch.nn import functional as F

from self_supervised.moco import ViTLoRAMoCo
from self_supervised.moco.vit_lora_moco import lora_parameters
from test_moco_multistep_canary import SmallEncoder
from test_vit_lora_frame_encoder import encoder  # noqa: F401 - fixture
import training.moco_checkpoint as checkpoint


def key(seed):
    return F.normalize(torch.randn(128, generator=torch.Generator().manual_seed(seed)), dim=0)


def test_queue_order_and_metadata_round_trip(tmp_path):
    torch.manual_seed(0)
    model = ViTLoRAMoCo(SmallEncoder())
    for index, video in enumerate('aabbc'):
        model.queue.enqueue(key(index), video, index)
    optimizer = torch.optim.AdamW(model.query_parameters(), lr=1e-3, weight_decay=0.)
    identity = {'run_id': 'r'}
    counters = {'processed_videos': 1, 'next_video_index': 1, 'global_update_step': 4,
                'next_source_id': 'b', 'final': False}
    checkpoint.save_resume_checkpoint(tmp_path / 'latest.pt', identity, counters,
                                      checkpoint.training_state(model, optimizer))
    assert not list(tmp_path.glob('.*'))
    payload = checkpoint.load_resume_checkpoint(tmp_path / 'latest.pt')
    assert payload['identity'] == identity and payload['counters'] == counters
    torch.manual_seed(1)
    restored = ViTLoRAMoCo(SmallEncoder())
    checkpoint.restore_training_state(restored, torch.optim.AdamW(restored.query_parameters(), lr=1e-3,
                                                                  weight_decay=0.), payload['state'], 'cpu')
    assert [(e.sequence_id, e.sequence_index) for e in restored.queue.entries] == [
        ('a', 0), ('a', 1), ('b', 2), ('b', 3), ('c', 4)]
    assert all(torch.equal(a.key, b.key) for a, b in zip(restored.queue.entries, model.queue.entries))
    for side in ('query', 'key'):
        for name, p in lora_parameters(getattr(model, f'{side}_encoder')).items():
            assert torch.equal(lora_parameters(getattr(restored, f'{side}_encoder'))[name], p)


def test_rng_state_round_trip():
    state = checkpoint.rng_state()
    expected = (torch.rand(3), __import__('numpy').random.rand(2).tolist(), __import__('random').random())
    checkpoint.restore_rng_state(state)
    assert torch.equal(torch.rand(3), expected[0])
    assert __import__('numpy').random.rand(2).tolist() == expected[1]
    assert __import__('random').random() == expected[2]


def test_identity_mismatch_and_unsupported_schema_are_rejected(tmp_path):
    with pytest.raises(RuntimeError, match='queue_capacity'):
        checkpoint.validate_resume_identity({'queue_capacity': 4096}, {'queue_capacity': 1024})
    torch.save({'format': 'other'}, tmp_path / 'latest.pt')
    with pytest.raises(RuntimeError, match='schema'):
        checkpoint.load_resume_checkpoint(tmp_path / 'latest.pt')
    (tmp_path / 'broken.pt').write_bytes(b'not a checkpoint')
    with pytest.raises(RuntimeError, match='Corrupted'):
        checkpoint.load_resume_checkpoint(tmp_path / 'broken.pt')


def snapshot_metadata(model, final=False):
    return {'processed_videos': 1000, 'global_update_step': 12, 'final': final, 'run_id': 'r',
            'lora_config': checkpoint.lora_config(model)}


def test_snapshot_is_query_lora_only_and_reloads_exactly(encoder, tmp_path, monkeypatch):  # noqa: F811
    with torch.no_grad():
        for p in lora_parameters(encoder).values():
            p.normal_()
    target, metadata = checkpoint.write_evaluation_snapshot(tmp_path, encoder, snapshot_metadata(encoder))
    assert target.name == 'videos-001000_step-12'
    assert sorted(path.name for path in target.iterdir()) == [
        'adapter_config.json', 'adapter_model.safetensors', 'metadata.json']
    assert metadata['schema'] == checkpoint.SNAPSHOT_SCHEMA and set(metadata['files']) == set(checkpoint.SNAPSHOT_FILES)
    from safetensors.torch import load_file
    assert len(load_file(target / 'adapter_model.safetensors')) == 48

    from transformers import ViTConfig, ViTModel
    from model.backbones.vit import ViTLoRAFrameEncoder
    fresh_backbone = ViTModel(ViTConfig(intermediate_size=32), add_pooling_layer=False)
    fresh_backbone.load_state_dict({
        name.replace('.base_layer.', '.'): tensor for name, tensor in encoder.vit.get_base_model().state_dict().items()
        if '.lora_' not in name
    }, strict=True)
    monkeypatch.setattr('model.backbones.vit.vit_lora_frame_encoder.ViTModel.from_pretrained',
                        lambda *args, **kwargs: fresh_backbone)
    fresh = ViTLoRAFrameEncoder()
    checkpoint.load_query_lora_snapshot(fresh, target)
    for name, p in lora_parameters(encoder).items():
        assert torch.equal(lora_parameters(fresh)[name], p)
    pixels = torch.randn(2, 3, 224, 224)
    torch.testing.assert_close(fresh.eval()(pixels), encoder.eval()(pixels), rtol=0, atol=0)

    # PEFT itself can load the published directory.
    from peft import PeftModel
    PeftModel.from_pretrained(ViTModel(ViTConfig(intermediate_size=32), add_pooling_layer=False), target)

    # Same content is reused; different content with the same name is refused.
    assert checkpoint.write_evaluation_snapshot(tmp_path, encoder, snapshot_metadata(encoder))[0] == target
    with torch.no_grad():
        next(iter(lora_parameters(encoder).values())).add_(1)
    with pytest.raises(RuntimeError, match='Different evaluation snapshot'):
        checkpoint.write_evaluation_snapshot(tmp_path, encoder, snapshot_metadata(encoder))


def test_snapshot_load_rejects_tampering_and_config_mismatch(encoder, tmp_path):  # noqa: F811
    target, _ = checkpoint.write_evaluation_snapshot(tmp_path, encoder, snapshot_metadata(encoder, final=True))
    assert target.name.endswith('_final')
    data = bytearray((target / 'adapter_model.safetensors').read_bytes())
    data[-1] ^= 1
    (target / 'adapter_model.safetensors').write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match='hashes'):
        checkpoint.load_query_lora_snapshot(encoder, target)
    target, _ = checkpoint.write_evaluation_snapshot(tmp_path / 'other', encoder, snapshot_metadata(encoder))
    config = json.loads((target / 'adapter_config.json').read_text())
    metadata = json.loads((target / 'metadata.json').read_text())
    metadata['lora_config']['r'] = 4
    (target / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match='config'):
        checkpoint.load_query_lora_snapshot(encoder, target)
    metadata['schema'] = 'unknown/v0'
    (target / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match='schema'):
        checkpoint.load_query_lora_snapshot(encoder, target)
    assert config['r'] == 8


def test_real_provenance_survives_weights_only_checkpoint(tmp_path):
    from utils.provenance import collect_provenance
    provenance = collect_provenance()
    assert all(type(version) is str for version in provenance['versions'].values())
    torch.manual_seed(0)
    model = ViTLoRAMoCo(SmallEncoder())
    optimizer = torch.optim.AdamW(model.query_parameters(), lr=1e-3, weight_decay=0.)
    counters = {'processed_videos': 1, 'next_video_index': 1, 'global_update_step': 0,
                'next_source_id': None, 'final': True}
    identity = {'dependencies': {'versions': provenance['versions'],
                                 'sequential_loader': provenance['sequential_loader']}}
    checkpoint.save_resume_checkpoint(tmp_path / 'latest.pt', identity, counters,
                                      checkpoint.training_state(model, optimizer))
    assert checkpoint.load_resume_checkpoint(tmp_path / 'latest.pt')['identity'] == identity
