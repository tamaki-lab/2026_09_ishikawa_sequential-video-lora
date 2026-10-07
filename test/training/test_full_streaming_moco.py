"""Full-dataset single-pass Streaming MoCo: boundaries, triggers and exact resume on CPU."""

from contextlib import contextmanager
import json
from pathlib import Path

import pytest
import sequential_loader as sl
import torch

from self_supervised.moco import ViTLoRAMoCo
from test_activitynet_vit_clip_feature import RecordingProcessor
from test_moco_multistep_canary import SmallEncoder
import training.moco_checkpoint as checkpoint
import training.streaming_moco_full as full


PROVENANCE = {
    'implementation': {'repository': 'repo', 'branch': 'dev', 'commit': 'abc', 'dirty': False,
                       'tracked_diff_sha256': 'clean'},
    'sequential_loader': {'branch': 'ActivityNet', 'commit': 'def'},
    'versions': {'python': 'test'}, 'base_model': 'base',
}
CONFIG = {'dataset': {'name': 'synthetic', 'split': 'training'}, 'seed': 0,
          'device_identity': {'type': 'cpu'}}


def make_sources(lengths):
    return tuple(sl.SequenceSource(
        sequence_id=video, source_id=video, source=Path(f'{video}.mp4'),
        start_frame=0, stop_frame=None, evaluation_reference=object(),
    ) for video in lengths)


@pytest.fixture
def videos(monkeypatch):
    """Videos with a configurable chunk count; the terminal chunk has 3 frames."""
    state = {'lengths': {}, 'reads': [], 'opened': [], 'closed': [], 'failure': None}

    class Reader:
        @contextmanager
        def open(self, source):
            self.source = source
            state['opened'].append(source.sequence_id)
            try:
                yield self
            finally:
                state['closed'].append(source.sequence_id)

        def read(self, chunk):
            video = self.source.sequence_id
            state['reads'].append((video, chunk.chunk_index))
            if state['failure'] == (video, chunk.chunk_index):
                raise RuntimeError('decode failure')
            terminal = chunk.chunk_index == state['lengths'][video] - 1
            count = 3 if terminal else 16
            seed = sum(map(ord, video)) * 7 + chunk.chunk_index * 3
            frames = ((torch.arange(count * 12).reshape(count, 3, 2, 2) * 5 + seed) % 251).to(torch.uint8)
            return sl.DecodedChunk(
                frames=frames, frame_indices=torch.arange(chunk.start_frame, chunk.start_frame + count),
                timestamps=torch.arange(count, dtype=torch.float64), reached_eof=terminal,
            )

    monkeypatch.setattr(sl, 'SequentialVideoReader', Reader)
    return state


@pytest.fixture
def small(monkeypatch):
    """SmallEncoder MoCo; only the ViT/PEFT-specific audits and writers are substituted."""
    def initial(model):
        assert len(model.queue) == 0
        for a, b in zip(model.query_encoder.parameters(), model.key_encoder.parameters()):
            assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()

    snapshots, snapshot_metadata = [], []

    def snapshot(root, encoder, metadata):
        name = checkpoint.snapshot_name(metadata['processed_videos'], metadata['global_update_step'],
                                        metadata['final'])
        target = Path(root) / name
        target.mkdir(parents=True)
        lora = {k: v.clone() for k, v in checkpoint.lora_parameters(encoder).items()}
        snapshots.append((name, lora))
        snapshot_metadata.append(metadata)
        (target / 'metadata.json').write_text(json.dumps(metadata))
        return target, {**metadata, 'files': {f: 'x' for f in checkpoint.SNAPSHOT_FILES}}

    monkeypatch.setattr(full, 'audit_initial_state', initial)
    monkeypatch.setattr(full, 'audit_parameters', lambda model: None)
    monkeypatch.setattr(full, 'encoder_base_fingerprint', lambda encoder: 'base-fingerprint')
    monkeypatch.setattr(full, 'lora_config', lambda encoder: {'r': 8})
    monkeypatch.setattr(full, 'write_evaluation_snapshot', snapshot)
    monkeypatch.setattr(full, 'log_artifact', lambda *args, **kwargs: {'status': 'disabled'})

    def build(seed=0):
        torch.manual_seed(seed)
        return ViTLoRAMoCo(SmallEncoder()).train()
    build.snapshots = snapshots
    build.snapshot_metadata = snapshot_metadata
    return build


def run(model, sources, run_dir, **kwargs):
    options = dict(run_id='r', provenance=PROVENANCE, run_config=CONFIG, expected_source_count=len(sources))
    options.update(kwargs)
    return full.run_full_streaming_moco(model, RecordingProcessor(), sources, torch.device('cpu'), run_dir, **options)


def events(run_dir, name=None):
    rows = [json.loads(line) for line in (Path(run_dir) / 'audit.jsonl').read_text().splitlines()]
    return [row for row in rows if name is None or row['event'] == name]


def full_state(model):
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


def test_single_warmup_and_state_kept_across_video_boundaries(small, videos, tmp_path, monkeypatch):
    videos['lengths'] = {'a': 3, 'b': 2, 'c': 1}
    model = small()
    factory, optimizers = torch.optim.AdamW, []
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **k: optimizers.append(factory(*a, **k)) or optimizers[-1])
    result = run(model, make_sources(videos['lengths']), tmp_path / 'run')
    warmups = events(tmp_path / 'run', 'warmup')
    steps = events(tmp_path / 'run', 'step')
    assert [(row['sequence_id'], row['sequence_index']) for row in warmups] == [('a', 0)]
    assert [(row['sequence_id'], row['sequence_index'], row['step']) for row in steps] == [
        ('a', 1, 0), ('a', 2, 1), ('b', 0, 2), ('b', 1, 3), ('c', 0, 4)]
    # Negatives for B1 are the old queue including all of video A: no reset at the boundary.
    assert steps[2]['valid_negative_count'] == 3 and steps[4]['valid_negative_count'] == 5
    assert len(optimizers) == 1 and all(int(s['step']) == 5 for s in optimizers[0].state.values())
    assert [(e.sequence_id, e.sequence_index) for e in model.queue.entries] == [
        ('a', 0), ('a', 1), ('a', 2), ('b', 0), ('b', 1), ('c', 0)]
    assert videos['opened'] == videos['closed'] == ['a', 'b', 'c']
    assert [row['processed_videos'] for row in events(tmp_path / 'run', 'video_complete')] == [1, 2, 3]
    assert result == {'status': 'complete', 'processed_videos': 3, 'next_video_index': 3,
                      'global_update_step': 5, 'next_source_id': None, 'final': True, 'queue_count': 6}


def test_one_chunk_first_video_is_only_warmup(small, videos, tmp_path):
    videos['lengths'] = {'a': 1, 'b': 2}
    run(small(), make_sources(videos['lengths']), tmp_path / 'run')
    steps = events(tmp_path / 'run', 'step')
    assert [(row['sequence_id'], row['sequence_index']) for row in events(tmp_path / 'run', 'warmup')] == [('a', 0)]
    assert [(row['sequence_id'], row['sequence_index'], row['step']) for row in steps] == [('b', 0, 0), ('b', 1, 1)]


def test_resume_and_snapshot_triggers(small, videos, tmp_path):
    videos['lengths'] = {name: 2 for name in 'abcdefg'}
    run(small(), make_sources(videos['lengths']), tmp_path / 'run', resume_interval=2, snapshot_interval=3)
    saved = [row['processed_videos'] for row in events(tmp_path / 'run', 'resume_checkpoint')]
    snapshots = [(row['processed_videos'], row['final']) for row in events(tmp_path / 'run', 'evaluation_snapshot')]
    assert saved == [2, 4, 6, 7]
    assert snapshots == [(3, False), (6, False), (7, True)]
    assert [name for name, _ in small.snapshots] == [
        'videos-000003_step-5', 'videos-000006_step-11', 'videos-000007_step-13_final']
    final_metadata = small.snapshot_metadata[-1]
    assert final_metadata['seed'] == 0 and final_metadata['device_identity'] == {'type': 'cpu'}
    assert final_metadata['base_fingerprint'] and final_metadata['versions'] == {'python': 'test'}
    latest = checkpoint.load_resume_checkpoint(tmp_path / 'run' / 'resume' / 'latest.pt')
    assert latest['counters'] == {'processed_videos': 7, 'next_video_index': 7, 'global_update_step': 13,
                                  'next_source_id': None, 'final': True}


def test_metric_window_every_100_updates(small, videos, tmp_path, monkeypatch):
    videos['lengths'] = {'a': 4, 'b': 3}
    logged = []
    monkeypatch.setattr(full, 'log_metrics', lambda experiment, metrics, step: logged.append((step, metrics)))
    run(small(), make_sources(videos['lengths']), tmp_path / 'run', metric_interval=2)
    assert [step for step, _ in logged] == [2, 4, 6]
    assert set(logged[0][1]) == {
        'moco/loss', 'moco/positive_similarity', 'moco/valid_negative_count', 'moco/query_lora_grad_norm',
        'moco/processed_videos', 'moco/global_update_step', 'moco/queue_count', 'moco/queue_unique_sequence_count'}
    assert logged[1][1]['moco/processed_videos'] == 1


def test_video_boundary_resume_is_exact(small, videos, tmp_path):
    videos['lengths'] = {'a': 3, 'b': 1, 'c': 2, 'd': 3}
    sources = make_sources(videos['lengths'])
    reference = small(seed=1)
    run(reference, sources, tmp_path / 'reference', run_id='x')
    reference_optimizer = checkpoint.load_resume_checkpoint(tmp_path / 'reference' / 'resume' / 'latest.pt')

    first = small(seed=1)
    snapshot_count = len(small.snapshots)
    paused = run(first, sources, tmp_path / 'resumed', run_id='x', stop_after_videos=2)
    assert paused['status'] == 'paused' and paused['processed_videos'] == 2 and paused['next_source_id'] == 'c'
    assert len(small.snapshots) == snapshot_count
    assert not events(tmp_path / 'resumed', 'evaluation_snapshot')
    del first
    # A newly constructed process state; perturbed trainable / EMA tensors must be overwritten.
    second = small(seed=1)
    with torch.no_grad():
        for name, parameter in second.named_parameters():
            if '.lora_' in name or 'projector' in name:
                parameter.add_(1.0)
    result = run(second, sources, tmp_path / 'resumed', run_id='x', resume=True)
    assert result['global_update_step'] == 8 and result['final']
    assert full_state(second).keys() == full_state(reference).keys()
    for name, tensor in full_state(reference).items():
        assert torch.equal(full_state(second)[name], tensor), name
    assert [(e.sequence_id, e.sequence_index) for e in second.queue.entries] == [
        (e.sequence_id, e.sequence_index) for e in reference.queue.entries]
    assert all(torch.equal(a.key, b.key) for a, b in zip(second.queue.entries, reference.queue.entries))
    resumed = checkpoint.load_resume_checkpoint(tmp_path / 'resumed' / 'resume' / 'latest.pt')
    assert resumed['counters'] == reference_optimizer['counters']
    for (name, a), (_, b) in zip(sorted(resumed['state']['optimizer']['state'].items()),
                                 sorted(reference_optimizer['state']['optimizer']['state'].items())):
        assert all(torch.equal(a[key], b[key]) for key in a), name
    resumed_steps = events(tmp_path / 'resumed', 'step')
    assert [(row['sequence_id'], row['sequence_index']) for row in resumed_steps] == [
        ('a', 1), ('a', 2), ('b', 0), ('c', 0), ('c', 1), ('d', 0), ('d', 1), ('d', 2)]
    assert len(events(tmp_path / 'resumed', 'warmup')) == 1


def test_stop_after_at_or_beyond_selected_count_finishes_at_source_exhaustion(
    small, videos, tmp_path,
):
    videos['lengths'] = {'a': 2, 'b': 2}
    selection_record = {
        'schema': 'activitynet-source-selection/v1',
        'created': '2026-10-08T00:00:00Z',
        'implementation': PROVENANCE['implementation'],
    }
    config = {
        **CONFIG,
        'source_selection': {'profile_id': 'test-reduced-v1'},
        'selection_sha256': 'a' * 64,
        'selection_record': selection_record,
    }
    result = run(
        small(), make_sources(videos['lengths']), tmp_path / 'run',
        stop_after_videos=99, run_config=config,
    )
    assert result['status'] == 'complete'
    assert result['processed_videos'] == 2 and result['next_video_index'] == 2
    assert result['next_source_id'] is None and result['final'] is True
    assert not events(tmp_path / 'run', 'paused')
    assert [(row['processed_videos'], row['final']) for row in events(
        tmp_path / 'run', 'evaluation_snapshot',
    )] == [(2, True)]
    assert small.snapshot_metadata[-1]['selection_record'] == selection_record


def test_resume_rejects_mismatch_corruption_mid_video_and_final(small, videos, tmp_path):
    videos['lengths'] = {'a': 2, 'b': 2, 'c': 2}
    sources = make_sources(videos['lengths'])
    with pytest.raises(FileNotFoundError):
        run(small(), sources, tmp_path / 'run', resume=True)
    run(small(), sources, tmp_path / 'run', stop_after_videos=1)
    with pytest.raises(FileExistsError):
        run(small(), sources, tmp_path / 'run')
    with pytest.raises(ValueError, match='Expected exactly'):
        run(small(), sources[:2], tmp_path / 'run', resume=True, expected_source_count=3)
    with pytest.raises(RuntimeError, match='does not match'):
        run(small(), sources[::-1], tmp_path / 'run', resume=True)
    with pytest.raises(RuntimeError, match='does not match'):
        run(small(), sources, tmp_path / 'run', resume=True, run_id='other')
    with pytest.raises(RuntimeError, match='does not match'):
        run(small(), sources, tmp_path / 'run', resume=True,
            run_config={**CONFIG, 'seed': 1})
    changed = {**PROVENANCE, 'implementation': {**PROVENANCE['implementation'],
                                                'tracked_diff_sha256': 'changed'}}
    with pytest.raises(RuntimeError, match='does not match'):
        run(small(), sources, tmp_path / 'run', resume=True, provenance=changed)
    with pytest.raises(RuntimeError, match='device identity'):
        run(small(), sources, tmp_path / 'run', resume=True,
            run_config={**CONFIG, 'device_identity': {'type': 'cuda'}})

    latest = tmp_path / 'run' / 'resume' / 'latest.pt'
    original = latest.read_bytes()
    payload = checkpoint.load_resume_checkpoint(latest)
    payload['counters']['next_video_index'] = 0  # Pretend a mid-video position.
    checkpoint_path = tmp_path / 'mid.pt'
    with pytest.raises(RuntimeError, match='video boundary'):
        checkpoint.save_resume_checkpoint(checkpoint_path, payload['identity'], payload['counters'], payload['state'])
    data = bytearray(original)
    data[len(data) // 2] ^= 0xFF
    latest.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match='Corrupted|hash|schema'):
        run(small(), sources, tmp_path / 'run', resume=True)
    latest.write_bytes(original)
    assert run(small(), sources, tmp_path / 'run', resume=True)['final']
    with pytest.raises(RuntimeError, match='already completed'):
        run(small(), sources, tmp_path / 'run', resume=True)


@pytest.mark.parametrize('field, value', [
    ('profile_id', 'other-profile'),
    ('strategy', 'other-strategy'),
    ('selection_seed', 9),
    ('selection_sha256', 'b' * 64),
])
def test_resume_rejects_source_selection_mismatch(small, videos, tmp_path, field, value):
    videos['lengths'] = {'a': 2, 'b': 2}
    sources = make_sources(videos['lengths'])
    source_selection = {
        'profile_id': 'test-reduced-v1', 'strategy': 'sha256_rank_preserve_adapter_order_v1',
        'selection_seed': 7,
    }
    config = {
        **CONFIG, 'selection_sha256': 'a' * 64, 'source_selection': source_selection,
    }
    run(small(), sources, tmp_path / 'run', stop_after_videos=1, run_config=config)
    changed = dict(config)
    if field == 'selection_sha256':
        changed[field] = value
    else:
        changed['source_selection'] = {**source_selection, field: value}
    with pytest.raises(RuntimeError, match='source_selection'):
        run(small(), sources, tmp_path / 'run', resume=True, run_config=changed)


def test_dirty_full_run_is_rejected_before_creating_output(small, videos, tmp_path):
    videos['lengths'] = {'a': 2}
    dirty = {**PROVENANCE, 'implementation': {**PROVENANCE['implementation'], 'dirty': True}}
    target = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='clean implementation'):
        run(small(), make_sources(videos['lengths']), target, provenance=dirty)
    assert not target.exists()


def test_decode_failure_stops_and_keeps_last_boundary_checkpoint(small, videos, tmp_path):
    videos['lengths'] = {'a': 2, 'b': 2, 'c': 3}
    videos['failure'] = ('c', 1)
    sources = make_sources(videos['lengths'])
    with pytest.raises(RuntimeError, match='sequential iteration failed'):
        run(small(), sources, tmp_path / 'run', resume_interval=1)
    assert checkpoint.load_resume_checkpoint(tmp_path / 'run' / 'resume' / 'latest.pt')['counters'][
        'processed_videos'] == 2
    assert events(tmp_path / 'run')[-1]['event'] == 'stopped'
    assert videos['opened'] == videos['closed']
    videos['failure'] = None
    assert run(small(), sources, tmp_path / 'run', resume=True)['processed_videos'] == 3


def test_missing_snapshot_is_written_on_resume(small, videos, tmp_path, monkeypatch):
    videos['lengths'] = {'a': 2, 'b': 2, 'c': 2}
    sources = make_sources(videos['lengths'])
    original = full.write_evaluation_snapshot

    def crash(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(full, 'write_evaluation_snapshot', crash)
    with pytest.raises(KeyboardInterrupt):
        run(small(), sources, tmp_path / 'run', resume_interval=2, snapshot_interval=2)
    monkeypatch.setattr(full, 'write_evaluation_snapshot', original)
    run(small(), sources, tmp_path / 'run', resume=True, resume_interval=2, snapshot_interval=2)
    assert [name for name, _ in small.snapshots] == ['videos-000002_step-3', 'videos-000003_step-5_final']
