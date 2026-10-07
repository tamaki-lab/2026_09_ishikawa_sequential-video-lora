"""Production full-dataset single-pass Streaming MoCo with video-boundary resume.

One active video at a time, in the given source order. Only the first chunk of
the first video is a Key-only warm-up; Query / Key / optimizer / EMA / FIFO
Queue and the global step persist across video boundaries. Every source is
consumed to EOF before `processed_videos` increases. Resume checkpoints are
written only at completed video boundaries; evaluation snapshots keep only the
Query LoRA. The per-chunk update and audits reuse training.streaming_moco.
"""

from collections import deque
import json
from pathlib import Path

import torch

from integration.sequential_moco import encode_key_view, make_two_views
from integration.sequential_stream import ordered_samples
from logger.comet_lineage import log_artifact, log_metrics
from training.moco_audit import audit_initial_state, audit_parameters, audit_views, parameter_groups
from training.moco_checkpoint import (
    PROTOCOL_VERSION, SNAPSHOT_FILES, base_fingerprint, encoder_base_fingerprint, load_resume_checkpoint,
    lora_config, optimizer_config, ordered_source_sha256, restore_rng_state, restore_training_state,
    save_resume_checkpoint, snapshot_name, training_state, validate_resume_identity, write_evaluation_snapshot,
)
from training.moco_protocol import STAGE6B_PROTOCOL
from training.moco_config import canonical_activitynet_config, canonical_moco_config, canonical_sequential_config
from training.streaming_moco import (
    audit_bases, audit_finite, audit_queue, count_changed, enqueue_key, train_step,
)
from utils.artifact_io import read_json, write_json_atomic
from utils.configuration import load_config_group
from utils.provenance import device_identity, utc_now


_ACTIVITYNET_CONFIG = canonical_activitynet_config()
_MOCO_CONFIG = canonical_moco_config()
_SEQUENTIAL_CONFIG = canonical_sequential_config()
_TRACKING_CONFIG = load_config_group('tracking', 'default')
ACTIVITYNET_TRAINING_COUNT = _ACTIVITYNET_CONFIG.expected_source_counts['training']
RESUME_INTERVAL = _MOCO_CONFIG.intervals.resume_videos
SNAPSHOT_INTERVAL = _MOCO_CONFIG.intervals.snapshot_videos
METRIC_INTERVAL = _MOCO_CONFIG.intervals.comet_metric_updates
LEARNING_RATE = _MOCO_CONFIG.optimizer.lr
WEIGHT_DECAY = _MOCO_CONFIG.optimizer.weight_decay
SNAPSHOT_ARTIFACT = _TRACKING_CONFIG['artifacts']['moco_snapshot']


class RunPaths:
    def __init__(self, root):
        self.root = Path(root)
        self.latest = self.root / 'resume' / 'latest.pt'
        self.snapshots = self.root / 'evaluation_snapshots'
        self.audit_log = self.root / 'audit.jsonl'
        self.metadata = self.root / 'run_metadata.json'
        self.sources = self.root / 'ordered_sources.json'


class _Recorder:
    """Full-granularity local audit log plus 100-update Comet aggregation."""

    def __init__(self, path, experiment, interval):
        self.file = Path(path).open('a', encoding='utf-8')
        self.experiment = experiment
        self.interval = interval
        self.window = []
        self.context = {}

    def event(self, event, **values):
        line = json.dumps({'event': event, 'time': utc_now(), **values}, allow_nan=False)
        self.file.write(line + '\n')
        if event != 'step':
            print(line, flush=True)

    def step(self, event, **values):
        self.event(event, **values, **self.context)
        self.window.append(values)

    def flush_metrics(self, global_update_step, processed_videos):
        if global_update_step % self.interval or not self.window:
            return
        rows, self.window = self.window, []
        mean = lambda name: sum(row[name] for row in rows) / len(rows)  # noqa: E731
        last = rows[-1]
        metrics = {
            'moco/loss': mean('loss'),
            'moco/positive_similarity': mean('positive_similarity'),
            'moco/valid_negative_count': last['valid_negative_count'],
            'moco/query_lora_grad_norm': mean_grad(rows),
            'moco/processed_videos': processed_videos,
            'moco/global_update_step': global_update_step,
            'moco/queue_count': last['queue_count'],
            'moco/queue_unique_sequence_count': last['queue_unique_sequence_id_count'],
        }
        log_metrics(self.experiment, metrics, step=global_update_step)
        print(json.dumps({'event': 'metrics', 'window': len(rows), **metrics}), flush=True)
        self.file.flush()

    def close(self):
        self.file.close()


def mean_grad(rows):
    return sum(row['gradients']['query LoRA']['norm'] for row in rows) / len(rows)


def run_identity(
    moco, optimizer, run_id, source_ids, provenance, base_fingerprints, run_config, runtime_device, *,
    protocol=STAGE6B_PROTOCOL, protocol_version=PROTOCOL_VERSION,
):
    implementation = provenance['implementation']
    return {
        'run_id': run_id, 'protocol_version': protocol_version,
        'protocol': {'stream_mode': protocol.stream_mode, 'key_transform': protocol.key_transform,
                     'negative_policy': protocol.negative_policy},
        'repository': implementation['repository'], 'branch': implementation['branch'],
        'commit': implementation['commit'], 'dirty': implementation['dirty'],
        'tracked_diff_sha256': implementation['tracked_diff_sha256'],
        'dependencies': {'versions': provenance.get('versions'),
                         'sequential_loader': provenance.get('sequential_loader')},
        'dataset': run_config['dataset'],
        'stream_config': {
            'frames_per_chunk': run_config.get('frames_per_chunk', _SEQUENTIAL_CONFIG.frames_per_chunk),
            'round_robin_stream_count': run_config.get(
                'round_robin_stream_count', _SEQUENTIAL_CONFIG.round_robin_stream_count,
            ),
        },
        'base_model': provenance['base_model'], 'base_fingerprints': base_fingerprints,
        'lora_config': lora_config(moco.query_encoder), 'feature_size': moco.feature_size,
        'projection_size': moco.projection_size, 'queue_capacity': moco.queue.capacity,
        'momentum': moco.momentum, 'temperature': moco.temperature,
        'optimizer_config': optimizer_config(optimizer), 'seed': run_config['seed'],
        'device_identity': runtime_device,
        'source_count': len(source_ids), 'ordered_source_sha256': ordered_source_sha256(source_ids),
        'source_selection': {
            'selection_sha256': run_config.get('selection_sha256'),
            'profile_id': (run_config.get('source_selection') or {}).get('profile_id'),
            'strategy': (run_config.get('source_selection') or {}).get('strategy'),
            'selection_seed': (run_config.get('source_selection') or {}).get('selection_seed'),
            'selected_source_count': len(source_ids),
            'selected_ordered_source_sha256': ordered_source_sha256(source_ids),
        },
    }


def _new_optimizer(moco, groups, optimizer_settings=None):
    optimizer_settings = _MOCO_CONFIG.optimizer if optimizer_settings is None else optimizer_settings
    if optimizer_settings.name != 'AdamW':
        raise ValueError(f'Unsupported MoCo optimizer: {optimizer_settings.name}')
    optimizer = torch.optim.AdamW(
        moco.query_parameters(), lr=optimizer_settings.lr, weight_decay=optimizer_settings.weight_decay,
        betas=optimizer_settings.betas, eps=optimizer_settings.eps,
    )
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    trainable = {**groups['query LoRA'], **groups['query Projector']}
    if len(optimized) != len(trainable) or {id(p) for p in optimized} != {id(p) for p in trainable.values()}:
        raise RuntimeError('Optimizer must contain exactly Query LoRA + Query Projector')
    return optimizer


def _device_snapshot(parameters):
    # Bases stay on their device so per-step frozen audits avoid host copies.
    return {name: p.detach().clone() for name, p in parameters.items()}


def validate_full_sources(sources, expected_count):
    if len(sources) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} training sources, got {len(sources)}')
    if len({source.sequence_id for source in sources}) != len(sources):
        raise ValueError('Training source IDs must be unique')


def run_full_streaming_moco(
    moco, processor, sources, device, run_dir, *, run_id, provenance, run_config, resume=False,
    stop_after_videos=None, experiment=None, config=None, expected_source_count=None,
    resume_interval=None, snapshot_interval=None, metric_interval=None,
):
    """Process `sources` once, or continue from `resume/latest.pt`.

    Returns a summary dict. `stop_after_videos` pauses at that completed video
    boundary after writing `latest.pt` (non-final); the run continues only via
    an explicit resume. Any integrity, decode or numerical failure propagates.
    """
    moco_settings = _MOCO_CONFIG if config is None else config.moco
    sequential_settings = _SEQUENTIAL_CONFIG if config is None else config.sequential
    protocol = moco_settings.protocol
    protocol_version = run_config.get('protocol_version', moco_settings.protocol_version)
    optimizer_settings = moco_settings.optimizer
    production_config = True if config is None else config.production_groups_match()
    snapshot_artifact = SNAPSHOT_ARTIFACT if config is None else config.tracking['artifacts']['moco_snapshot']
    if config is not None:
        actual_lora = lora_config(moco.query_encoder)
        expected_lora = config.encoder.lora.metadata()
        if (
            moco.feature_size, moco.projection_size, moco.queue.capacity, moco.momentum, moco.temperature,
        ) != (
            config.encoder.feature_size, moco_settings.projection_size, moco_settings.queue_capacity,
            moco_settings.momentum, moco_settings.temperature,
        ) or actual_lora != expected_lora:
            raise RuntimeError('Resolved encoder/MoCo config differs from the constructed model')
    expected_source_count = (
        ACTIVITYNET_TRAINING_COUNT if expected_source_count is None and config is None
        else config.source_selection.counts['training'] if expected_source_count is None
        else expected_source_count
    )
    resume_interval = moco_settings.intervals.resume_videos if resume_interval is None else resume_interval
    snapshot_interval = moco_settings.intervals.snapshot_videos if snapshot_interval is None else snapshot_interval
    metric_interval = moco_settings.intervals.comet_metric_updates if metric_interval is None else metric_interval
    for name, value in (
        ('expected_source_count', expected_source_count), ('resume_interval', resume_interval),
        ('snapshot_interval', snapshot_interval), ('metric_interval', metric_interval),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    validate_full_sources(sources, expected_source_count)
    if production_config and provenance.get('implementation', {}).get('dirty') is not False:
        raise RuntimeError('Production full-dataset MoCo requires a clean implementation checkout')
    if type(run_config.get('seed')) is not int or not 0 <= run_config['seed'] < 2 ** 32:
        raise RuntimeError('Full-dataset MoCo requires a resolved seed in [0, 2**32)')
    runtime_device = device_identity(device)
    if run_config.get('device_identity') != runtime_device:
        raise RuntimeError('Resolved device identity differs from the actual training device')
    if stop_after_videos is not None and (type(stop_after_videos) is not int or stop_after_videos < 1):
        raise ValueError('stop_after_videos must be a positive integer')
    paths = RunPaths(run_dir)
    source_ids = [source.sequence_id for source in sources]
    groups = parameter_groups(moco)

    if resume:
        if not paths.latest.is_file():
            raise FileNotFoundError(f'No resume checkpoint: {paths.latest}')
        checkpoint = load_resume_checkpoint(paths.latest)
        audit_parameters(moco)
        if len(moco.queue) or any(p.grad is not None for p in moco.parameters()):
            raise RuntimeError('Resume requires a freshly constructed model')
        fingerprints = {side: base_fingerprint(getattr(moco, f'{side}_encoder')) for side in ('query', 'key')}
        optimizer = _new_optimizer(moco, groups, optimizer_settings)
        identity = run_identity(
            moco, optimizer, run_id, source_ids, provenance, fingerprints, run_config, runtime_device,
            protocol=protocol, protocol_version=protocol_version,
        )
        validate_resume_identity(checkpoint['identity'], identity)
        counters = dict(checkpoint['counters'])
        if counters['next_source_id'] != (source_ids[counters['next_video_index']]
                                          if counters['next_video_index'] < len(source_ids) else None):
            raise RuntimeError('Next source ID in the checkpoint does not match the ordered sources')
        if read_json(paths.sources) != source_ids:
            raise RuntimeError('Saved ordered source list differs from the current Adapter order')
        restore_training_state(moco, optimizer, checkpoint['state'], device)
        restore_rng_state(checkpoint['rng'])
        warmed = True
    else:
        if paths.root.exists():
            raise FileExistsError(f'Run directory already exists; use resume or a new run_id: {paths.root}')
        audit_initial_state(moco)
        if any(p.grad is not None for p in moco.parameters()):
            raise RuntimeError('Fresh model must have no gradients')
        fingerprints = {side: base_fingerprint(getattr(moco, f'{side}_encoder')) for side in ('query', 'key')}
        if fingerprints['query'] != fingerprints['key']:
            raise RuntimeError('Query / Key base weights differ')
        paths.root.mkdir(parents=True)
        write_json_atomic(paths.sources, source_ids)
        counters = {'processed_videos': 0, 'next_video_index': 0, 'global_update_step': 0,
                    'next_source_id': source_ids[0], 'final': False}
        optimizer = None
        warmed = False
    evaluation_base_fingerprint = encoder_base_fingerprint(moco.query_encoder)
    audit_finite(moco)
    before = _device_snapshot({**groups['query base'], **groups['key base']})
    expected_queue = deque(
        ((entry.sequence_id, entry.sequence_index, entry.key.detach().clone()) for entry in moco.queue.entries),
        maxlen=moco.queue.capacity,
    )
    audit_queue(moco, expected_queue)

    def identity_now():
        return run_identity(
            moco, optimizer, run_id, source_ids, provenance, fingerprints, run_config, runtime_device,
            protocol=protocol, protocol_version=protocol_version,
        )

    def snapshot_metadata(final):
        return {
            'protocol_version': protocol_version, 'run_id': run_id,
            'implementation': {key: provenance['implementation'].get(key) for key in (
                'repository', 'branch', 'commit', 'dirty', 'tracked_diff_sha256')},
            'sequential_loader': provenance['sequential_loader'], 'base_model': provenance['base_model'],
            'base_fingerprint': evaluation_base_fingerprint, 'versions': provenance['versions'],
            'dataset': {**run_config['dataset'], 'source_count': len(source_ids),
                        'ordered_source_sha256': ordered_source_sha256(source_ids)},
            'processed_videos': counters['processed_videos'],
            'global_update_step': counters['global_update_step'], 'final': final,
            'protocol': identity_now()['protocol'],
            'feature_size': moco.feature_size, 'projection_size': moco.projection_size,
            'frames_per_chunk': sequential_settings.frames_per_chunk,
            'queue_capacity': moco.queue.capacity,
            'momentum': moco.momentum, 'temperature': moco.temperature,
            'lora_config': lora_config(moco.query_encoder), 'optimizer_config': optimizer_config(optimizer),
            'seed': run_config['seed'], 'device_identity': runtime_device,
            'moco_experiment_key': None if experiment is None else experiment.get_key(),
            'production_config': production_config,
            'source_selection': run_config.get('source_selection'),
            'selection_sha256': run_config.get('selection_sha256'),
            'selection_record': run_config.get('selection_record'),
            'created': utc_now(),
        }

    def save_snapshot():
        final = counters['final']
        target, metadata = write_evaluation_snapshot(paths.snapshots, moco.query_encoder, snapshot_metadata(final))
        if metadata.get('comet', {}).get('status') not in ('logged',):
            aliases = [f'run-{run_id}-videos-{counters["processed_videos"]:06d}'] + (
                [f'run-{run_id}-final'] if final else [])
            log_artifact(experiment, target, 'metadata.json', snapshot_artifact, 'model',
                         {file: metadata['files'][file] for file in SNAPSHOT_FILES},
                         {key: metadata.get(key) for key in (
                             'run_id', 'processed_videos', 'global_update_step', 'final',
                             'protocol_version', 'selection_sha256',
                         )}, aliases,
                         project_name=(
                             _TRACKING_CONFIG['comet_project'] if config is None
                             else config.tracking['comet_project']
                         ))
        recorder.event('evaluation_snapshot', path=str(target), **{
            key: counters[key] for key in ('processed_videos', 'global_update_step', 'final')})

    def snapshot_due():
        return counters['final'] or counters['processed_videos'] % snapshot_interval == 0

    def save_resume():
        audit_bases(groups, before)
        audit_finite(moco)
        audit_queue(moco, expected_queue)
        digest = save_resume_checkpoint(paths.latest, identity_now(), dict(counters),
                                        training_state(moco, optimizer))
        recorder.event('resume_checkpoint', path=str(paths.latest), sha256=digest, **counters)

    recorder = _Recorder(paths.audit_log, experiment, metric_interval)
    try:
        if resume:
            recorder.event('resume', **counters)
            if counters['processed_videos'] and snapshot_due() and not (
                paths.snapshots / snapshot_name(counters['processed_videos'], counters['global_update_step'],
                                                counters['final'])).exists():
                save_snapshot()  # A crash between latest.pt and the snapshot left it missing.
            if counters['final']:
                raise RuntimeError('This run already completed its single pass; final runs are not resumed')
        else:
            write_json_atomic(paths.metadata, {
                'run_id': run_id, 'created': utc_now(), 'protocol_version': protocol_version,
                'config': run_config, 'provenance': provenance, 'source_count': len(source_ids),
                'ordered_source_sha256': ordered_source_sha256(source_ids),
                'production_config': production_config,
                'source_selection': run_config.get('source_selection'),
                'selection_sha256': run_config.get('selection_sha256'),
                'selection_record': run_config.get('selection_record'),
                'moco_experiment_key': None if experiment is None else experiment.get_key(),
            })
            recorder.event('start', **counters)
        for video_index in range(counters['next_video_index'], len(sources)):
            source = sources[video_index]
            chunk_count = 0
            recorder.context = {'video_index': video_index}
            with ordered_samples(
                (source,), stream_mode=protocol.stream_mode,
                frames_per_chunk=sequential_settings.frames_per_chunk,
                round_robin_stream_count=sequential_settings.round_robin_stream_count,
            ) as samples:
                for sample in samples:
                    if not warmed:
                        if (video_index, sample.sequence_index, len(moco.queue)) != (0, 0, 0):
                            raise RuntimeError('Warm-up is only allowed on the first chunk of the first video')
                        initial = _device_snapshot(dict(moco.named_parameters()))
                        query_view, key_view = make_two_views(sample, protocol.key_transform)
                        audit_views(sample, query_view, key_view, protocol.key_transform)
                        key = encode_key_view(key_view, processor, moco, device)[-1]
                        enqueue_key(moco, key, sample, expected_queue)
                        if count_changed(dict(moco.named_parameters()), initial) or any(
                            p.grad is not None for p in moco.parameters()
                        ):
                            raise RuntimeError('Warm-up must not change model parameters or create gradients')
                        recorder.event('warmup', sequence_id=sample.sequence_id,
                                       sequence_index=sample.sequence_index, queue_count=len(moco.queue))
                        del initial
                        optimizer = _new_optimizer(moco, groups, optimizer_settings)
                        warmed = True
                    else:
                        train_step(moco, processor, sample, device, optimizer, groups, before, expected_queue,
                                   counters['global_update_step'], protocol, report=recorder.step)
                        counters['global_update_step'] += 1
                        recorder.flush_metrics(counters['global_update_step'], counters['processed_videos'])
                    chunk_count += 1
            if chunk_count == 0:
                raise RuntimeError(f'Source produced no chunks: {source.sequence_id}')
            counters['processed_videos'] = video_index + 1
            counters['next_video_index'] = video_index + 1
            counters['next_source_id'] = source_ids[video_index + 1] if video_index + 1 < len(sources) else None
            counters['final'] = counters['processed_videos'] == len(sources)
            recorder.event('video_complete', sequence_id=source.sequence_id, chunk_count=chunk_count, **counters)
            stopping = stop_after_videos is not None and counters['processed_videos'] >= stop_after_videos
            if counters['final'] or stopping or counters['processed_videos'] % resume_interval == 0:
                save_resume()
            if snapshot_due():
                save_snapshot()
            if stopping and not counters['final']:
                recorder.event('paused', **counters)
                return {'status': 'paused', **counters, 'queue_count': len(moco.queue)}
        audit_bases(groups, before)
        audit_finite(moco)
        audit_queue(moco, expected_queue)
        recorder.event('complete', **counters, queue_count=len(moco.queue))
        return {'status': 'complete', **counters, 'queue_count': len(moco.queue)}
    except BaseException as error:
        recorder.event('stopped', reason=repr(error), **counters)
        raise
    finally:
        recorder.close()
