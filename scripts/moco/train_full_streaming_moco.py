"""Full-dataset single-pass Stage 6B Streaming MoCo on ActivityNet training.

Fresh run (the run directory must not exist):
    python -m scripts.moco.train_full_streaming_moco runtime.dataset_root=/path/to/ActivityNet \
        runtime.run_id=<id> runtime.seed=<seed> runtime.device=cuda

Resume from log/moco/<id>/resume/latest.pt at the saved video boundary:
    python -m scripts.moco.train_full_streaming_moco runtime.dataset_root=/path/to/ActivityNet \
        runtime.run_id=<id> runtime.seed=<same-seed> runtime.device=cuda runtime.resume=true

`runtime.stop_after_videos=N` pauses at the N-th completed video boundary (writes
latest.pt, no final snapshot); intended for short smoke and resume checks.
"""

import json

import hydra
from omegaconf import DictConfig, OmegaConf
import sequential_loader as sl
import torch
from transformers import AutoImageProcessor

from integration.activitynet_source_selection import select_activitynet_sources
from logger.comet_lineage import display_tag, end_experiment, scope_tag, start_experiment
from model.backbones.vit import ViTLoRAFrameEncoder
from self_supervised.moco import ViTLoRAMoCo
from training.moco_checkpoint import seed_all
from training.moco_config import FullMoCoConfig
from training import streaming_moco_full as full
from utils.artifact_io import read_json, sha256_file, write_json_atomic
from utils.configuration import plain_config
from utils.provenance import collect_provenance, device_identity, utc_now


def run(cfg):
    resolved = plain_config(cfg)
    settings = FullMoCoConfig.from_mapping(resolved)
    runtime = settings.runtime
    if runtime.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    production_config = settings.production_groups_match()
    protocol_name, protocol_version = settings.effective_protocol_identity()
    seed_all(runtime.seed)
    provenance = {
        **collect_provenance(policy=settings.provenance, require_clean=production_config),
        'base_model': settings.encoder.checkpoint_id,
    }
    device = torch.device(runtime.device)
    runtime_device = device_identity(device)
    dataset_root = runtime.dataset_root.resolve()
    run_dir = runtime.output_root / runtime.run_id
    protocol = settings.moco.protocol
    run_config = {
        'run_id': runtime.run_id, 'protocol_version': protocol_version,
        'resume': runtime.resume, 'seed': runtime.seed,
        'dataset': {'name': settings.activitynet.name, 'version': settings.activitynet.version,
                    'split': settings.activitynet.splits['training']},
        'protocol': {'stream_mode': protocol.stream_mode, 'key_transform': protocol.key_transform,
                     'negative_policy': protocol.negative_policy},
        'optimizer': {'class': settings.moco.optimizer.name, 'lr': settings.moco.optimizer.lr,
                      'weight_decay': settings.moco.optimizer.weight_decay,
                      'betas': list(settings.moco.optimizer.betas), 'eps': settings.moco.optimizer.eps},
        'intervals': {'resume_videos': settings.moco.intervals.resume_videos,
                      'snapshot_videos': settings.moco.intervals.snapshot_videos,
                      'comet_metric_updates': settings.moco.intervals.comet_metric_updates},
        'expected_full_source_count': settings.activitynet.expected_source_counts['training'],
        'expected_source_count': settings.source_selection.counts['training'],
        'frames_per_chunk': settings.sequential.frames_per_chunk,
        'round_robin_stream_count': settings.sequential.round_robin_stream_count,
        'production_config': production_config,
        'stop_after_videos': runtime.stop_after_videos,
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'device_identity': runtime_device,
        'locations': {'dataset_root': str(dataset_root), 'run_dir': str(run_dir.resolve())},
    }
    adapter = sl.ActivityNetAdapter(dataset_root=dataset_root)
    full_sources = {
        split: tuple(adapter.sequence_sources(adapter_split))
        for split, adapter_split in settings.activitynet.splits.items()
    }
    for split, sources in full_sources.items():
        full.validate_full_sources(sources, settings.activitynet.expected_source_counts[split])
    annotation_paths = {
        source.evaluation_reference.annotation_path
        for sources in full_sources.values() for source in sources
    }
    annotation_hashes = {sha256_file(path) for path in annotation_paths}
    if len(annotation_hashes) != 1:
        raise RuntimeError('ActivityNet splits must use one identical annotation file')
    selection = select_activitynet_sources(
        full_sources, settings.source_selection,
        expected_counts=settings.activitynet.expected_source_counts,
        dataset={'name': settings.activitynet.name, 'version': settings.activitynet.version},
        sequential_loader=provenance['sequential_loader'],
        annotation_sha256=next(iter(annotation_hashes)),
    )
    sources = selection.sources['training']
    run_config.update({
        'source_selection': selection.identity,
        'selection_sha256': selection.selection_sha256,
        'selection_record': {
            'schema': selection.identity['schema'], 'created': utc_now(),
            'implementation': provenance['implementation'],
        },
    })
    logged_selection = {
        **selection.identity,
        'splits': {
            split: {key: value for key, value in identity.items() if key != 'ordered_selected_ids'}
            for split, identity in selection.identity['splits'].items()
        },
    }
    logged_run_config = {**run_config, 'source_selection': logged_selection}
    print('Resolved Hydra config:\n' + OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    print('Resolved run config: ' + json.dumps(logged_run_config, sort_keys=True), flush=True)
    print('Provenance: ' + json.dumps(provenance, sort_keys=True), flush=True)
    existing_key = None
    if runtime.resume:
        existing_key = read_json(run_dir / 'run_metadata.json').get('moco_experiment_key')
    # Tracking scope only: an explicit stop makes a canonical-config run a smoke run.
    production_run = production_config and runtime.stop_after_videos is None
    protocol_tag = display_tag(protocol_name)
    experiment, comet = start_experiment(
        f'{protocol_tag}__moco__{runtime.run_id}', {'config': logged_run_config, 'provenance': provenance},
        tags=('moco', scope_tag(production_run), protocol_tag, f'seed-{runtime.seed}'),
        disabled=settings.disable_comet, existing_key=existing_key,
        project_name=settings.tracking['comet_project'],
    )
    print(f'Comet: {json.dumps(comet)}', flush=True)

    processor = AutoImageProcessor.from_pretrained(settings.encoder.checkpoint_id)
    encoder = ViTLoRAFrameEncoder(
        settings.encoder.checkpoint_id, feature_size=settings.encoder.feature_size,
        image_size=settings.encoder.image_size, channels=settings.encoder.channels,
        lora_config=settings.encoder.lora,
    )
    moco = ViTLoRAMoCo(encoder, config=settings.moco).to(device).train()
    try:
        result = full.run_full_streaming_moco(
            moco, processor, sources, device, run_dir, run_id=runtime.run_id, provenance=provenance,
            run_config=run_config, resume=runtime.resume, stop_after_videos=runtime.stop_after_videos,
            experiment=experiment, config=settings,
        )
    finally:
        error = end_experiment(experiment)
        if error:
            print(f'Comet end failed: {error}', flush=True)
    if runtime.resume:
        metadata = read_json(run_dir / 'run_metadata.json')
        metadata.setdefault('resumes', []).append({
            'config': run_config, 'provenance': provenance, 'result': result,
        })
        write_json_atomic(run_dir / 'run_metadata.json', metadata)
    print(f'Run directory: {run_dir}', flush=True)
    print('Result: ' + json.dumps(result, sort_keys=True), flush=True)
    return result


@hydra.main(version_base='1.3', config_path='../../conf', config_name='moco_full')
def main(cfg: DictConfig):
    run(cfg)


if __name__ == '__main__':
    main()
