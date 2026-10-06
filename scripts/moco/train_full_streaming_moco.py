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

from logger.comet_lineage import end_experiment, start_experiment
from model.backbones.vit import ViTLoRAFrameEncoder
from self_supervised.moco import ViTLoRAMoCo
from training.moco_checkpoint import seed_all
from training.moco_config import FullMoCoConfig
from training import streaming_moco_full as full
from utils.artifact_io import read_json, write_json_atomic
from utils.configuration import plain_config
from utils.provenance import collect_provenance, device_identity


def run(cfg):
    resolved = plain_config(cfg)
    settings = FullMoCoConfig.from_mapping(resolved)
    runtime = settings.runtime
    if runtime.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')

    production_config = settings.production_groups_match()
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
        'run_id': runtime.run_id, 'protocol_version': settings.moco.protocol_version,
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
        'expected_source_count': settings.activitynet.expected_source_counts['training'],
        'frames_per_chunk': settings.sequential.frames_per_chunk,
        'round_robin_stream_count': settings.sequential.round_robin_stream_count,
        'production_config': production_config,
        'stop_after_videos': runtime.stop_after_videos,
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'device_identity': runtime_device,
        'locations': {'dataset_root': str(dataset_root), 'run_dir': str(run_dir.resolve())},
    }
    print('Resolved Hydra config:\n' + OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    print('Resolved run config: ' + json.dumps(run_config, sort_keys=True), flush=True)
    print('Provenance: ' + json.dumps(provenance, sort_keys=True), flush=True)

    split = settings.activitynet.splits['training']
    sources = sl.ActivityNetAdapter(dataset_root=dataset_root).sequence_sources(split)
    full.validate_full_sources(sources, settings.activitynet.expected_source_counts['training'])
    existing_key = None
    if runtime.resume:
        existing_key = read_json(run_dir / 'run_metadata.json').get('moco_experiment_key')
    experiment, comet = start_experiment(
        f'moco-full__{runtime.run_id}', {'config': run_config, 'provenance': provenance},
        tags=('moco', 'full-dataset'), disabled=settings.disable_comet, existing_key=existing_key,
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
