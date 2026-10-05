"""Full-dataset single-pass Stage 6B Streaming MoCo on ActivityNet training.

Fresh run (the run directory must not exist):
    python -m scripts.moco.train_full_streaming_moco /path/to/ActivityNet --run-id <id> --device cuda

Resume from log/moco/<id>/resume/latest.pt at the saved video boundary:
    python -m scripts.moco.train_full_streaming_moco /path/to/ActivityNet --run-id <id> --device cuda --resume

`--stop-after-videos N` pauses at the N-th completed video boundary (writes
latest.pt, no final snapshot); intended for short smoke and resume checks.
"""

import argparse
import json
import re
from pathlib import Path

import sequential_loader as sl
import torch
from transformers import AutoImageProcessor

from logger.comet_lineage import end_experiment, start_experiment
from model.backbones.vit import ViTLoRAFrameEncoder
from self_supervised.moco import ViTLoRAMoCo
from training.moco_checkpoint import PROTOCOL_VERSION
from training.moco_protocol import STAGE6B_PROTOCOL
from training import streaming_moco_full as full
from utils.artifact_io import read_json, write_json_atomic
from utils.provenance import CHECKPOINT_ID, collect_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true', help='Continue from resume/latest.pt only')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--stop-after-videos', type=int)
    parser.add_argument('--output-root', type=Path, default=Path('log/moco'))
    parser.add_argument('--disable-comet', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('--run-id may contain only letters, digits, ".", "_" and "-"')
    if args.stop_after_videos is not None and args.stop_after_videos < 1:
        parser.error('--stop-after-videos must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available')

    provenance = {**collect_provenance(), 'base_model': CHECKPOINT_ID}
    device = torch.device(args.device)
    run_dir = args.output_root / args.run_id
    config = {
        'run_id': args.run_id, 'protocol_version': PROTOCOL_VERSION, 'resume': args.resume,
        'dataset': {'name': 'ActivityNet', 'version': '1.3', 'split': 'training',
                    'root': str(args.dataset_root)},
        'protocol': {'stream_mode': STAGE6B_PROTOCOL.stream_mode, 'key_transform': STAGE6B_PROTOCOL.key_transform,
                     'negative_policy': STAGE6B_PROTOCOL.negative_policy},
        'optimizer': {'class': 'AdamW', 'lr': full.LEARNING_RATE, 'weight_decay': full.WEIGHT_DECAY},
        'intervals': {'resume_videos': full.RESUME_INTERVAL, 'snapshot_videos': full.SNAPSHOT_INTERVAL,
                      'comet_metric_updates': full.METRIC_INTERVAL},
        'expected_source_count': full.ACTIVITYNET_TRAINING_COUNT,
        'stop_after_videos': args.stop_after_videos,
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'run_dir': str(run_dir),
    }
    print('Resolved config: ' + json.dumps(config, sort_keys=True), flush=True)
    print('Provenance: ' + json.dumps(provenance, sort_keys=True), flush=True)

    sources = sl.ActivityNetAdapter(dataset_root=args.dataset_root).sequence_sources('training')
    full.validate_full_sources(sources, full.ACTIVITYNET_TRAINING_COUNT)
    existing_key = None
    if args.resume:
        existing_key = read_json(run_dir / 'run_metadata.json').get('moco_experiment_key')
    experiment, comet = start_experiment(
        f'moco-full__{args.run_id}', {'config': config, 'provenance': provenance}, tags=('moco', 'full-dataset'),
        disabled=args.disable_comet, existing_key=existing_key,
    )
    print(f'Comet: {json.dumps(comet)}', flush=True)

    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    moco = ViTLoRAMoCo(ViTLoRAFrameEncoder(CHECKPOINT_ID)).to(device).train()
    try:
        result = full.run_full_streaming_moco(
            moco, processor, sources, device, run_dir, run_id=args.run_id, provenance=provenance,
            run_config=config, resume=args.resume, stop_after_videos=args.stop_after_videos, experiment=experiment,
        )
    finally:
        error = end_experiment(experiment)
        if error:
            print(f'Comet end failed: {error}', flush=True)
    if args.resume:
        metadata = read_json(run_dir / 'run_metadata.json')
        metadata.setdefault('resumes', []).append({'config': config, 'provenance': provenance, 'result': result})
        write_json_atomic(run_dir / 'run_metadata.json', metadata)
    print(f'Run directory: {run_dir}', flush=True)
    print('Result: ' + json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
