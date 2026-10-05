"""Extract raw [768] segment features for one condition from a Gate-passed manifest.

    python -m scripts.linear_probe.extract_features /path/to/ActivityNet --manifest-id lp-v1 \
        --condition base_vit --device cuda
    python -m scripts.linear_probe.extract_features /path/to/ActivityNet --manifest-id lp-v1 \
        --condition moco_query_lora_final --snapshot log/moco/<run>/evaluation_snapshots/<..._final> --device cuda

Runs once per condition (both splits), independent of Probe seeds. An existing
feature directory is reused only if its inputs and file hashes match.
"""

import argparse
import json
from pathlib import Path

import sequential_loader as sl
import torch
from transformers import AutoImageProcessor

from evaluation import activitynet_manifest as manifest
from evaluation import segment_features as features
from logger.comet_lineage import end_experiment, log_artifact, start_experiment
from model.backbones.vit import ViTFrameEncoder, ViTLoRAFrameEncoder
from training.moco_checkpoint import base_fingerprint, load_query_lora_snapshot
from utils.artifact_io import canonical_json_bytes, read_json, sha256_bytes
from utils.provenance import CHECKPOINT_ID, collect_provenance, utc_now


ARTIFACT_NAMES = {'base_vit': 'activitynet-segment-features-base-vit',
                  'moco_query_lora_final': 'activitynet-segment-features-moco-query-lora'}


def build_encoder(condition, snapshot, production):
    if condition == 'base_vit':
        if snapshot is not None:
            raise ValueError('base_vit takes no snapshot')
        encoder = ViTFrameEncoder(CHECKPOINT_ID)
        return encoder, {'type': 'base_vit', 'base_model': CHECKPOINT_ID,
                         'base_fingerprint': base_fingerprint(encoder)}
    if snapshot is None:
        raise ValueError('moco_query_lora_final requires --snapshot')
    encoder = ViTLoRAFrameEncoder(CHECKPOINT_ID)
    fingerprint = base_fingerprint(encoder)
    metadata = load_query_lora_snapshot(encoder, snapshot)
    if production and metadata['final'] is not True:
        raise RuntimeError('Production features require the final Query LoRA snapshot')
    return encoder, {
        'type': 'moco_query_lora', 'base_model': CHECKPOINT_ID, 'base_fingerprint': fingerprint,
        'snapshot': {'path': str(snapshot), 'files': metadata['files'], 'run_id': metadata['run_id'],
                     'processed_videos': metadata['processed_videos'],
                     'global_update_step': metadata['global_update_step'], 'final': metadata['final'],
                     'moco_experiment_key': metadata.get('moco_experiment_key'), 'comet': metadata.get('comet')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--manifest-id', required=True)
    parser.add_argument('--condition', choices=features.CONDITIONS, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--feature-id')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output-root', type=Path, default=Path('log/linear_probe'))
    parser.add_argument('--disable-comet', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available')
    print('Resolved config: ' + json.dumps({key: str(value) for key, value in vars(args).items()}), flush=True)

    manifest_dir = args.output_root / 'manifest' / args.manifest_id
    rows, _, manifest_metadata = manifest.load_manifest(manifest_dir)
    production = manifest_metadata['production']
    provenance = collect_provenance()
    encoder, encoder_info = build_encoder(args.condition, args.snapshot, production)
    encoder_sha = sha256_bytes(canonical_json_bytes(encoder_info))
    feature_id = args.feature_id or f'{args.manifest_id}__{encoder_sha[:12]}'
    directory = args.output_root / 'features' / args.condition / feature_id
    inputs = {
        'condition': args.condition, 'feature_id': feature_id, 'production': production,
        'manifest': {key: manifest_metadata[key] for key in (
            'manifest_id', 'segment_manifest_sha256', 'label_mapping_sha256')}
        | {'local_path': str(manifest_dir), 'comet': manifest_metadata.get('comet')},
        'encoder': encoder_info, 'encoder_sha256': encoder_sha,
        'feature_definition': 'valid frame CLS [T,768] -> masked mean chunk [768] -> mean of manifest chunks; '
                              'raw float32, no normalization',
        'preprocessing': {'processor': CHECKPOINT_ID, 'frames_per_chunk': 16, 'dtype': 'float32'},
    }
    if directory.exists():
        existing = read_json(directory / 'metadata.json')
        if {key: existing.get(key) for key in inputs if key != 'manifest'} != {
            key: value for key, value in inputs.items() if key != 'manifest'
        } or existing['manifest']['segment_manifest_sha256'] != inputs['manifest']['segment_manifest_sha256']:
            raise RuntimeError(f'Different feature artifact already exists: {directory}')
        features.load_features(directory, rows, manifest_metadata, args.condition)
        print(f'Reusing verified feature artifact: {directory}', flush=True)
        return

    adapter = sl.ActivityNetAdapter(dataset_root=args.dataset_root)
    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    # Extraction never trains: both encoders are fully frozen and in eval mode.
    encoder = encoder.to(device).eval().requires_grad_(False)
    experiment, comet = start_experiment(
        f'lp-v1__features__{args.condition.replace("_", "-")}', inputs, tags=('linear-probe', 'features'),
        disabled=args.disable_comet)
    splits = {}
    for split in manifest.SPLITS:
        split_rows = [row for row in rows if row['split'] == split]

        def progress(done, total, split=split):
            if done == total or done % 100 == 0:
                print(json.dumps({'event': 'feature_progress', 'split': split, 'videos': done, 'total': total}),
                      flush=True)
        splits[split] = features.extract_split(split_rows, adapter.sequence_sources(split), processor, encoder,
                                               device, progress)
    metadata = features.write_feature_artifact(directory, splits, rows, {
        **inputs, 'counts': {split: len(splits[split]['segment_ids']) for split in manifest.SPLITS},
        'device': str(device), 'provenance': provenance, 'created': utc_now(), 'local_path': str(directory),
        'comet_experiment': comet,
    })
    status = log_artifact(experiment, directory, 'metadata.json', ARTIFACT_NAMES[args.condition], 'dataset',
                          metadata['files'], {'feature_id': feature_id, 'condition': args.condition,
                                              'segment_manifest_sha256': manifest_metadata['segment_manifest_sha256'],
                                              'encoder_sha256': encoder_sha}, aliases=(feature_id,))
    end_experiment(experiment)
    print(json.dumps({'event': 'features_written', 'path': str(directory), 'files': metadata['files'],
                      'counts': metadata['counts'], 'comet': {**comet, **status}}), flush=True)


if __name__ == '__main__':
    main()
