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
from training.moco_checkpoint import (
    encoder_base_fingerprint, load_query_lora_snapshot, validate_production_snapshot_metadata,
)
from utils.artifact_io import canonical_json_bytes, read_json, sha256_bytes, sha256_file
from utils.provenance import CHECKPOINT_ID, collect_provenance, device_identity, utc_now


ARTIFACT_NAMES = {'base_vit': 'activitynet-segment-features-base-vit',
                  'moco_query_lora_final': 'activitynet-segment-features-moco-query-lora'}
FEATURE_DEFINITION = {
    'schema': 'activitynet-segment-feature-definition/v1',
    'frame_feature': 'ViT CLS without pooler', 'chunk_aggregation': 'masked mean over valid frames',
    'segment_aggregation': 'mean over manifest chunks', 'size': 768, 'dtype': 'float32',
    'normalization': None,
}


def dataset_sources_and_identity(dataset_root, adapter, manifest_metadata):
    """Bind extraction to the exact manifest root, annotation and source order."""
    resolved_root = str(Path(dataset_root).resolve())
    expected_root = str(Path(manifest_metadata['dataset']['root']).resolve())
    if resolved_root != expected_root:
        raise RuntimeError(f'Dataset root differs from the manifest: {resolved_root} != {expected_root}')
    sources_by_split, split_identity, annotation_paths = {}, {}, set()
    for split in manifest.SPLITS:
        sources = tuple(adapter.sequence_sources(split))
        expected = manifest_metadata['splits'][split]
        if len(sources) != manifest.SPLIT_COUNTS[split] or len(sources) != expected['source_count']:
            raise RuntimeError(f'{split} source count differs from the manifest')
        used_count = expected['used_source_count']
        if not 0 < used_count <= len(sources):
            raise RuntimeError(f'Invalid {split} used source count in the manifest')
        ordered_sha = sha256_bytes(canonical_json_bytes(
            [source.sequence_id for source in sources[:used_count]]
        ))
        if ordered_sha != expected['ordered_used_source_sha256']:
            raise RuntimeError(f'{split} Adapter source order differs from the manifest')
        annotation_paths.add(Path(sources[0].evaluation_reference.annotation_path))
        sources_by_split[split] = sources
        split_identity[split] = {
            'source_count': len(sources), 'used_source_count': used_count,
            'ordered_used_source_sha256': ordered_sha,
        }
    annotation_hashes = {sha256_file(path) for path in annotation_paths}
    if annotation_hashes != {manifest_metadata['annotation']['sha256']}:
        raise RuntimeError('ActivityNet annotation differs from the manifest')
    return sources_by_split, {
        'name': manifest_metadata['dataset']['name'], 'version': manifest_metadata['dataset']['version'],
        'resolved_root': resolved_root, 'annotation_sha256': manifest_metadata['annotation']['sha256'],
        'splits': split_identity,
    }


def extraction_provenance(provenance):
    implementation = provenance['implementation']
    return {
        'implementation': {key: implementation.get(key) for key in (
            'repository', 'commit', 'dirty', 'tracked_diff_sha256')},
        'sequential_loader': provenance['sequential_loader'], 'versions': provenance['versions'],
    }


def preprocessing_identity(processor, frames_per_chunk):
    configuration = processor.to_dict()
    return {
        'processor_checkpoint': CHECKPOINT_ID,
        'processor_class': f'{type(processor).__module__}.{type(processor).__qualname__}',
        'processor_config_sha256': sha256_bytes(canonical_json_bytes(configuration)),
        'frames_per_chunk': frames_per_chunk, 'dtype': 'float32',
    }


def build_encoder(condition, snapshot, manifest_metadata):
    if condition == 'base_vit':
        if snapshot is not None:
            raise ValueError('base_vit takes no snapshot')
        encoder = ViTFrameEncoder(CHECKPOINT_ID)
        return encoder, {'type': 'base_vit', 'base_model': CHECKPOINT_ID,
                         'base_fingerprint': encoder_base_fingerprint(encoder)}
    if snapshot is None:
        raise ValueError('moco_query_lora_final requires --snapshot')
    encoder = ViTLoRAFrameEncoder(CHECKPOINT_ID)
    fingerprint = encoder_base_fingerprint(encoder)
    metadata = load_query_lora_snapshot(encoder, snapshot)
    if manifest_metadata['production']:
        validate_production_snapshot_metadata(
            metadata,
            expected_source_sha256=manifest_metadata['splits']['training']['ordered_used_source_sha256'],
            expected_dataset_root=manifest_metadata['dataset']['root'],
            expected_base_fingerprint=fingerprint,
            expected_source_count=manifest.SPLIT_COUNTS['training'],
        )
    snapshot_fields = (
        'schema', 'protocol_version', 'run_id', 'implementation', 'sequential_loader', 'versions',
        'base_model', 'base_fingerprint', 'dataset', 'processed_videos', 'global_update_step', 'final',
        'protocol', 'queue_capacity', 'momentum', 'temperature', 'lora_config', 'optimizer_config',
        'seed', 'device_identity', 'moco_experiment_key',
    )
    return encoder, {
        'type': 'moco_query_lora', 'base_model': CHECKPOINT_ID, 'base_fingerprint': fingerprint,
        'snapshot': {'local_path': str(snapshot), 'files': metadata['files'],
                     **{key: metadata.get(key) for key in snapshot_fields}},
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
    provenance = collect_provenance(require_clean=production)
    adapter = sl.ActivityNetAdapter(dataset_root=args.dataset_root.resolve())
    sources_by_split, dataset_identity = dataset_sources_and_identity(
        args.dataset_root, adapter, manifest_metadata,
    )
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    preprocessing = preprocessing_identity(processor, manifest_metadata['frames_per_chunk'])
    encoder, encoder_info = build_encoder(args.condition, args.snapshot, manifest_metadata)
    device = torch.device(args.device)
    shared_contract = {
        'schema': 'activitynet-shared-segment-feature-contract/v1',
        'dataset': dataset_identity,
        'manifest': {key: manifest_metadata[key] for key in (
            'manifest_id', 'segment_manifest_sha256', 'label_mapping_sha256')},
        'base_encoder': {'model': CHECKPOINT_ID, 'fingerprint': encoder_info['base_fingerprint']},
        'preprocessing': preprocessing, 'feature_definition': FEATURE_DEFINITION,
        'extraction_provenance': extraction_provenance(provenance),
        'device_identity': device_identity(device),
    }
    shared_contract_sha = sha256_bytes(canonical_json_bytes(shared_contract))
    encoder_sha = sha256_bytes(canonical_json_bytes(encoder_info))
    feature_id = args.feature_id or f'{args.manifest_id}__{encoder_sha[:12]}'
    directory = args.output_root / 'features' / args.condition / feature_id
    inputs = {
        'condition': args.condition, 'feature_id': feature_id, 'production': production,
        'manifest': {key: manifest_metadata[key] for key in (
            'manifest_id', 'segment_manifest_sha256', 'label_mapping_sha256')}
        | {'local_path': str(manifest_dir), 'comet': manifest_metadata.get('comet')},
        'encoder': encoder_info, 'encoder_sha256': encoder_sha,
        'feature_definition': FEATURE_DEFINITION, 'preprocessing': preprocessing,
        'shared_feature_contract': shared_contract,
        'shared_feature_contract_sha256': shared_contract_sha,
    }
    if directory.exists():
        existing = read_json(directory / 'metadata.json')
        if {key: existing.get(key) for key in inputs if key != 'manifest'} != {
            key: value for key, value in inputs.items() if key != 'manifest'
        } or any(existing['manifest'].get(key) != inputs['manifest'][key] for key in (
            'manifest_id', 'segment_manifest_sha256', 'label_mapping_sha256')):
            raise RuntimeError(f'Different feature artifact already exists: {directory}')
        features.load_features(directory, rows, manifest_metadata, args.condition)
        print(f'Reusing verified feature artifact: {directory}', flush=True)
        return

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
        splits[split] = features.extract_split(split_rows, sources_by_split[split], processor, encoder,
                                               device, progress)
    metadata = features.write_feature_artifact(directory, splits, rows, {
        **inputs, 'counts': {split: len(splits[split]['segment_ids']) for split in manifest.SPLITS},
        'device': str(device), 'provenance': provenance, 'created': utc_now(), 'local_path': str(directory),
        'comet_experiment': comet,
    })
    status = log_artifact(experiment, directory, 'metadata.json', ARTIFACT_NAMES[args.condition], 'dataset',
                          metadata['files'], {'feature_id': feature_id, 'condition': args.condition,
                                              'segment_manifest_sha256': manifest_metadata['segment_manifest_sha256'],
                                              'encoder_sha256': encoder_sha,
                                              'shared_feature_contract_sha256': shared_contract_sha},
                          aliases=(feature_id,))
    end_experiment(experiment)
    print(json.dumps({'event': 'features_written', 'path': str(directory), 'files': metadata['files'],
                      'counts': metadata['counts'], 'comet': {**comet, **status}}), flush=True)


if __name__ == '__main__':
    main()
