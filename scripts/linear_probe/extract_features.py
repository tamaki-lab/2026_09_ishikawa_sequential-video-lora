"""Extract segment features for one condition from a Gate-passed manifest.

    python -m scripts.linear_probe.extract_features runtime.dataset_root=/path/to/ActivityNet \
        runtime.condition=base_vit runtime.device=cuda
    python -m scripts.linear_probe.extract_features runtime.dataset_root=/path/to/ActivityNet \
        runtime.condition=moco_query_lora_final runtime.snapshot=log/moco/<run>/evaluation_snapshots/<..._final>

Runs once per condition (both splits), independent of Probe seeds. An existing
feature directory is reused only if its inputs and file hashes match.
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
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
from scripts.linear_probe.configuration import feature_science_contract, science_contract
from utils.artifact_io import canonical_json_bytes, read_json, sha256_bytes, sha256_file
from utils.configuration import plain_config, validate_path_component
from utils.provenance import collect_provenance, device_identity, utc_now


MANIFEST_SCIENCE_GROUPS = ('activitynet', 'sequential', 'provenance')


def dataset_sources_and_identity(adapter, manifest_metadata, expected_counts):
    """Bind extraction to annotation content and source order, not its mount path."""
    sources_by_split, split_identity, annotation_paths = {}, {}, set()
    for split in manifest.SPLITS:
        sources = tuple(adapter.sequence_sources(split))
        expected = manifest_metadata['splits'][split]
        if len(sources) != expected_counts[split] or len(sources) != expected['source_count']:
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
        'annotation_sha256': manifest_metadata['annotation']['sha256'],
        'splits': split_identity,
    }


def extraction_provenance(provenance):
    implementation = provenance['implementation']
    return {
        'implementation': {key: implementation.get(key) for key in (
            'repository', 'commit', 'dirty', 'tracked_diff_sha256')},
        'sequential_loader': provenance['sequential_loader'], 'versions': provenance['versions'],
    }


def preprocessing_identity(processor, checkpoint_id, frames_per_chunk):
    configuration = processor.to_dict()
    return {
        'processor_checkpoint': checkpoint_id,
        'processor_class': f'{type(processor).__module__}.{type(processor).__qualname__}',
        'processor_config_sha256': sha256_bytes(canonical_json_bytes(configuration)),
        'frames_per_chunk': frames_per_chunk, 'dtype': 'float32',
    }


def build_encoder(condition, snapshot, manifest_metadata, encoder_config, production, expected_source_count):
    encoder_kwargs = {
        'feature_size': encoder_config['feature_size'], 'image_size': encoder_config['image_size'],
        'channels': encoder_config['channels'],
    }
    checkpoint_id = encoder_config['checkpoint_id']
    if condition == 'base_vit':
        if snapshot is not None:
            raise ValueError('base_vit takes no snapshot')
        encoder = ViTFrameEncoder(checkpoint_id, **encoder_kwargs)
        return encoder, {
            'type': 'base_vit', 'base_model': checkpoint_id,
            'base_fingerprint': encoder_base_fingerprint(encoder),
        }
    if snapshot is None:
        raise ValueError('moco_query_lora_final requires runtime.snapshot')
    encoder = ViTLoRAFrameEncoder(
        checkpoint_id, **encoder_kwargs, lora_config=encoder_config['lora'],
    )
    fingerprint = encoder_base_fingerprint(encoder)
    metadata = load_query_lora_snapshot(encoder, snapshot)
    if production:
        validate_production_snapshot_metadata(
            metadata,
            expected_source_sha256=manifest_metadata['splits']['training']['ordered_used_source_sha256'],
            expected_base_fingerprint=fingerprint,
            expected_source_count=expected_source_count,
        )
    snapshot_fields = (
        'schema', 'protocol_version', 'run_id', 'implementation', 'sequential_loader', 'versions',
        'base_model', 'base_fingerprint', 'dataset', 'processed_videos', 'global_update_step', 'final',
        'protocol', 'feature_size', 'projection_size', 'frames_per_chunk', 'queue_capacity',
        'momentum', 'temperature', 'lora_config', 'optimizer_config',
        'seed', 'device_identity', 'moco_experiment_key', 'production_config',
    )
    snapshot_identity = {key: metadata.get(key) for key in snapshot_fields}
    if isinstance(snapshot_identity.get('dataset'), dict):
        # Legacy snapshots stored the mount point inside ``dataset``. Keep the
        # content identity while excluding that placement-only field.
        snapshot_identity['dataset'] = {
            key: value for key, value in snapshot_identity['dataset'].items() if key != 'root'
        }
    return encoder, {
        'type': 'moco_query_lora', 'base_model': checkpoint_id, 'base_fingerprint': fingerprint,
        'snapshot': {'files': metadata['files'], **snapshot_identity},
    }


def run(cfg):
    resolved = plain_config(cfg)
    runtime = resolved['runtime']
    logging = resolved['logging']
    if type(logging['disable_comet']) is not bool:
        raise ValueError('logging.disable_comet must be boolean')
    if type(logging['progress_interval_videos']) is not int or logging['progress_interval_videos'] < 1:
        raise ValueError('logging.progress_interval_videos must be a positive integer')
    validate_path_component('runtime.manifest_id', runtime['manifest_id'])
    if runtime['feature_id'] is not None:
        validate_path_component('runtime.feature_id', runtime['feature_id'])
    validate_path_component('runtime.condition', runtime['condition'])
    if runtime['condition'] not in resolved['linear_probe']['conditions']:
        raise ValueError(f'Unknown feature condition: {runtime["condition"]}')
    features.validate_feature_definition(resolved['linear_probe']['feature_definition'])
    if runtime['device'] not in ('cpu', 'cuda'):
        raise ValueError('runtime.device must be cpu or cuda')
    if runtime['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    print('Resolved config:\n' + OmegaConf.to_yaml(cfg, resolve=True), flush=True)

    output_root = Path(runtime['output_root'])
    dataset_root = Path(runtime['dataset_root']).resolve()
    snapshot = None if runtime['snapshot'] is None else Path(runtime['snapshot'])
    manifest_dir = output_root / 'manifest' / runtime['manifest_id']
    rows, _, manifest_metadata = manifest.load_manifest(manifest_dir)
    manifest_science = science_contract(cfg, MANIFEST_SCIENCE_GROUPS)
    if (manifest_metadata.get('science') or {}).get('sha256') != manifest_science['sha256']:
        raise RuntimeError('Manifest was built with a different dataset/stream configuration')
    science = feature_science_contract(cfg)
    production = manifest_metadata['production'] and science['canonical']
    provenance = collect_provenance(policy=resolved['provenance'], require_clean=production)
    adapter = sl.ActivityNetAdapter(dataset_root=dataset_root)
    sources_by_split, dataset_identity = dataset_sources_and_identity(
        adapter, manifest_metadata, resolved['activitynet']['expected_source_counts'],
    )
    if manifest_metadata['frames_per_chunk'] != resolved['sequential']['frames_per_chunk']:
        raise RuntimeError('Manifest frames_per_chunk differs from the resolved stream config')
    checkpoint_id = resolved['encoder']['checkpoint_id']
    processor = AutoImageProcessor.from_pretrained(checkpoint_id)
    preprocessing = preprocessing_identity(
        processor, checkpoint_id, resolved['sequential']['frames_per_chunk'],
    )
    encoder, encoder_info = build_encoder(
        runtime['condition'], snapshot, manifest_metadata, resolved['encoder'], production,
        resolved['activitynet']['expected_source_counts']['training'],
    )
    feature_definition = {
        **resolved['linear_probe']['feature_definition'], 'size': resolved['encoder']['feature_size'],
    }
    device = torch.device(runtime['device'])
    shared_contract = {
        'schema': 'activitynet-shared-segment-feature-contract/v2',
        'dataset': dataset_identity,
        'manifest': {key: manifest_metadata[key] for key in (
            'manifest_id', 'segment_manifest_sha256', 'label_mapping_sha256')},
        'base_encoder': {'model': checkpoint_id, 'fingerprint': encoder_info['base_fingerprint']},
        'preprocessing': preprocessing, 'feature_definition': feature_definition,
        'extraction_provenance': extraction_provenance(provenance),
        'science': science,
    }
    shared_contract_sha = sha256_bytes(canonical_json_bytes(shared_contract))
    encoder_sha = sha256_bytes(canonical_json_bytes(encoder_info))
    feature_identity_sha = sha256_bytes(canonical_json_bytes({
        'condition': runtime['condition'], 'encoder_sha256': encoder_sha,
        'shared_feature_contract_sha256': shared_contract_sha,
    }))
    feature_id = runtime['feature_id'] or f'{runtime["manifest_id"]}__{feature_identity_sha[:12]}'
    directory = output_root / 'features' / runtime['condition'] / feature_id
    inputs = {
        'condition': runtime['condition'], 'feature_id': feature_id, 'production': production,
        'manifest': {key: manifest_metadata[key] for key in (
            'manifest_id', 'segment_manifest_sha256', 'label_mapping_sha256')},
        'encoder': encoder_info, 'encoder_sha256': encoder_sha,
        'feature_identity_sha256': feature_identity_sha,
        'feature_definition': feature_definition, 'preprocessing': preprocessing,
        'science': science,
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
        features.load_features(directory, rows, manifest_metadata, runtime['condition'])
        print(f'Reusing verified feature artifact: {directory}', flush=True)
        return existing

    # Extraction never trains: both encoders are fully frozen and in eval mode.
    encoder = encoder.to(device).eval().requires_grad_(False)
    experiment, comet = start_experiment(
        f'{resolved["linear_probe"]["id"]}__features__{runtime["condition"].replace("_", "-")}',
        inputs, tags=('linear-probe', 'features'), disabled=logging['disable_comet'],
        project_name=resolved['tracking']['comet_project'])
    splits = {}
    for split in manifest.SPLITS:
        split_rows = [row for row in rows if row['split'] == split]

        def progress(done, total, split=split):
            if done == total or done % logging['progress_interval_videos'] == 0:
                print(json.dumps({'event': 'feature_progress', 'split': split, 'videos': done, 'total': total}),
                      flush=True)
        splits[split] = features.extract_split(
            split_rows, sources_by_split[split], processor, encoder, device,
            resolved['encoder']['feature_size'], resolved['sequential']['frames_per_chunk'], progress,
        )
    metadata = features.write_feature_artifact(directory, splits, rows, {
        **inputs, 'counts': {split: len(splits[split]['segment_ids']) for split in manifest.SPLITS},
        'device': str(device), 'device_identity': device_identity(device), 'provenance': provenance,
        'created': utc_now(),
        'locations': {'dataset_root': str(dataset_root), 'manifest': str(manifest_dir.resolve()),
                      'snapshot': None if snapshot is None else str(snapshot.resolve()),
                      'artifact': str(directory.resolve())},
        'comet_experiment': comet,
    })
    artifact_name = resolved['tracking']['artifacts']['linear_probe_features'][runtime['condition']]
    status = log_artifact(experiment, directory, 'metadata.json', artifact_name, 'dataset',
                          metadata['files'], {'feature_id': feature_id, 'condition': runtime['condition'],
                                              'segment_manifest_sha256': manifest_metadata['segment_manifest_sha256'],
                                              'encoder_sha256': encoder_sha,
                                              'shared_feature_contract_sha256': shared_contract_sha},
                          aliases=(feature_id,), project_name=resolved['tracking']['comet_project'])
    end_experiment(experiment)
    print(json.dumps({'event': 'features_written', 'path': str(directory), 'files': metadata['files'],
                      'counts': metadata['counts'], 'comet': {**comet, **status}}), flush=True)
    return metadata


@hydra.main(version_base='1.3', config_path='../../conf', config_name='linear_probe_features')
def main(cfg: DictConfig):
    run(cfg)


if __name__ == '__main__':
    main()
