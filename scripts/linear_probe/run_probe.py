"""Run the Linear Probe (Base ViT and final Query LoRA x seeds 0, 1, 2) and aggregate.

    python -m scripts.linear_probe.run_probe \
        runtime.base_features=log/linear_probe/features/base_vit/<id> \
        runtime.lora_features=log/linear_probe/features/moco_query_lora_final/<id>

Hyperparameters come from the versioned Hydra Linear Probe preset.
Use `linear_probe.probe.epochs=<N>` for a short plumbing check; any scientific
override is recorded and marks the result non-production.
Existing seed results are reused only when their inputs match exactly.
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from evaluation import linear_probe as probe
from evaluation import activitynet_manifest as manifest
from evaluation import segment_features as features
from logger.comet_lineage import display_tag, end_experiment, log_artifact, log_metrics, scope_tag, start_experiment
from scripts.linear_probe.configuration import feature_science_contract, science_contract
from training.moco_checkpoint import validate_production_snapshot_metadata
from utils.artifact_io import canonical_json_bytes, read_json, sha256_bytes, sha256_file
from utils.artifact_io import write_bytes_atomic, write_json_atomic
from utils.configuration import plain_config, validate_path_component
from utils.provenance import collect_provenance, utc_now


AGGREGATE_METADATA_SCHEMA = 'activitynet-linear-probe-aggregate-metadata/v2'
AGGREGATE_PAYLOAD_FILES = ('aggregate_summary.json', 'aggregate_summary.csv')
MANIFEST_SCIENCE_GROUPS = ('activitynet', 'sequential', 'provenance')


def aggregate_metadata(comparison, files, artifact_name, project_name):
    return {
        'schema': AGGREGATE_METADATA_SCHEMA, 'aggregate_schema': probe.AGGREGATE_SCHEMA,
        'files': files, 'locations': {'artifact': str(comparison.resolve())}, 'created': utc_now(),
        'comet': {
            'status': 'pending', 'retry_needed': True, 'artifact_name': artifact_name,
            'artifact_type': 'results', 'aliases': [], 'project_name': project_name,
            'files': files, 'time': utc_now(),
        },
    }


def run_seed(
    runtime, config, condition, seed, splits, feature_metadata, feature_path, manifest_metadata, label_names,
    provenance, production, class_count, science, project_name,
):
    directory = Path(runtime['result_root']) / condition / f'seed-{seed}'
    feature_size = features.feature_size_from_metadata(feature_metadata)
    inputs = {
        'schema': probe.RESULT_SCHEMA, 'protocol': config.protocol, 'condition': condition, 'seed': seed,
        'production': production, 'hyperparameters': config.hyperparameters(feature_size, class_count),
        'result_identity_sha256': runtime['result_identity_sha256'],
        'science': science,
        'manifest': {key: manifest_metadata[key] for key in (
            'manifest_id', 'segment_manifest_sha256', 'label_mapping_sha256')},
        'features': {'feature_id': feature_metadata['feature_id'], 'files': feature_metadata['files'],
                     'encoder_sha256': feature_metadata['encoder_sha256'],
                     'shared_feature_contract_sha256': feature_metadata['shared_feature_contract_sha256'],
                     },
    }
    if directory.exists():
        summary = probe.load_summary(directory)
        if {key: summary.get(key) for key in inputs} != inputs:
            raise RuntimeError(f'Different probe result already exists: {directory}')
        print(f'Reusing verified probe result: {directory}', flush=True)
        return summary
    name = probe.experiment_name(config.protocol, condition, seed)
    experiment, comet = start_experiment(
        name, inputs, tags=('linear-probe', 'probe', display_tag(condition), f'seed-{seed}', scope_tag(production),
                            display_tag(config.protocol)),
        disabled=runtime['disable_comet'], project_name=project_name,
    )
    train, validation = splits['training'], splits['validation']
    before = (train['features'].clone(), validation['features'].clone())
    device = torch.device(runtime['device'])
    classifier, history = probe.train_probe(
        train['features'].to(device), train['labels'].to(device), seed, config, class_count,
        on_epoch=lambda row: log_metrics(experiment, {'train/loss': row['train_loss'], 'train/top1': row['train_top1']},
                                         step=row['epoch'], epoch=row['epoch']),
    )
    classifier = classifier.cpu()
    metrics = probe.evaluate(classifier, validation['features'], validation['labels'])
    if not (torch.equal(before[0], train['features']) and torch.equal(before[1], validation['features'])):
        raise RuntimeError('Frozen features changed during the probe')
    if production and metrics['macro_class_count'] != class_count:
        raise RuntimeError(f'Production validation must cover all {class_count} classes')
    log_metrics(experiment, {'val/top1': metrics['top1'], 'val/macro_class_accuracy': metrics['macro_class_accuracy']},
                step=config.epochs)
    summary = probe.write_result(directory, classifier, history, metrics, {
        **inputs, 'top1': metrics['top1'], 'macro_class_accuracy': metrics['macro_class_accuracy'],
        'macro_class_count': metrics['macro_class_count'],
        'sample_counts': {split: len(splits[split]['segment_ids']) for split in splits},
        'final_epoch': config.epochs, 'validation_used_for_selection': False,
        'code': {key: provenance['implementation'].get(key)
                 for key in ('repository', 'branch', 'commit', 'dirty', 'tracked_diff_sha256')},
        'provenance': provenance, 'comet_experiment_key': comet.get('experiment_key'), 'comet': comet,
        'created': utc_now(),
        'locations': {'feature_artifact': str(Path(feature_path).resolve()),
                      'result_artifact': str(directory.resolve())},
    }, label_names)
    end_experiment(experiment)
    print(json.dumps({'event': 'probe_result', 'condition': condition, 'seed': seed, 'top1': summary['top1'],
                      'macro_class_accuracy': summary['macro_class_accuracy'], 'path': str(directory)}), flush=True)
    return summary


def run(cfg):
    resolved = plain_config(cfg)
    runtime = resolved['runtime']
    logging = resolved['logging']
    if type(logging['disable_comet']) is not bool:
        raise ValueError('logging.disable_comet must be boolean')
    validate_path_component('runtime.manifest_id', runtime['manifest_id'])
    if runtime['result_id'] is not None:
        validate_path_component('runtime.result_id', runtime['result_id'])
    if runtime['device'] not in ('cpu', 'cuda'):
        raise ValueError('runtime.device must be cpu or cuda')
    if runtime['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    print('Resolved config:\n' + OmegaConf.to_yaml(cfg, resolve=True), flush=True)

    output_root = Path(runtime['output_root'])
    rows, mapping, manifest_metadata = manifest.load_manifest(
        output_root / 'manifest' / runtime['manifest_id'],
    )
    manifest_science = science_contract(cfg, MANIFEST_SCIENCE_GROUPS)
    if (manifest_metadata.get('science') or {}).get('sha256') != manifest_science['sha256']:
        raise RuntimeError('Manifest was built with a different dataset/stream configuration')
    label_names = {label_id: label for label, label_id in mapping.items()}
    class_count = len(mapping)
    base_features = Path(runtime['base_features'])
    lora_features = Path(runtime['lora_features'])
    feature_paths = {
        'base_vit': base_features,
        'moco_query_lora_final': lora_features,
    }
    loaded = {
        'base_vit': features.load_features(base_features, rows, manifest_metadata, 'base_vit'),
        'moco_query_lora_final': features.load_features(
            lora_features, rows, manifest_metadata, 'moco_query_lora_final'),
    }
    feature_science = feature_science_contract(cfg)
    for _, metadata in loaded.values():
        if (metadata.get('science') or {}).get('sha256') != feature_science['sha256']:
            raise RuntimeError('Feature artifact was built with a different feature configuration')
    shared_contract = features.validate_feature_pair(loaded['base_vit'][1], loaded['moco_query_lora_final'][1])
    for split in manifest.SPLITS:
        if loaded['base_vit'][0][split]['segment_ids'] != loaded['moco_query_lora_final'][0][split]['segment_ids']:
            raise RuntimeError(f'Base / LoRA {split} segment IDs differ')
    snapshot = loaded['moco_query_lora_final'][1]['encoder']['snapshot']
    probe_config = probe.ProbeConfig.from_mapping(
        resolved['linear_probe']['id'], resolved['linear_probe']['probe'],
    )
    probe_science = science_contract(cfg)
    result_identity = {
        'science_sha256': probe_science['sha256'],
        'manifest': {key: manifest_metadata[key] for key in (
            'segment_manifest_sha256', 'label_mapping_sha256',
        )},
        'features': {
            condition: {
                'feature_id': metadata['feature_id'], 'encoder_sha256': metadata['encoder_sha256'],
                'files': metadata['files'],
            }
            for condition, (_, metadata) in loaded.items()
        },
    }
    result_identity_sha = sha256_bytes(canonical_json_bytes(result_identity))
    result_id = runtime['result_id'] or f'{probe_config.protocol}__{result_identity_sha[:12]}'
    validate_path_component('resolved result_id', result_id)
    runtime = {
        **runtime, 'disable_comet': logging['disable_comet'],
        'result_id': result_id, 'result_identity_sha256': result_identity_sha,
        'result_root': str(output_root / 'results' / result_id),
    }
    production = (
        manifest_metadata['production'] and probe_science['canonical']
        and feature_science['canonical'] and snapshot['final'] is True
        and all(metadata['production'] for _, metadata in loaded.values())
        and class_count == resolved['activitynet']['class_count']
    )
    if production:
        validate_production_snapshot_metadata(
            snapshot,
            expected_source_sha256=manifest_metadata['splits']['training']['ordered_used_source_sha256'],
            expected_base_fingerprint=shared_contract['base_encoder']['fingerprint'],
            expected_source_count=resolved['activitynet']['expected_source_counts']['training'],
        )
    provenance = collect_provenance(policy=resolved['provenance'], require_clean=production)
    summaries = [
        run_seed(
            runtime, probe_config, condition, seed, splits, feature_metadata, feature_paths[condition],
            manifest_metadata,
            label_names, provenance, production, class_count, probe_science,
            resolved['tracking']['comet_project'],
        )
        for condition, (splits, feature_metadata) in loaded.items() for seed in probe_config.seeds
    ]

    comparison = Path(runtime['result_root']) / 'comparison'
    comparison.mkdir(parents=True, exist_ok=True)
    result = probe.aggregate(summaries)
    aggregate = {
        'schema': probe.AGGREGATE_SCHEMA, 'protocol': probe_config.protocol, 'production': production,
        'science': probe_science, 'result_id': result_id,
        'result_identity_sha256': result_identity_sha,
        'primary_metric': 'top1', 'secondary_metric': 'macro_class_accuracy', 'std': 'sample (ddof=1)',
        'conditions': result,
        'inputs': {condition: {'feature_id': metadata['feature_id'], 'files': metadata['files'],
                               'shared_feature_contract_sha256': metadata['shared_feature_contract_sha256']}
                   for condition, (_, metadata) in loaded.items()},
        'segment_manifest_sha256': manifest_metadata['segment_manifest_sha256'],
        'seed_results': {f'{row["condition"]}/seed-{row["seed"]}': row['files'] for row in summaries},
        'probe_experiment_keys': {
            probe.experiment_name(probe_config.protocol, row['condition'], row['seed']): row['comet_experiment_key']
            for row in summaries
        },
        'moco_snapshot': snapshot, 'lineage_metadata': 'metadata.json', 'created': utc_now(),
    }
    identity = (
        'production', 'science', 'result_id', 'result_identity_sha256', 'conditions', 'inputs',
        'segment_manifest_sha256', 'seed_results',
    )
    summary_path = comparison / 'aggregate_summary.json'
    csv_path = comparison / 'aggregate_summary.csv'
    metadata_path = comparison / 'metadata.json'
    csv_bytes = probe.aggregate_csv(result)
    if summary_path.exists():
        if not csv_path.is_file():
            raise RuntimeError(f'Aggregate CSV is missing: {csv_path}')
        existing = read_json(summary_path)
        if existing.get('schema') != probe.AGGREGATE_SCHEMA or 'comet' in existing:
            raise RuntimeError(f'Unsupported or mutable aggregate summary: {summary_path}')
        if {key: existing.get(key) for key in identity} != {key: aggregate[key] for key in identity}:
            raise RuntimeError(f'Different aggregate result already exists: {comparison}')
        if csv_path.read_bytes() != csv_bytes:
            raise RuntimeError(f'Aggregate CSV differs from the seed results: {csv_path}')
        files = {name: sha256_file(comparison / name) for name in AGGREGATE_PAYLOAD_FILES}
        if metadata_path.exists():
            sidecar = read_json(metadata_path)
            if sidecar.get('schema') != AGGREGATE_METADATA_SCHEMA or sidecar.get('files') != files or (
                sidecar.get('comet') or {}
            ).get('files') != files:
                raise RuntimeError(f'Aggregate lineage metadata differs from payload hashes: {metadata_path}')
        else:
            write_json_atomic(metadata_path, aggregate_metadata(
                comparison, files, resolved['tracking']['artifacts']['linear_probe_results'],
                resolved['tracking']['comet_project'],
            ))
        print(f'Reusing identical aggregate result: {comparison}', flush=True)
        return existing
    if csv_path.exists() or metadata_path.exists():
        raise RuntimeError(f'Incomplete aggregate result already exists: {comparison}')
    write_json_atomic(summary_path, aggregate)
    write_bytes_atomic(csv_path, csv_bytes)
    files = {name: sha256_file(comparison / name) for name in AGGREGATE_PAYLOAD_FILES}
    artifact_name = resolved['tracking']['artifacts']['linear_probe_results']
    write_json_atomic(metadata_path, aggregate_metadata(
        comparison, files, artifact_name, resolved['tracking']['comet_project'],
    ))
    experiment, comet = start_experiment(f'{probe_config.protocol}__aggregate', {
        key: aggregate[key] for key in ('production', 'segment_manifest_sha256', 'probe_experiment_keys', 'inputs')
    }, tags=('linear-probe', 'aggregate', scope_tag(production), display_tag(probe_config.protocol)),
        disabled=runtime['disable_comet'], project_name=resolved['tracking']['comet_project'])
    for condition, values in result.items():
        log_metrics(experiment, {f'{condition}/top1_mean': values['top1_mean'],
                                 f'{condition}/macro_class_accuracy_mean': values['macro_class_accuracy_mean']})
    status = log_artifact(experiment, comparison, 'metadata.json', artifact_name, 'results', files,
                          {'production': production, 'segment_manifest_sha256': aggregate['segment_manifest_sha256'],
                           'aggregate_sha256': files['aggregate_summary.json'],
                           'inputs_sha256': sha256_bytes(canonical_json_bytes(aggregate['inputs']))},
                          project_name=resolved['tracking']['comet_project'])
    end_experiment(experiment)
    print(json.dumps({'event': 'aggregate', 'path': str(comparison), 'conditions': result,
                      'comet': {**comet, **status}}), flush=True)
    return aggregate


@hydra.main(version_base='1.3', config_path='../../conf', config_name='linear_probe_run')
def main(cfg: DictConfig):
    run(cfg)


if __name__ == '__main__':
    main()
