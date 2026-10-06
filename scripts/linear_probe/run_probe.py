"""Run the Linear Probe (Base ViT and final Query LoRA x seeds 0, 1, 2) and aggregate.

    python -m scripts.linear_probe.run_probe --manifest-id lp-v1 \
        --base-features log/linear_probe/features/base_vit/<id> \
        --lora-features log/linear_probe/features/moco_query_lora_final/<id>

Hyperparameters are fixed (see evaluation.linear_probe.HYPERPARAMETERS).
`--smoke-epochs` exists for plumbing checks only and marks results non-production.
Existing seed results are reused only when their inputs match exactly.
"""

import argparse
import json
from pathlib import Path

import torch

from evaluation import linear_probe as probe
from evaluation import activitynet_manifest as manifest
from evaluation import segment_features as features
from logger.comet_lineage import end_experiment, log_artifact, log_metrics, start_experiment
from training.moco_checkpoint import validate_production_snapshot_metadata
from utils.artifact_io import canonical_json_bytes, read_json, sha256_bytes, sha256_file
from utils.artifact_io import write_bytes_atomic, write_json_atomic
from utils.provenance import collect_provenance, utc_now


ARTIFACT_NAME = 'activitynet-linear-probe-results'
AGGREGATE_METADATA_SCHEMA = 'activitynet-linear-probe-aggregate-metadata/v1'
AGGREGATE_PAYLOAD_FILES = ('aggregate_summary.json', 'aggregate_summary.csv')


def aggregate_metadata(comparison, files):
    return {
        'schema': AGGREGATE_METADATA_SCHEMA, 'aggregate_schema': probe.AGGREGATE_SCHEMA,
        'files': files, 'local_path': str(comparison), 'created': utc_now(),
        'comet': {
            'status': 'pending', 'retry_needed': True, 'artifact_name': ARTIFACT_NAME,
            'artifact_type': 'results', 'aliases': [], 'files': files, 'time': utc_now(),
        },
    }


def run_seed(args, condition, seed, splits, feature_metadata, manifest_metadata, label_names, provenance,
             production, epochs):
    directory = args.output_root / 'results' / condition / f'seed-{seed}'
    inputs = {
        'schema': probe.RESULT_SCHEMA, 'protocol': probe.PROTOCOL, 'condition': condition, 'seed': seed,
        'production': production, 'hyperparameters': {**probe.HYPERPARAMETERS, 'epochs': epochs},
        'manifest': {key: manifest_metadata[key] for key in (
            'manifest_id', 'segment_manifest_sha256', 'label_mapping_sha256')}
        | {'comet_artifact_version': (manifest_metadata.get('comet') or {}).get('artifact_version')},
        'features': {'feature_id': feature_metadata['feature_id'], 'files': feature_metadata['files'],
                     'local_path': feature_metadata['local_path'], 'encoder_sha256': feature_metadata['encoder_sha256'],
                     'shared_feature_contract_sha256': feature_metadata['shared_feature_contract_sha256'],
                     'comet_artifact_version': (feature_metadata.get('comet') or {}).get('artifact_version'),
                     'upstream_experiment_key': (feature_metadata.get('comet') or {}).get('experiment_key')},
    }
    if directory.exists():
        summary = probe.load_summary(directory)
        if {key: summary.get(key) for key in inputs} != inputs:
            raise RuntimeError(f'Different probe result already exists: {directory}')
        print(f'Reusing verified probe result: {directory}', flush=True)
        return summary
    name = probe.experiment_name(condition, seed)
    experiment, comet = start_experiment(name, inputs, tags=('linear-probe', condition), disabled=args.disable_comet)
    train, validation = splits['training'], splits['validation']
    before = (train['features'].clone(), validation['features'].clone())
    device = torch.device(args.device)
    classifier, history = probe.train_probe(
        train['features'].to(device), train['labels'].to(device), seed, epochs,
        on_epoch=lambda row: log_metrics(experiment, {'train/loss': row['train_loss'], 'train/top1': row['train_top1']},
                                         step=row['epoch'], epoch=row['epoch']),
    )
    classifier = classifier.cpu()
    metrics = probe.evaluate(classifier, validation['features'], validation['labels'])
    if not (torch.equal(before[0], train['features']) and torch.equal(before[1], validation['features'])):
        raise RuntimeError('Frozen features changed during the probe')
    if production and metrics['macro_class_count'] != probe.CLASS_COUNT:
        raise RuntimeError('Production validation must cover all 200 classes')
    log_metrics(experiment, {'val/top1': metrics['top1'], 'val/macro_class_accuracy': metrics['macro_class_accuracy']},
                step=epochs)
    summary = probe.write_result(directory, classifier, history, metrics, {
        **inputs, 'top1': metrics['top1'], 'macro_class_accuracy': metrics['macro_class_accuracy'],
        'macro_class_count': metrics['macro_class_count'],
        'sample_counts': {split: len(splits[split]['segment_ids']) for split in splits},
        'final_epoch': epochs, 'validation_used_for_selection': False,
        'code': {key: provenance['implementation'].get(key)
                 for key in ('repository', 'branch', 'commit', 'dirty', 'tracked_diff_sha256')},
        'provenance': provenance, 'comet_experiment_key': comet.get('experiment_key'), 'comet': comet,
        'created': utc_now(), 'local_path': str(directory),
    }, label_names)
    end_experiment(experiment)
    print(json.dumps({'event': 'probe_result', 'condition': condition, 'seed': seed, 'top1': summary['top1'],
                      'macro_class_accuracy': summary['macro_class_accuracy'], 'path': str(directory)}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest-id', required=True)
    parser.add_argument('--base-features', type=Path, required=True)
    parser.add_argument('--lora-features', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output-root', type=Path, default=Path('log/linear_probe'))
    parser.add_argument('--smoke-epochs', type=int, help='Plumbing checks only; marks results non-production')
    parser.add_argument('--disable-comet', action='store_true')
    args = parser.parse_args()
    if args.smoke_epochs is not None and args.smoke_epochs < 1:
        parser.error('--smoke-epochs must be positive')
    print('Resolved config: ' + json.dumps({key: str(value) for key, value in vars(args).items()}), flush=True)

    rows, mapping, manifest_metadata = manifest.load_manifest(args.output_root / 'manifest' / args.manifest_id)
    label_names = {label_id: label for label, label_id in mapping.items()}
    loaded = {
        'base_vit': features.load_features(args.base_features, rows, manifest_metadata, 'base_vit'),
        'moco_query_lora_final': features.load_features(
            args.lora_features, rows, manifest_metadata, 'moco_query_lora_final'),
    }
    shared_contract = features.validate_feature_pair(loaded['base_vit'][1], loaded['moco_query_lora_final'][1])
    for split in manifest.SPLITS:
        if loaded['base_vit'][0][split]['segment_ids'] != loaded['moco_query_lora_final'][0][split]['segment_ids']:
            raise RuntimeError(f'Base / LoRA {split} segment IDs differ')
    snapshot = loaded['moco_query_lora_final'][1]['encoder']['snapshot']
    if manifest_metadata['production'] and args.smoke_epochs is None:
        validate_production_snapshot_metadata(
            snapshot,
            expected_source_sha256=manifest_metadata['splits']['training']['ordered_used_source_sha256'],
            expected_dataset_root=manifest_metadata['dataset']['root'],
            expected_base_fingerprint=shared_contract['base_encoder']['fingerprint'],
            expected_source_count=manifest.SPLIT_COUNTS['training'],
        )
    production = (manifest_metadata['production'] and args.smoke_epochs is None and snapshot['final'] is True
                  and all(metadata['production'] for _, metadata in loaded.values()))
    if manifest_metadata['production'] and args.smoke_epochs is None and not production:
        raise RuntimeError('Production probe requires production features from the final Query LoRA snapshot')
    epochs = args.smoke_epochs or probe.HYPERPARAMETERS['epochs']
    provenance = collect_provenance(require_clean=production)
    summaries = [
        run_seed(args, condition, seed, splits, feature_metadata, manifest_metadata, label_names, provenance,
                 production, epochs)
        for condition, (splits, feature_metadata) in loaded.items() for seed in probe.SEEDS
    ]

    comparison = args.output_root / 'results' / 'comparison'
    comparison.mkdir(parents=True, exist_ok=True)
    result = probe.aggregate(summaries)
    aggregate = {
        'schema': probe.AGGREGATE_SCHEMA, 'protocol': probe.PROTOCOL, 'production': production,
        'primary_metric': 'top1', 'secondary_metric': 'macro_class_accuracy', 'std': 'sample (ddof=1)',
        'conditions': result,
        'inputs': {condition: {'feature_id': metadata['feature_id'], 'files': metadata['files'],
                               'shared_feature_contract_sha256': metadata['shared_feature_contract_sha256']}
                   for condition, (_, metadata) in loaded.items()},
        'segment_manifest_sha256': manifest_metadata['segment_manifest_sha256'],
        'seed_results': {f'{row["condition"]}/seed-{row["seed"]}': row['files'] for row in summaries},
        'probe_experiment_keys': {probe.experiment_name(row['condition'], row['seed']): row['comet_experiment_key']
                                  for row in summaries},
        'moco_snapshot': snapshot, 'lineage_metadata': 'metadata.json', 'created': utc_now(),
    }
    identity = ('production', 'conditions', 'inputs', 'segment_manifest_sha256', 'seed_results')
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
            write_json_atomic(metadata_path, aggregate_metadata(comparison, files))
        print(f'Reusing identical aggregate result: {comparison}', flush=True)
        return
    if csv_path.exists() or metadata_path.exists():
        raise RuntimeError(f'Incomplete aggregate result already exists: {comparison}')
    write_json_atomic(summary_path, aggregate)
    write_bytes_atomic(csv_path, csv_bytes)
    files = {name: sha256_file(comparison / name) for name in AGGREGATE_PAYLOAD_FILES}
    write_json_atomic(metadata_path, aggregate_metadata(comparison, files))
    experiment, comet = start_experiment(f'{probe.PROTOCOL}__aggregate', {
        key: aggregate[key] for key in ('production', 'segment_manifest_sha256', 'probe_experiment_keys', 'inputs')
    }, tags=('linear-probe', 'aggregate'), disabled=args.disable_comet)
    for condition, values in result.items():
        log_metrics(experiment, {f'{condition}/top1_mean': values['top1_mean'],
                                 f'{condition}/macro_class_accuracy_mean': values['macro_class_accuracy_mean']})
    status = log_artifact(experiment, comparison, 'metadata.json', ARTIFACT_NAME, 'results', files,
                          {'production': production, 'segment_manifest_sha256': aggregate['segment_manifest_sha256'],
                           'aggregate_sha256': files['aggregate_summary.json'],
                           'inputs_sha256': sha256_bytes(canonical_json_bytes(aggregate['inputs']))})
    end_experiment(experiment)
    print(json.dumps({'event': 'aggregate', 'path': str(comparison), 'conditions': result,
                      'comet': {**comet, **status}}), flush=True)


if __name__ == '__main__':
    main()
