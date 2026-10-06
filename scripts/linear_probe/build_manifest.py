"""Build or audit the common ActivityNet annotation-segment manifest.

    python -m scripts.linear_probe.build_manifest runtime.command=build runtime.dataset_root=/path/to/ActivityNet
    python -m scripts.linear_probe.build_manifest runtime.command=audit runtime.dataset_root=/path/to/ActivityNet

`build` writes the manifest, label mapping and Gate checks 1-8 (status
PENDING_AUDIT or FAIL). `audit` regenerates the manifest from the same inputs,
compares SHA-256 (check 9), re-checks 1-8 from the saved files, and only then
marks the Gate PASS and registers the Comet artifact. Feature extraction
accepts PASS (production) or SMOKE_PASS (`runtime.max_videos_per_split`, smoke only,
where class coverage checks 1-4 are recorded but not required).
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import sequential_loader as sl

from evaluation import activitynet_manifest as manifest
from logger.comet_lineage import display_tag, end_experiment, log_artifact, scope_tag, start_experiment
from scripts.linear_probe.configuration import science_contract
from utils.artifact_io import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic
from utils.configuration import load_config_group, plain_config, validate_path_component
from utils.provenance import collect_provenance, utc_now


ARTIFACT_NAME = load_config_group('tracking', 'default')['artifacts']['linear_probe_manifest']
MANIFEST_SCIENCE_GROUPS = ('activitynet', 'sequential', 'provenance')


def selected_sources(dataset_root, max_videos, expected_counts):
    adapter = sl.ActivityNetAdapter(dataset_root=Path(dataset_root).resolve())
    full = {split: adapter.sequence_sources(split) for split in manifest.SPLITS}
    for split, count in expected_counts.items():
        if len(full[split]) != count:
            raise RuntimeError(f'Expected {count} {split} sources, got {len(full[split])}')
    used = {split: sources[:max_videos] if max_videos else sources for split, sources in full.items()}
    return full, used


def progress(split, done, total, interval):
    if done == total or done % interval == 0:
        print(json.dumps({'event': 'manifest_progress', 'split': split, 'videos': done, 'total': total}), flush=True)


def build(runtime, activitynet, sequential, science, directory, progress_interval=100):
    if directory.exists():
        raise FileExistsError(f'Manifest already exists; use audit to verify it: {directory}')
    production = runtime['max_videos_per_split'] is None and science['canonical']
    provenance = collect_provenance(
        policy=runtime['provenance_policy'], require_clean=production,
    )
    index = manifest.AnnotationIndex(activitynet['annotation_version'])
    full, used = selected_sources(
        runtime['dataset_root'], runtime['max_videos_per_split'], activitynet['expected_source_counts'],
    )
    mapping = manifest.canonical_label_mapping(index, full, activitynet['class_count'])
    rows, skips = manifest.build_manifest(
        index, used, mapping, sequential['frames_per_chunk'],
        lambda split, done, total: progress(split, done, total, progress_interval),
    )
    mapping_sha = sha256_bytes(canonical_json_bytes(mapping))
    gate = manifest.integrity_gate(
        rows, mapping, mapping_sha, mapping_sha, activitynet['class_count'],
    )
    status, failed = manifest.gate_status(gate, production, None)
    annotation_path = full['training'][0].evaluation_reference.annotation_path
    metadata = {
        'schema': manifest.MANIFEST_SCHEMA, 'manifest_id': runtime['manifest_id'], 'production': production,
        'scope': {'max_videos_per_split': runtime['max_videos_per_split']},
        'dataset': {'name': activitynet['name'], 'version': activitynet['version']},
        'annotation': {'sha256': sha256_file(annotation_path)},
        'splits': {split: {
            'source_count': len(full[split]), 'used_source_count': len(used[split]),
            'ordered_used_source_sha256': sha256_bytes(canonical_json_bytes(
                [source.sequence_id for source in used[split]])),
            'skip_counts': skips[split], 'skipped_total': sum(skips[split].values()),
        } for split in manifest.SPLITS},
        'class_count': len(mapping), 'frames_per_chunk': sequential['frames_per_chunk'],
        'chunk_rule': manifest.CHUNK_RULE,
        'row_order': 'split -> Adapter source order -> annotation_index',
        'label_mapping_sha256': mapping_sha,
        'segment_manifest_sha256': sha256_bytes(manifest.manifest_bytes(rows)),
        'gate': {**gate, 'status': status, 'failed': failed, 'reproducibility': None},
        'science': science,
        'provenance': provenance, 'created': utc_now(),
        'locations': {
            'dataset_root': str(Path(runtime['dataset_root']).resolve()),
            'annotation_path': str(Path(annotation_path).resolve()),
            'artifact': str(directory.resolve()),
        },
    }
    manifest.write_manifest(directory, rows, mapping, metadata)
    print(json.dumps({'event': 'manifest_built', 'path': str(directory), 'status': status, 'failed': failed,
                      'rows': gate['rows'], 'segment_manifest_sha256': metadata['segment_manifest_sha256'],
                      'label_mapping_sha256': mapping_sha, 'skipped': {
                          split: metadata['splits'][split]['skipped_total'] for split in manifest.SPLITS}}),
          flush=True)
    return status != 'FAIL'


def audit(runtime, activitynet, sequential, science, directory, progress_interval=100):
    rows, mapping, metadata = manifest.load_manifest(directory, require_gate=False)
    if metadata['gate']['status'] == 'FAIL':
        raise RuntimeError(f'Manifest Gate already failed: {metadata["gate"]["failed"]}')
    if metadata.get('science', {}).get('sha256') != science['sha256']:
        raise RuntimeError('Manifest was built with a different scientific config')
    audit_provenance = collect_provenance(
        policy=runtime['provenance_policy'], require_clean=metadata['production'],
    )
    index = manifest.AnnotationIndex(activitynet['annotation_version'])
    full, used = selected_sources(
        runtime['dataset_root'], metadata['scope']['max_videos_per_split'],
        activitynet['expected_source_counts'],
    )
    annotation_paths = {
        Path(source.evaluation_reference.annotation_path)
        for sources in full.values() for source in sources
    }
    if {sha256_file(path) for path in annotation_paths} != {metadata['annotation']['sha256']}:
        raise RuntimeError('ActivityNet annotation differs from the manifest')
    regenerated_mapping = manifest.canonical_label_mapping(index, full, activitynet['class_count'])
    regenerated, _ = manifest.build_manifest(
        index, used, regenerated_mapping, sequential['frames_per_chunk'],
        lambda split, done, total: progress(split, done, total, progress_interval),
    )
    regenerated_sha = sha256_bytes(manifest.manifest_bytes(regenerated))
    reproducible = (regenerated_sha == metadata['segment_manifest_sha256']
                    and sha256_bytes(canonical_json_bytes(regenerated_mapping)) == metadata['label_mapping_sha256'])
    gate = manifest.integrity_gate(rows, mapping, sha256_file(directory / manifest.MAPPING_FILE),
                                   metadata['label_mapping_sha256'], activitynet['class_count'])
    status, failed = manifest.gate_status(gate, metadata['production'], reproducible)
    metadata['gate'] = {**gate, 'status': status, 'failed': failed, 'reproducibility': {
        'regenerated_sha256': regenerated_sha, 'matches': reproducible, 'time': utc_now(),
        'provenance': audit_provenance,
        'locations': {'dataset_root': str(Path(runtime['dataset_root']).resolve())}}}
    write_json_atomic(directory / 'metadata.json', metadata)
    print(json.dumps({'event': 'manifest_audit', 'path': str(directory), 'status': status, 'failed': failed,
                      'segment_manifest_sha256': metadata['segment_manifest_sha256']}), flush=True)
    if status not in ('PASS', 'SMOKE_PASS'):
        return False
    experiment, comet = start_experiment(
        f'{runtime["protocol"]}__manifest__{runtime["manifest_id"]}', {
            'manifest_id': runtime['manifest_id'], 'production': metadata['production'],
            'segment_manifest_sha256': metadata['segment_manifest_sha256'],
            'label_mapping_sha256': metadata['label_mapping_sha256']},
        tags=('linear-probe', 'manifest', scope_tag(metadata['production']), display_tag(runtime['protocol'])),
        disabled=runtime['disable_comet'], project_name=runtime['tracking']['comet_project'])
    files = {name: sha256_file(directory / name) for name in (manifest.MANIFEST_FILE, manifest.MAPPING_FILE)}
    artifact_name = runtime['tracking']['artifacts']['linear_probe_manifest']
    status = log_artifact(experiment, directory, 'metadata.json', artifact_name, 'dataset', files, {
        'manifest_id': runtime['manifest_id'], 'production': metadata['production'],
        'segment_manifest_sha256': metadata['segment_manifest_sha256'],
        'label_mapping_sha256': metadata['label_mapping_sha256']}, aliases=(runtime['manifest_id'],),
        project_name=runtime['tracking']['comet_project'])
    end_experiment(experiment)
    print(f'Comet: {json.dumps({**comet, **status})}', flush=True)
    return True


def run(cfg):
    resolved = plain_config(cfg)
    runtime = resolved['runtime']
    if runtime['command'] not in ('build', 'audit'):
        raise ValueError('runtime.command must be build or audit')
    validate_path_component('runtime.manifest_id', runtime['manifest_id'])
    if runtime['max_videos_per_split'] is not None and runtime['max_videos_per_split'] < 1:
        raise ValueError('runtime.max_videos_per_split must be positive')
    logging = resolved['logging']
    if type(logging['progress_interval_videos']) is not int or logging['progress_interval_videos'] < 1:
        raise ValueError('logging.progress_interval_videos must be a positive integer')
    if type(logging['disable_comet']) is not bool:
        raise ValueError('logging.disable_comet must be boolean')
    if type(resolved['sequential']['frames_per_chunk']) is not int or resolved['sequential']['frames_per_chunk'] < 1:
        raise ValueError('sequential.frames_per_chunk must be a positive integer')
    science = science_contract(cfg, MANIFEST_SCIENCE_GROUPS)
    runtime = {
        **runtime, 'disable_comet': logging['disable_comet'],
        'provenance_policy': resolved['provenance'], 'tracking': resolved['tracking'],
        'protocol': resolved['linear_probe']['id'],
    }
    directory = Path(runtime['output_root']) / 'manifest' / runtime['manifest_id']
    print('Resolved config:\n' + OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    action = build if runtime['command'] == 'build' else audit
    passed = action(
        runtime, resolved['activitynet'], resolved['sequential'], science, directory,
        logging['progress_interval_videos'],
    )
    if not passed:
        raise SystemExit(1)


@hydra.main(version_base='1.3', config_path='../../conf', config_name='linear_probe_manifest')
def main(cfg: DictConfig):
    run(cfg)


if __name__ == '__main__':
    main()
