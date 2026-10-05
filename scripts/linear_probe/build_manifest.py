"""Build or audit the common ActivityNet annotation-segment manifest.

    python -m scripts.linear_probe.build_manifest build /path/to/ActivityNet --manifest-id lp-v1
    python -m scripts.linear_probe.build_manifest audit /path/to/ActivityNet --manifest-id lp-v1

`build` writes the manifest, label mapping and Gate checks 1-8 (status
PENDING_AUDIT or FAIL). `audit` regenerates the manifest from the same inputs,
compares SHA-256 (check 9), re-checks 1-8 from the saved files, and only then
marks the Gate PASS and registers the Comet artifact. Feature extraction
accepts PASS (production) or SMOKE_PASS (`--max-videos-per-split`, smoke only,
where class coverage checks 1-4 are recorded but not required).
"""

import argparse
import json
from pathlib import Path
import sys

import sequential_loader as sl

from evaluation import activitynet_manifest as manifest
from logger.comet_lineage import end_experiment, log_artifact, start_experiment
from utils.artifact_io import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic
from utils.provenance import collect_provenance, utc_now


ARTIFACT_NAME = 'activitynet-linear-probe-manifest'


def selected_sources(dataset_root, max_videos):
    adapter = sl.ActivityNetAdapter(dataset_root=dataset_root)
    full = {split: adapter.sequence_sources(split) for split in manifest.SPLITS}
    for split, count in manifest.SPLIT_COUNTS.items():
        if len(full[split]) != count:
            raise RuntimeError(f'Expected {count} {split} sources, got {len(full[split])}')
    used = {split: sources[:max_videos] if max_videos else sources for split, sources in full.items()}
    return full, used


def progress(split, done, total):
    if done == total or done % 100 == 0:
        print(json.dumps({'event': 'manifest_progress', 'split': split, 'videos': done, 'total': total}), flush=True)


def build(args, directory):
    if directory.exists():
        raise FileExistsError(f'Manifest already exists; use audit to verify it: {directory}')
    provenance = collect_provenance()
    index = manifest.AnnotationIndex()
    full, used = selected_sources(args.dataset_root, args.max_videos_per_split)
    mapping = manifest.canonical_label_mapping(index, full)
    rows, skips = manifest.build_manifest(index, used, mapping, progress)
    mapping_sha = sha256_bytes(canonical_json_bytes(mapping))
    gate = manifest.integrity_gate(rows, mapping, mapping_sha, mapping_sha)
    production = args.max_videos_per_split is None
    status, failed = manifest.gate_status(gate, production, None)
    annotation_path = full['training'][0].evaluation_reference.annotation_path
    metadata = {
        'schema': manifest.MANIFEST_SCHEMA, 'manifest_id': args.manifest_id, 'production': production,
        'scope': {'max_videos_per_split': args.max_videos_per_split},
        'dataset': {'name': 'ActivityNet', 'version': '1.3', 'root': str(args.dataset_root)},
        'annotation': {'path': str(annotation_path), 'sha256': sha256_file(annotation_path)},
        'splits': {split: {
            'source_count': len(full[split]), 'used_source_count': len(used[split]),
            'ordered_used_source_sha256': sha256_bytes(canonical_json_bytes(
                [source.sequence_id for source in used[split]])),
            'skip_counts': skips[split], 'skipped_total': sum(skips[split].values()),
        } for split in manifest.SPLITS},
        'class_count': len(mapping), 'frames_per_chunk': 16, 'chunk_rule': manifest.CHUNK_RULE,
        'row_order': 'split -> Adapter source order -> annotation_index',
        'label_mapping_sha256': mapping_sha,
        'segment_manifest_sha256': sha256_bytes(manifest.manifest_bytes(rows)),
        'gate': {**gate, 'status': status, 'failed': failed, 'reproducibility': None},
        'provenance': provenance, 'created': utc_now(), 'local_path': str(directory),
    }
    manifest.write_manifest(directory, rows, mapping, metadata)
    print(json.dumps({'event': 'manifest_built', 'path': str(directory), 'status': status, 'failed': failed,
                      'rows': gate['rows'], 'segment_manifest_sha256': metadata['segment_manifest_sha256'],
                      'label_mapping_sha256': mapping_sha, 'skipped': {
                          split: metadata['splits'][split]['skipped_total'] for split in manifest.SPLITS}}),
          flush=True)
    return status != 'FAIL'


def audit(args, directory):
    rows, mapping, metadata = manifest.load_manifest(directory, require_gate=False)
    if metadata['gate']['status'] == 'FAIL':
        raise RuntimeError(f'Manifest Gate already failed: {metadata["gate"]["failed"]}')
    index = manifest.AnnotationIndex()
    full, used = selected_sources(args.dataset_root, metadata['scope']['max_videos_per_split'])
    regenerated_mapping = manifest.canonical_label_mapping(index, full)
    regenerated, _ = manifest.build_manifest(index, used, regenerated_mapping, progress)
    regenerated_sha = sha256_bytes(manifest.manifest_bytes(regenerated))
    reproducible = (regenerated_sha == metadata['segment_manifest_sha256']
                    and sha256_bytes(canonical_json_bytes(regenerated_mapping)) == metadata['label_mapping_sha256'])
    gate = manifest.integrity_gate(rows, mapping, sha256_file(directory / manifest.MAPPING_FILE),
                                   metadata['label_mapping_sha256'])
    status, failed = manifest.gate_status(gate, metadata['production'], reproducible)
    metadata['gate'] = {**gate, 'status': status, 'failed': failed, 'reproducibility': {
        'regenerated_sha256': regenerated_sha, 'matches': reproducible, 'time': utc_now()}}
    write_json_atomic(directory / 'metadata.json', metadata)
    print(json.dumps({'event': 'manifest_audit', 'path': str(directory), 'status': status, 'failed': failed,
                      'segment_manifest_sha256': metadata['segment_manifest_sha256']}), flush=True)
    if status not in ('PASS', 'SMOKE_PASS'):
        return False
    experiment, comet = start_experiment(
        f'{args.manifest_id}__manifest', {'manifest_id': args.manifest_id, 'production': metadata['production'],
                                          'segment_manifest_sha256': metadata['segment_manifest_sha256'],
                                          'label_mapping_sha256': metadata['label_mapping_sha256']},
        tags=('linear-probe', 'manifest'), disabled=args.disable_comet)
    files = {name: sha256_file(directory / name) for name in (manifest.MANIFEST_FILE, manifest.MAPPING_FILE)}
    status = log_artifact(experiment, directory, 'metadata.json', ARTIFACT_NAME, 'dataset', files, {
        'manifest_id': args.manifest_id, 'production': metadata['production'],
        'segment_manifest_sha256': metadata['segment_manifest_sha256'],
        'label_mapping_sha256': metadata['label_mapping_sha256']}, aliases=(args.manifest_id,))
    end_experiment(experiment)
    print(f'Comet: {json.dumps({**comet, **status})}', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'audit'))
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--manifest-id', required=True)
    parser.add_argument('--max-videos-per-split', type=int, help='Smoke only: first N Adapter sources per split')
    parser.add_argument('--output-root', type=Path, default=Path('log/linear_probe'))
    parser.add_argument('--disable-comet', action='store_true')
    args = parser.parse_args()
    if args.max_videos_per_split is not None and args.max_videos_per_split < 1:
        parser.error('--max-videos-per-split must be positive')
    directory = args.output_root / 'manifest' / args.manifest_id
    print('Resolved config: ' + json.dumps({key: str(value) for key, value in vars(args).items()}), flush=True)
    passed = build(args, directory) if args.command == 'build' else audit(args, directory)
    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
