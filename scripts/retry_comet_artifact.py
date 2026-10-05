"""Retry a failed or disabled Comet artifact registration for a local artifact.

    python -m scripts.retry_comet_artifact log/linear_probe/manifest/lp-v1
    python -m scripts.retry_comet_artifact log/linear_probe/results/comparison --metadata-name aggregate_summary.json

Uses the artifact name, type, aliases and file hashes recorded under `comet` in
the local metadata. Local SHA-256 is re-verified before upload; a mismatch stops.
"""

import argparse
import json
from pathlib import Path

from logger.comet_lineage import end_experiment, log_artifact, start_experiment
from utils.artifact_io import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--metadata-name', default='metadata.json')
    args = parser.parse_args()
    metadata = read_json(args.directory / args.metadata_name)
    comet = metadata.get('comet') or {}
    if not comet.get('retry_needed'):
        parser.error(f'No pending Comet registration in {args.directory / args.metadata_name}')
    # Attach to the experiment that produced the artifact when it exists, else a new retry experiment.
    upstream = (comet.get('experiment_key') or metadata.get('moco_experiment_key')
                or (metadata.get('comet_experiment') or {}).get('experiment_key'))
    experiment, status = start_experiment(f'retry__{comet["artifact_name"]}', {'retry_of': str(args.directory)},
                                          tags=('retry',), existing_key=upstream)
    result = log_artifact(experiment, args.directory, args.metadata_name, comet['artifact_name'],
                          comet['artifact_type'], comet['files'],
                          {'retry': True, 'local_path': str(args.directory)}, aliases=comet.get('aliases', ()))
    end_experiment(experiment)
    print(json.dumps({'experiment': status, 'artifact': result}), flush=True)


if __name__ == '__main__':
    main()
