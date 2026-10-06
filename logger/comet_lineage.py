"""Comet experiments and artifacts as tracking / lineage over local artifacts.

Local files are the reproducible artifact. Comet calls happen only after a
local artifact is complete, re-verify local SHA-256 before every upload, and
report failures as status instead of raising or deleting local data. API keys
come from ./.comet.config and ~/.comet.config only; nothing here reads them.
"""

from pathlib import Path

from utils.artifact_io import read_json, sha256_file, write_json_atomic
from utils.provenance import COMET_PROJECT, utc_now


def scope_tag(production):
    """Comet scope tag; only an explicit bool decides production vs smoke."""
    if type(production) is not bool:
        raise TypeError(f'production must be bool, got {type(production).__name__}')
    return 'production' if production else 'smoke'


def display_tag(value):
    """Comet display token for a config / condition value; the value itself is unchanged."""
    return value.replace('_', '-')


def start_experiment(
    name, parameters, tags=(), disabled=False, existing_key=None, project_name=COMET_PROJECT,
):
    """Return (experiment or None, status). Never raises for network failures."""
    if disabled:
        return None, {'status': 'disabled'}
    try:
        import comet_ml
        if existing_key:
            experiment = comet_ml.ExistingExperiment(previous_experiment=existing_key)
        else:
            experiment = comet_ml.Experiment(project_name=project_name)
            experiment.set_name(name)
            for tag in tags:
                experiment.add_tag(tag)
        experiment.log_parameters(_flatten(parameters))
        return experiment, {'status': 'started', 'experiment_key': experiment.get_key(), 'experiment_name': name}
    except Exception as error:  # noqa: BLE001 - tracking must not stop local work
        return None, {'status': 'failed', 'error': repr(error), 'retry_needed': True}


def experiment_key(experiment):
    return None if experiment is None else experiment.get_key()


def log_metrics(experiment, metrics, step=None, epoch=None):
    if experiment is None:
        return
    try:
        experiment.log_metrics(metrics, step=step, epoch=epoch)
    except Exception as error:  # noqa: BLE001
        print(f'Comet metric logging failed: {error!r}', flush=True)


def end_experiment(experiment):
    if experiment is None:
        return None
    try:
        experiment.end()
        return None
    except Exception as error:  # noqa: BLE001
        return repr(error)


def _flatten(value, prefix=''):
    """Nested mappings become dotted Comet parameter names."""
    if isinstance(value, dict):
        flat = {}
        for key, item in value.items():
            flat.update(_flatten(item, f'{prefix}{key}.'))
        return flat
    name = prefix[:-1]
    return {name: value if isinstance(value, (bool, int, float, str)) or value is None else str(value)}


def artifact_files(directory, relative_paths):
    """Local file list with hashes, the identity that every upload re-verifies."""
    directory = Path(directory)
    return {path: sha256_file(directory / path) for path in relative_paths}


def log_artifact(
    experiment, directory, metadata_name, name, artifact_type, files, metadata, aliases=(),
    project_name=COMET_PROJECT,
):
    """Register local files as a new Comet artifact version and record status.

    `files` maps relative path -> expected SHA-256. Mismatches stop before any
    upload. The result is written into the local metadata JSON under `comet`.
    """
    directory = Path(directory)
    if metadata_name in files:
        raise ValueError('Mutable Comet metadata cannot also be an immutable artifact payload file')
    actual = artifact_files(directory, files)
    if actual != dict(files):
        raise RuntimeError(f'Local artifact hash changed before Comet upload: {directory}')
    status = {
        'artifact_name': name, 'artifact_type': artifact_type, 'aliases': list(aliases),
        'project_name': project_name, 'files': dict(files), 'time': utc_now(),
    }
    if experiment is None:
        status.update(status='disabled', retry_needed=True)
    else:
        try:
            import comet_ml
            artifact = comet_ml.Artifact(name, artifact_type, aliases=list(aliases), metadata=metadata)
            for path in files:
                artifact.add(str(directory / path), logical_path=path)
            logged = experiment.log_artifact(artifact)
            status.update(status='logged', artifact_version=str(logged.version),
                          experiment_key=experiment.get_key(), retry_needed=False)
        except Exception as error:  # noqa: BLE001
            status.update(status='failed', error=repr(error), retry_needed=True)
    record_comet_status(directory / metadata_name, status)
    return status


def record_comet_status(metadata_path, status):
    metadata = read_json(metadata_path)
    metadata['comet'] = {**metadata.get('comet', {}), **status}
    write_json_atomic(metadata_path, metadata)
    return metadata
