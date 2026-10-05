"""Comet tracking never replaces or deletes local artifacts."""

import json
from unittest.mock import Mock

import comet_ml
import pytest

from logger import comet_lineage as lineage
from utils.artifact_io import sha256_file


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / 'weights.bin').write_bytes(b'weights')
    (tmp_path / 'metadata.json').write_text(json.dumps({'name': 'local'}))
    return tmp_path, {'weights.bin': sha256_file(tmp_path / 'weights.bin')}


def test_disabled_comet_records_retry_and_keeps_files(artifact):
    directory, files = artifact
    status = lineage.log_artifact(None, directory, 'metadata.json', 'a', 'model', files, {}, aliases=('x',))
    assert status['status'] == 'disabled' and status['retry_needed']
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['name'] == 'local' and metadata['comet']['files'] == files
    assert (directory / 'weights.bin').read_bytes() == b'weights'
    assert lineage.start_experiment('n', {}, disabled=True) == (None, {'status': 'disabled'})


def test_upload_failure_is_recorded_not_raised(artifact):
    directory, files = artifact
    experiment = Mock()
    experiment.log_artifact.side_effect = ConnectionError('offline')
    status = lineage.log_artifact(experiment, directory, 'metadata.json', 'a', 'model', files, {})
    assert status['status'] == 'failed' and status['retry_needed'] and 'offline' in status['error']
    assert (directory / 'weights.bin').exists()


def test_successful_upload_records_version_and_experiment(artifact):
    directory, files = artifact
    experiment = Mock()
    experiment.log_artifact.return_value = Mock(version='1.2.0')
    experiment.get_key.return_value = 'key'
    status = lineage.log_artifact(experiment, directory, 'metadata.json', 'a', 'model', files, {'k': 1},
                                  aliases=('final',))
    assert status['status'] == 'logged' and status['artifact_version'] == '1.2.0'
    assert status['experiment_key'] == 'key' and not status['retry_needed']
    logged = experiment.log_artifact.call_args.args[0]
    assert isinstance(logged, comet_ml.Artifact) and logged.name == 'a'


def test_changed_local_file_blocks_upload(artifact):
    directory, files = artifact
    (directory / 'weights.bin').write_bytes(b'changed')
    experiment = Mock()
    with pytest.raises(RuntimeError, match='hash changed'):
        lineage.log_artifact(experiment, directory, 'metadata.json', 'a', 'model', files, {})
    experiment.log_artifact.assert_not_called()


def test_experiment_start_failure_returns_status(monkeypatch):
    monkeypatch.setattr(comet_ml, 'Experiment', Mock(side_effect=ValueError('no key')))
    experiment, status = lineage.start_experiment('n', {'a': {'b': 1}})
    assert experiment is None and status['status'] == 'failed' and status['retry_needed']


def test_nested_parameters_are_flattened():
    assert lineage._flatten({'a': {'b': 1, 'c': [1]}, 'd': None}) == {'a.b': 1, 'a.c': '[1]', 'd': None}
