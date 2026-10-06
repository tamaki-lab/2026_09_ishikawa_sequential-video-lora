from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from scripts.linear_probe.configuration import feature_science_contract, science_contract
from training.moco_config import FullMoCoConfig
from utils.configuration import plain_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "conf"

ROOT_CONFIGS = (
    "moco_full",
    "linear_probe_manifest",
    "linear_probe_features",
    "linear_probe_run",
)

REQUIRED_OVERRIDES = {
    "moco_full": (
        "runtime.dataset_root=/datasets/activitynet",
        "runtime.run_id=test-run",
        "runtime.seed=7",
    ),
    "linear_probe_manifest": (
        "runtime.command=build",
        "runtime.dataset_root=/datasets/activitynet",
    ),
    "linear_probe_features": (
        "runtime.dataset_root=/datasets/activitynet",
        "runtime.condition=base_vit",
    ),
    "linear_probe_run": (
        "runtime.base_features=/artifacts/base",
        "runtime.lora_features=/artifacts/lora",
    ),
}

MISSING_KEYS = {
    "moco_full": {
        "runtime.dataset_root",
        "runtime.run_id",
        "runtime.seed",
    },
    "linear_probe_manifest": {
        "runtime.command",
        "runtime.dataset_root",
    },
    "linear_probe_features": {
        "runtime.condition",
        "runtime.dataset_root",
    },
    "linear_probe_run": {
        "runtime.base_features",
        "runtime.lora_features",
    },
}


def compose_pipeline(config_name, overrides=(), *, return_hydra_config=False):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        return compose(
            config_name=config_name,
            overrides=list(overrides),
            return_hydra_config=return_hydra_config,
        )


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_pipeline_roots_compose_shared_science_defaults(config_name):
    cfg = compose_pipeline(config_name)

    assert cfg.activitynet.name == "ActivityNet"
    assert cfg.activitynet.version == "1.3"
    assert cfg.activitynet.expected_source_counts == {
        "training": 10_024,
        "validation": 4_926,
    }
    assert cfg.encoder.checkpoint_id == "google/vit-base-patch16-224"
    assert cfg.encoder.feature_size == 768
    assert cfg.encoder.lora.target_modules == ["q_proj", "v_proj"]
    assert cfg.sequential.frames_per_chunk == 16
    assert cfg.moco.name == "stage6b_v2"
    assert cfg.moco.protocol == {
        "stream_mode": "strict_single",
        "key_transform": "gbr_horizontal_flip",
        "negative_policy": "all_past",
    }
    assert cfg.moco.optimizer == {
        "name": "AdamW",
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
    }
    assert cfg.provenance.sequential_loader.branch == "ActivityNet"
    assert cfg.tracking.comet_project == "2026-09-ishikawa-sequential-video-lora"

    if config_name == "moco_full":
        assert "linear_probe" not in cfg
    else:
        assert cfg.linear_probe.id == "lp-v1"
        assert cfg.linear_probe.probe.seeds == [0, 1, 2]
        assert cfg.linear_probe.probe.optimizer == {
            "name": "AdamW",
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
        }
        assert cfg.runtime.manifest_id == "lp-v1"


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_pipeline_roots_declare_and_resolve_mandatory_values(config_name):
    unresolved = compose_pipeline(config_name)
    assert OmegaConf.missing_keys(unresolved) == MISSING_KEYS[config_name]
    with pytest.raises(MissingMandatoryValue):
        OmegaConf.to_container(unresolved, resolve=True, throw_on_missing=True)

    resolved = compose_pipeline(config_name, REQUIRED_OVERRIDES[config_name])
    assert OmegaConf.missing_keys(resolved) == set()
    value = OmegaConf.to_container(resolved, resolve=True, throw_on_missing=True)
    assert isinstance(value, dict)


@pytest.mark.parametrize(
    "config_name,overrides,expected",
    [
        (
            "moco_full",
            (
                *REQUIRED_OVERRIDES["moco_full"],
                "runtime.resume=true",
                "runtime.device=cpu",
                "runtime.stop_after_videos=3",
                "runtime.output_root=/tmp/moco",
                "logging.disable_comet=true",
                "moco.optimizer.lr=0.002",
            ),
            {
                "runtime.resume": True,
                "runtime.device": "cpu",
                "runtime.stop_after_videos": 3,
                "runtime.output_root": "/tmp/moco",
                "logging.disable_comet": True,
                "moco.optimizer.lr": 0.002,
            },
        ),
        (
            "linear_probe_manifest",
            (
                *REQUIRED_OVERRIDES["linear_probe_manifest"],
                "runtime.command=audit",
                "runtime.manifest_id=lp-custom",
                "runtime.max_videos_per_split=2",
                "runtime.output_root=/tmp/probe",
                "logging.disable_comet=true",
            ),
            {
                "runtime.command": "audit",
                "runtime.manifest_id": "lp-custom",
                "runtime.max_videos_per_split": 2,
                "runtime.output_root": "/tmp/probe",
                "logging.disable_comet": True,
            },
        ),
        (
            "linear_probe_features",
            (
                *REQUIRED_OVERRIDES["linear_probe_features"],
                "runtime.manifest_id=lp-custom",
                "runtime.condition=moco_query_lora_final",
                "runtime.snapshot=/artifacts/snapshot",
                "runtime.feature_id=feature-custom",
                "runtime.device=cpu",
                "runtime.output_root=/tmp/probe",
                "logging.disable_comet=true",
            ),
            {
                "runtime.manifest_id": "lp-custom",
                "runtime.condition": "moco_query_lora_final",
                "runtime.snapshot": "/artifacts/snapshot",
                "runtime.feature_id": "feature-custom",
                "runtime.device": "cpu",
                "runtime.output_root": "/tmp/probe",
                "logging.disable_comet": True,
            },
        ),
        (
            "linear_probe_run",
            (
                *REQUIRED_OVERRIDES["linear_probe_run"],
                "runtime.manifest_id=lp-custom",
                "runtime.device=cuda",
                "runtime.output_root=/tmp/probe",
                "runtime.result_id=smoke",
                "logging.disable_comet=true",
                "linear_probe.probe.epochs=2",
                "linear_probe.probe.batch_size=32",
                "linear_probe.probe.optimizer.lr=0.002",
            ),
            {
                "runtime.manifest_id": "lp-custom",
                "runtime.device": "cuda",
                "runtime.output_root": "/tmp/probe",
                "runtime.result_id": "smoke",
                "logging.disable_comet": True,
                "linear_probe.probe.epochs": 2,
                "linear_probe.probe.batch_size": 32,
                "linear_probe.probe.optimizer.lr": 0.002,
            },
        ),
    ],
)
def test_pipeline_root_overrides(config_name, overrides, expected):
    cfg = compose_pipeline(config_name, overrides)
    for path, value in expected.items():
        assert OmegaConf.select(cfg, path, throw_on_missing=True) == value, path


@pytest.mark.parametrize("config_name", ROOT_CONFIGS)
def test_pipeline_hydra_keeps_launch_working_directory(config_name):
    cfg = compose_pipeline(config_name, return_hydra_config=True)
    assert cfg.hydra.job.chdir is False
    assert cfg.hydra.run.dir.startswith("log/hydra/")
    assert cfg.hydra.sweep.dir.startswith("log/hydra/")
    assert OmegaConf.to_container(cfg.hydra.sweep, resolve=False)["subdir"] == "${hydra.job.num}"


def test_tracking_overrides_do_not_change_scientific_identity():
    base = compose_pipeline('linear_probe_features', REQUIRED_OVERRIDES['linear_probe_features'])
    moved = compose_pipeline('linear_probe_features', (
        *REQUIRED_OVERRIDES['linear_probe_features'],
        'tracking.comet_project=another-project',
        'tracking.artifacts.linear_probe_features.base_vit=another-artifact',
    ))
    assert feature_science_contract(base) == feature_science_contract(moved)


def test_probe_override_is_hashed_and_noncanonical():
    base = compose_pipeline('linear_probe_run', REQUIRED_OVERRIDES['linear_probe_run'])
    changed = compose_pipeline('linear_probe_run', (
        *REQUIRED_OVERRIDES['linear_probe_run'], 'linear_probe.probe.epochs=2',
    ))
    canonical = science_contract(base)
    overridden = science_contract(changed)
    assert canonical['canonical'] is True
    assert overridden['canonical'] is False
    assert canonical['sha256'] != overridden['sha256']


def test_moco_production_match_excludes_tracking_but_includes_science():
    required = REQUIRED_OVERRIDES['moco_full']
    canonical = FullMoCoConfig.from_mapping(plain_config(compose_pipeline('moco_full', required)))
    tracking = FullMoCoConfig.from_mapping(plain_config(compose_pipeline('moco_full', (
        *required, 'tracking.comet_project=another-project',
    ))))
    changed = FullMoCoConfig.from_mapping(plain_config(compose_pipeline('moco_full', (
        *required, 'moco.optimizer.lr=0.002',
    ))))
    assert canonical.production_groups_match() is True
    assert tracking.production_groups_match() is True
    assert changed.production_groups_match() is False
