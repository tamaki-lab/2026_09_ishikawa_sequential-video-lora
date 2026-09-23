import pytest
from omegaconf import OmegaConf


# Snapshot of ArgParse defaults at the migration base 7705e21 (spec section 3.7).
DEFAULTS = {
    "dataset.root": "./downloaded_data",
    "dataset.name": "CIFAR10",
    "model.torch_home": "./pretrained_models",
    "model.name": "resnet18",
    "model.use_pretrained": True,
    "video.frames_per_clip": 16,
    "video.clip_duration": 80 / 30,
    "video.clips_per_video": 1,
    "loader.batch_size": 8,
    "loader.num_workers": 2,
    "trainer.num_epochs": 25,
    "trainer.val_interval_epochs": 1,
    "trainer.log_interval_steps": 1,
    "optimizer.name": "SGD",
    "trainer.grad_accum": 1,
    "optimizer.lr": 1e-4,
    "optimizer.momentum": 0.9,
    "optimizer.weight_decay": 5e-4,
    "scheduler.enabled": False,
    "trainer.use_dp": False,
    "trainer.devices": "-1",
    "logging.comet_log_dir": "./comet_logs/",
    "logging.tf_log_dir": "./tf_logs/",
    "checkpoint.save_dir": "./log",
    "checkpoint.resume": None,
    "logging.disable_comet": False,
}


def test_legacy_defaults(compose_config):
    cfg = compose_config()
    for path, expected in DEFAULTS.items():
        actual = OmegaConf.select(cfg, path, throw_on_missing=True)
        assert actual == expected, path
        assert type(actual) is type(expected), path


@pytest.mark.parametrize("group,choice,name", [
    ("dataset", "cifar10", "CIFAR10"),
    ("dataset", "imagefolder", "ImageFolder"),
    ("dataset", "videofolder", "VideoFolder"),
    ("dataset", "zero_images", "ZeroImages"),
    *[("model", name, name) for name in
      ("resnet18", "resnet50", "x3d", "abn_r50", "vit_b", "zero_output_dummy")],
    ("optimizer", "sgd", "SGD"),
    ("optimizer", "adam", "Adam"),
])
def test_group_choices(compose_config, group, choice, name):
    cfg = compose_config([f"{group}={choice}"])
    assert cfg[group].name == name
    if group == "dataset" and choice in ("imagefolder", "videofolder"):
        assert cfg.dataset.root == "./downloaded_data"
        assert cfg.dataset.train_dir == "train"
        assert cfg.dataset.val_dir == "val"
    if group == "model":
        assert cfg.model.use_pretrained is True
        assert cfg.model.torch_home == "./pretrained_models"
    if group == "optimizer":
        assert cfg.optimizer.lr == 1e-4
        assert cfg.optimizer.weight_decay == 5e-4
        assert cfg.optimizer.momentum == 0.9


def test_representative_overrides(compose_config):
    cfg = compose_config([
        "dataset=videofolder", "model=vit_b", "optimizer=adam",
        "loader.batch_size=12", "loader.num_workers=0", "optimizer.lr=0.002",
        "model.use_pretrained=false", "scheduler.enabled=true",
        "logging.disable_comet=true", "checkpoint.resume=null",
        "video.frames_per_clip=8", "video.clip_duration=1.5", "video.clips_per_video=3",
        "trainer.devices=0",
    ])
    assert (cfg.dataset.name, cfg.model.name, cfg.optimizer.name) == ("VideoFolder", "vit_b", "Adam")
    assert (cfg.loader.batch_size, cfg.loader.num_workers) == (12, 0)
    assert cfg.optimizer.lr == 0.002
    assert cfg.model.use_pretrained is False
    assert cfg.scheduler.enabled is True
    assert cfg.logging.disable_comet is True
    assert cfg.checkpoint.resume is None
    assert (cfg.video.frames_per_clip, cfg.video.clip_duration, cfg.video.clips_per_video) == (8, 1.5, 3)
    assert str(cfg.trainer.devices) == "0"


def test_hydra_paths_and_cwd(compose_config):
    cfg = compose_config(return_hydra_config=True)
    assert cfg.hydra.job.chdir is False
    assert cfg.hydra.run.dir.startswith("log/hydra/")
    assert cfg.hydra.sweep.dir.startswith("log/hydra/multirun/")
    assert OmegaConf.to_container(cfg.hydra.sweep, resolve=False)["subdir"] == "${hydra.job.num}"
