from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ConstantLR, StepLR

import main
import main_pl
import dataset.dataloader_factory as data_factory
import model.simple_lightning_model as lightning_model
from dataset import TrainValDataModule
from model import ModelConfig


@pytest.mark.parametrize("choice,factory_name", [
    ("cifar10", "cifar10"), ("imagefolder", "image_folder"),
    ("videofolder", "video_folder"), ("zero_images", "zero_images"),
])
def test_dataset_subtrees_reach_factory(compose_config, monkeypatch, choice, factory_name):
    cfg = compose_config([
        f"dataset={choice}", "loader.batch_size=3", "loader.num_workers=0",
        "video.frames_per_clip=7", "video.clip_duration=1.5", "video.clips_per_video=4",
    ])
    if choice != "zero_images":
        cfg.dataset.root = "relative/data"
    if choice in ("imagefolder", "videofolder"):
        cfg.dataset.train_dir, cfg.dataset.val_dir = "training", "validation"
    train_loader, val_loader = object(), object()
    factory = Mock(return_value=(train_loader, val_loader, 7))
    transforms = Mock(return_value=("train-transform", "val-transform"))
    monkeypatch.setattr(data_factory, factory_name, factory)
    monkeypatch.setattr(data_factory, "transform_image", transforms)
    monkeypatch.setattr(data_factory, "transform_video", transforms)

    module = TrainValDataModule(cfg.dataset, cfg.loader, cfg.video)
    assert module.n_classes == 7
    assert module.train_dataloader() is train_loader
    assert module.val_dataloader() is val_loader
    info = factory.call_args.args[0]
    assert info.batch_size == 3
    assert info.num_workers == 0
    if choice != "zero_images":
        assert info.root == "relative/data"
    if choice in ("imagefolder", "videofolder"):
        assert (info.train_dir, info.val_dir) == ("training", "validation")
    if choice == "videofolder":
        assert transforms.call_args.args[0].frames_per_clip == 7
        assert info.clip_duration == 1.5
        assert info.clips_per_video == 4


@pytest.mark.parametrize("optimizer_name,optimizer_type", [("sgd", SGD), ("adam", Adam)])
@pytest.mark.parametrize("enabled,scheduler_type", [(False, ConstantLR), (True, StepLR)])
def test_lightning_subtrees_preserve_factories(
    compose_config, monkeypatch, tmp_path, optimizer_name, optimizer_type, enabled, scheduler_type,
):
    cfg = compose_config([
        "model=vit_b", "model.use_pretrained=false", f"optimizer={optimizer_name}",
        "optimizer.lr=0.002", "optimizer.weight_decay=0.003", "optimizer.momentum=0.8",
        f"scheduler.enabled={str(enabled).lower()}",
    ])
    cfg.checkpoint.save_dir = str(tmp_path / "checkpoints")
    factory = Mock(return_value=nn.Linear(2, 7))
    monkeypatch.setattr(lightning_model, "configure_model", factory)
    module = lightning_model.SimpleLightningModel(
        cfg.model, cfg.optimizer, cfg.scheduler, cfg.checkpoint, n_classes=7, exp_name="test",
    )
    factory.assert_called_once_with(ModelConfig(
        model_name="vit_b", use_pretrained=False, torch_home="./pretrained_models", n_classes=7,
    ))
    optimizers = module.configure_optimizers()
    optimizer = optimizers["optimizer"]
    assert isinstance(optimizer, optimizer_type)
    assert isinstance(optimizers["lr_scheduler"], scheduler_type)
    assert optimizer.param_groups[0]["lr"] == 0.002
    assert optimizer.param_groups[0]["weight_decay"] == 0.003
    if optimizer_name == "sgd":
        assert optimizer.param_groups[0]["momentum"] == 0.8
    assert set(module.state_dict()) == {"model.weight", "model.bias"}
    callback, = module.configure_callbacks()
    assert callback.dirpath == str(tmp_path / "checkpoints" / "test")
    assert callback.monitor == "val_top1"
    assert callback.save_top_k == 2


@pytest.mark.parametrize("resume", [None, "relative/checkpoint.pt"])
def test_plain_training_config_wiring(compose_config, monkeypatch, resume):
    cfg = compose_config([
        "dataset=videofolder", "model=vit_b", "model.use_pretrained=false",
        "optimizer=adam", "optimizer.lr=0.002", "optimizer.weight_decay=0.003",
        "optimizer.momentum=0.8", "scheduler.enabled=true", "trainer.grad_accum=3",
        "trainer.log_interval_steps=4", "trainer.use_dp=true", "logging.disable_comet=true",
        "checkpoint.save_dir=${dataset.root}/checkpoints",
    ])
    cfg.checkpoint.resume = resume
    model = Mock()
    model.to.return_value = model
    loaders = SimpleNamespace(n_classes=7)
    mocks = {
        "configure_logger": Mock(), "configure_dataloader": Mock(return_value=loaders),
        "configure_model": Mock(return_value=model), "configure_optimizer": Mock(),
        "configure_scheduler": Mock(), "load_from_checkpoint": Mock(),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(main, name, mock)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    dp = Mock(return_value=model)
    monkeypatch.setattr(main.nn, "DataParallel", dp)
    restored = (2, 11, 13, model, object(), object())
    mocks["load_from_checkpoint"].return_value = restored
    result = main.prepare_training(cfg)
    mocks["configure_dataloader"].assert_called_once_with(
        dataset_cfg=cfg.dataset, loader_cfg=cfg.loader, video_cfg=cfg.video,
    )
    mocks["configure_model"].assert_called_once_with(ModelConfig(
        model_name="vit_b", use_pretrained=False, torch_home="./pretrained_models", n_classes=7,
    ))
    dp.assert_called_once_with(model)
    assert mocks["configure_optimizer"].call_args.kwargs == dict(
        optimizer_name="Adam", lr=0.002, weight_decay=0.003, momentum=0.8,
        model_params=model.parameters(),
    )
    mocks["configure_scheduler"].assert_called_once_with(
        optimizer=mocks["configure_optimizer"].return_value, use_scheduler=True,
    )
    logged = mocks["configure_logger"].call_args.kwargs
    assert logged["disable_logging"] is True
    assert logged["model_name"] == "vit_b"
    assert isinstance(logged["logged_params"], dict)
    assert logged["logged_params"]["checkpoint"]["save_dir"] == "./downloaded_data/checkpoints"
    assert result[5].grad_accum_interval == 3
    assert result[5].log_interval_steps == 4
    if resume:
        assert mocks["load_from_checkpoint"].call_args.args[0] == resume
        assert result[6:] == (11, 13, 2)
    else:
        mocks["load_from_checkpoint"].assert_not_called()
        assert result[6:] == (1, 1, 0)


def test_plain_training_intervals_and_checkpoint(compose_config, monkeypatch):
    cfg = compose_config(["trainer.num_epochs=3", "trainer.val_interval_epochs=2", "checkpoint.save_dir=relative/checkpoints"])
    logger, model, optimizer, scheduler, train_config = [object() for _ in range(5)]
    loaders = SimpleNamespace(train_loader=object(), val_loader=object())
    monkeypatch.setattr(main, "prepare_training", Mock(return_value=(
        logger, loaders, model, optimizer, scheduler, train_config, 1, 1, 0,
    )))
    train = Mock(return_value=SimpleNamespace(train_step=9))
    validate = Mock(return_value=SimpleNamespace(val_step=5, top1=60.0))
    save = Mock(return_value=({"checkpoint": True}, "checkpoint.pt"))
    comet = Mock()
    monkeypatch.setattr(main, "train", train)
    monkeypatch.setattr(main, "validation", validate)
    monkeypatch.setattr(main, "save_to_checkpoint", save)
    monkeypatch.setattr(main, "save_to_comet", comet)
    main.main.__wrapped__(cfg)
    assert train.call_count == 3
    assert [call.args[3] for call in validate.call_args_list] == [2, 3]
    assert [call.args[0] for call in save.call_args_list] == ["relative/checkpoints"] * 2
    assert all(call.args[1] == cfg.model.name for call in comet.call_args_list)


@pytest.mark.parametrize("devices", ["-1", "0", '"0,2"'])
def test_lightning_entrypoint_wiring(compose_config, monkeypatch, devices):
    cfg = compose_config([
        f"trainer.devices={devices}", "trainer.num_epochs=2", "trainer.log_interval_steps=4",
        "trainer.grad_accum=3", "logging.disable_comet=true", "logging.comet_log_dir=relative/comet",
        "checkpoint.resume=relative/model.ckpt",
    ])
    monkeypatch.setattr(main_pl.torch.cuda, "is_available", lambda: True)
    logger = Mock(return_value=(object(), "experiment"))
    data_module = Mock(return_value=SimpleNamespace(n_classes=7))
    model = Mock()
    trainer = Mock()
    monkeypatch.setattr(main_pl, "configure_logger_pl", logger)
    monkeypatch.setattr(main_pl, "TrainValDataModule", data_module)
    monkeypatch.setattr(main_pl, "SimpleLightningModel", model)
    monkeypatch.setattr(main_pl, "configure_callbacks", Mock(return_value=[]))
    monkeypatch.setattr(main_pl.pl, "Trainer", trainer)
    main_pl.main.__wrapped__(cfg)
    data_module.assert_called_once_with(dataset_cfg=cfg.dataset, loader_cfg=cfg.loader, video_cfg=cfg.video)
    model.assert_called_once_with(
        model_cfg=cfg.model, optimizer_cfg=cfg.optimizer, scheduler_cfg=cfg.scheduler,
        checkpoint_cfg=cfg.checkpoint, n_classes=7, exp_name="experiment",
    )
    logger.assert_called_once_with(model_name="resnet18", disable_logging=True, save_dir="relative/comet")
    trainer_kwargs = trainer.call_args.kwargs
    assert trainer_kwargs["devices"] == str(cfg.trainer.devices)
    assert trainer_kwargs["max_epochs"] == 2
    assert trainer_kwargs["log_every_n_steps"] == 4
    assert trainer_kwargs["accumulate_grad_batches"] == 3
    trainer.return_value.fit.assert_called_once_with(
        model=model.return_value, datamodule=data_module.return_value, ckpt_path="relative/model.ckpt",
    )


def test_training_components_have_no_argparse_dependency():
    root = Path(__file__).resolve().parents[2]
    for name in ("main.py", "main_pl.py", "dataset/dataloader_factory.py", "dataset/dataset_pl.py", "model/simple_lightning_model.py"):
        text = (root / name).read_text()
        assert "ArgParse" not in text
        assert "argparse" not in text
        assert "command_line_args" not in text
