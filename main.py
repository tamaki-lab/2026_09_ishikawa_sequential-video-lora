import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import torch
from torch import nn

from dataset import configure_dataloader
from model import (
    configure_model,
    ModelConfig,
)
from setup import configure_optimizer, configure_scheduler

from logger import configure_logger
from utils import (
    save_to_checkpoint, save_to_comet,
    load_from_checkpoint,
)

from train import train, TrainConfig
from val import validation


class TqdmEpoch(tqdm):
    def __init__(
            self,
            start_epoch: int,
            num_epochs: int,
            *args,
            **kwargs):
        super().__init__(
            range(start_epoch + 1, num_epochs + 1), *args, **kwargs
        )


def prepare_training(cfg: DictConfig):
    """prepare training objects from the composed config

    Args:
        cfg (DictConfig): nested training configuration

    Returns:
        a set of training objects
    """

    logger = configure_logger(
        logged_params=OmegaConf.to_container(cfg, resolve=True),
        model_name=cfg.model.name,
        disable_logging=cfg.logging.disable_comet,
    )

    dataloaders = configure_dataloader(
        dataset_cfg=cfg.dataset,
        loader_cfg=cfg.loader,
        video_cfg=cfg.video,
    )

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model = configure_model(ModelConfig(
        model_name=cfg.model.name,
        use_pretrained=cfg.model.use_pretrained,
        torch_home=cfg.model.torch_home,
        n_classes=dataloaders.n_classes,
    ))
    model = model.to(device)
    if cfg.trainer.use_dp:
        model = nn.DataParallel(model)  # type: ignore[assignment]

    optimizer = configure_optimizer(
        optimizer_name=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        momentum=cfg.optimizer.momentum,
        model_params=model.parameters()
    )
    scheduler = configure_scheduler(
        optimizer=optimizer,
        use_scheduler=cfg.scheduler.enabled
    )

    train_config = TrainConfig(
        grad_accum_interval=cfg.trainer.grad_accum,
        log_interval_steps=cfg.trainer.log_interval_steps
    )

    if cfg.checkpoint.resume:
        (
            start_epoch,
            current_train_step,
            current_val_step,
            model,
            optimizer,
            scheduler,
        ) = load_from_checkpoint(  # type: ignore[assignment]
            cfg.checkpoint.resume,
            model,
            optimizer,
            scheduler,
            device
        )
    else:
        current_train_step = 1
        current_val_step = 1
        start_epoch = 0

    return (
        logger,
        dataloaders,
        model,
        optimizer,
        scheduler,
        train_config,
        current_train_step,
        current_val_step,
        start_epoch,
    )


class ValidationChecker:
    def __init__(self, val_interval_epochs, num_epochs):
        self.val_interval_epochs = val_interval_epochs
        self.num_epochs = num_epochs

    def should_validate(self, current_epoch):
        return (
            current_epoch % self.val_interval_epochs == 0
            or current_epoch == self.num_epochs
        )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.getLogger(__name__).info(
        "Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True)
    )

    (
        logger,
        dataloaders,
        model,
        optimizer,
        scheduler,
        train_config,
        current_train_step,
        current_val_step,
        start_epoch,
    ) = prepare_training(cfg)

    val_checker = ValidationChecker(cfg.trainer.val_interval_epochs, cfg.trainer.num_epochs)

    with TqdmEpoch(
        start_epoch, cfg.trainer.num_epochs, unit='epoch',
    ) as progress_bar_epoch:
        for current_epoch in progress_bar_epoch:
            progress_bar_epoch.set_description(f"[epoch {current_epoch:03d}]")

            train_output = train(
                model,
                optimizer,
                scheduler,
                dataloaders.train_loader,
                current_train_step,
                current_epoch,
                logger,
                train_config
            )
            current_train_step = train_output.train_step

            if val_checker.should_validate(current_epoch):

                val_output = validation(
                    model,
                    dataloaders.val_loader,
                    current_val_step,
                    current_epoch,
                    logger,
                )
                current_val_step = val_output.val_step

                checkpoint_dict, _ = save_to_checkpoint(
                    cfg.checkpoint.save_dir,
                    current_epoch,
                    current_train_step,
                    current_val_step,
                    val_output.top1,
                    model,
                    optimizer,
                    scheduler,
                    logger
                )
                save_to_comet(
                    checkpoint_dict,
                    cfg.model.name,
                    logger
                )


if __name__ == "__main__":
    main()
