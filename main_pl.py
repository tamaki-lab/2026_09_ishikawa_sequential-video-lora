import logging

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import lightning.pytorch as pl
from lightning.pytorch.plugins import TorchSyncBatchNorm


from logger import configure_logger_pl
from callback import configure_callbacks
from dataset import TrainValDataModule
from model import SimpleLightningModel


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.getLogger(__name__).info(
        "Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True)
    )
    assert torch.cuda.is_available()

    loggers, exp_name = configure_logger_pl(
        model_name=cfg.model.name,
        disable_logging=cfg.logging.disable_comet,
        save_dir=cfg.logging.comet_log_dir,
    )
    data_module = TrainValDataModule(
        dataset_cfg=cfg.dataset,
        loader_cfg=cfg.loader,
        video_cfg=cfg.video,
    )
    model_lightning = SimpleLightningModel(
        model_cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        checkpoint_cfg=cfg.checkpoint,
        n_classes=data_module.n_classes,
        exp_name=exp_name
    )

    callbacks = configure_callbacks()

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    trainer = pl.Trainer(
        # Keep GPU IDs as strings: Hydra parses devices=0 as an integer.
        devices=str(cfg.trainer.devices),
        accelerator="gpu",
        strategy="auto",
        max_epochs=cfg.trainer.num_epochs,
        logger=loggers,
        log_every_n_steps=cfg.trainer.log_interval_steps,
        accumulate_grad_batches=cfg.trainer.grad_accum,
        num_sanity_val_steps=0,
        # precision="16-true",  # for FP16 training, use with caution for nan/inf
        # fast_dev_run=True, # only for debug
        # fast_dev_run=5,  # only for debug
        # limit_train_batches=15,  # only for debug
        # limit_val_batches=15,  # only for debug
        callbacks=callbacks,
        plugins=[TorchSyncBatchNorm()],
        # profiler="simple",
    )

    trainer.fit(
        model=model_lightning,
        datamodule=data_module,
        ckpt_path=cfg.checkpoint.resume,
    )


if __name__ == "__main__":
    main()
