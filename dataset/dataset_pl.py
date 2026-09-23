from omegaconf import DictConfig

import lightning.pytorch as pl

from dataset import configure_dataloader


class TrainValDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_cfg: DictConfig,
        loader_cfg: DictConfig,
        video_cfg: DictConfig,
    ):
        super().__init__()

        self.dataloaders_info = \
            configure_dataloader(
                dataset_cfg=dataset_cfg,
                loader_cfg=loader_cfg,
                video_cfg=video_cfg,
            )

    def train_dataloader(self):
        return self.dataloaders_info.train_loader

    def val_dataloader(self):
        return self.dataloaders_info.val_loader

    @property
    def n_classes(self):
        return self.dataloaders_info.n_classes
