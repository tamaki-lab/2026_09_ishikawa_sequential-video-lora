from dataclasses import dataclass
from omegaconf import DictConfig

from torch.utils.data import DataLoader

from dataset import (
    cifar10,
    Cifar10Info,
    image_folder,
    ImageFolderInfo,
    video_folder,
    VideoFolderInfo,
    zero_images,
    ZeroImageInfo,
    transform_image,
    TransformImageInfo,
    transform_video,
    TransformVideoInfo,
)


@dataclass
class DataloadersInfo:
    """DataloadersInfo

        train_loader (torch.utils.data.DataLoader): training set loader
        val_loader (torch.utils.data.DataLoader): validation set loader
        n_classes (int): number of classes
    """
    train_loader: DataLoader
    val_loader: DataLoader
    n_classes: int


def configure_dataloader(
    dataset_cfg: DictConfig,
    loader_cfg: DictConfig,
    video_cfg: DictConfig,
):
    """dataloader factory

    Args:
        dataset_cfg (DictConfig): dataset name, root and split directories
        loader_cfg (DictConfig): batch size and worker count
        video_cfg (DictConfig): clip sampling settings

    Raises:
        ValueError: invalid dataset_name is given

    Returns:
        (DataloadersInfo): dataset information
    """

    dataset_name = dataset_cfg.name

    if dataset_name == "CIFAR10":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            cifar10(Cifar10Info(
                root=dataset_cfg.root,
                batch_size=loader_cfg.batch_size,
                num_workers=loader_cfg.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_name == "ImageFolder":
        train_transform, val_transform = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            image_folder(ImageFolderInfo(
                root=dataset_cfg.root,
                train_dir=dataset_cfg.train_dir,
                val_dir=dataset_cfg.val_dir,
                batch_size=loader_cfg.batch_size,
                num_workers=loader_cfg.num_workers,
                train_transform=train_transform,
                val_transform=val_transform
            ))

    elif dataset_name == "VideoFolder":
        train_transform, val_transform = \
            transform_video(TransformVideoInfo(
                frames_per_clip=video_cfg.frames_per_clip
            ))
        train_loader, val_loader, n_classes = \
            video_folder(VideoFolderInfo(
                root=dataset_cfg.root,
                train_dir=dataset_cfg.train_dir,
                val_dir=dataset_cfg.val_dir,
                batch_size=loader_cfg.batch_size,
                num_workers=loader_cfg.num_workers,
                train_transform=train_transform,
                val_transform=val_transform,
                clip_duration=video_cfg.clip_duration,
                clips_per_video=video_cfg.clips_per_video
            ))

    elif dataset_name == "ZeroImages":
        train_transform, _ = \
            transform_image(TransformImageInfo())
        train_loader, val_loader, n_classes = \
            zero_images(ZeroImageInfo(
                batch_size=loader_cfg.batch_size,
                num_workers=loader_cfg.num_workers,
                transform=train_transform,
            ))

    else:
        raise ValueError("invalid dataset_name")

    return DataloadersInfo(
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=n_classes
    )
