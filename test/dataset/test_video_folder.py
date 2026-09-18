import pytest
import torch

from dataset import (
    video_folder,
    VideoFolderInfo,
    transform_video,
    TransformVideoInfo,
)


@pytest.mark.parametrize(
    'root,train_dir,val_dir',
    [
        ('/mnt/NAS-TVS872XT/dataset-lab/UCF101.split/split01', 'train', 'test'),
        ('/mnt/NAS-TVS872XT/dataset-lab/HMDB51.split/video.split1', 'train', 'test'),
        ('/mnt/NAS-TVS872XT/dataset/Kinetics400', 'train', 'val'),
    ]
)
@pytest.mark.parametrize(
    'batch_size,num_workers,frames_per_clip,clips_per_video',
    [
        [1, 1, 1, 1],
        [4, 4, 8, 2]
    ]
)
@pytest.mark.parametrize('clip_duration', [0.5, 2.0])
def test_video_folder(
    root,
    train_dir,
    val_dir,
    batch_size,
    num_workers,
    frames_per_clip,
    clip_duration,
    clips_per_video,
):

    train_transform, val_transform = \
        transform_video(TransformVideoInfo(
            frames_per_clip=frames_per_clip
        ))
    train_loader, val_loader, n_classes = \
        video_folder(VideoFolderInfo(
            root=root,
            train_dir=train_dir,
            val_dir=val_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=train_transform,
            val_transform=val_transform,
            clip_duration=clip_duration,
            clips_per_video=clips_per_video
        ))

    assert n_classes > 1

    for loader in [train_loader, val_loader]:
        for data, labels in loader:

            assert isinstance(data, torch.Tensor)
            assert data.ndim == 5  # BCTHW
            assert data.shape[0] == batch_size
            assert data.shape[1] == 3  # 3 channels (RGB)
            assert data.shape[2] == frames_per_clip
            assert data.shape[3] == data.shape[4]  # H == W

            assert isinstance(labels, torch.Tensor)
            assert labels.ndim == 1
            assert labels.shape[0] == batch_size

            break  # check only the first sample
