# import pytest
# import torch

# from dataset import (
#     cifar10,
#     Cifar10Info,
#     transform_image,
#     TransformImageInfo
# )

# @pytest.fixture(scope="session")  # do not work. WHY?
# @pytest.mark.parametrize('batch_size', [1, 2, 4])
# @pytest.mark.parametrize('num_workers', [1, 2])
# def test_cifar10(
#     batch_size,
#     num_workers,
#     tmp_path_factory,
# ):

#     train_transform, val_transform = \
#         transform_image(TransformImageInfo())
#     train_loader, val_loader, n_classes = \
#         cifar10(Cifar10Info(
#             root=str(tmp_path_factory.mktemp('cifar10')),
#             batch_size=batch_size,
#             num_workers=num_workers,
#             train_transform=train_transform,
#             val_transform=val_transform
#         ))

#     assert n_classes == 10

#     for loader in [train_loader, val_loader]:
#         for data, labels in loader:

#             assert isinstance(data, torch.Tensor)
#             assert data.ndim == 4  # BCHW
#             assert data.shape[0] == batch_size
#             assert data.shape[1] == 3  # 3 channels (RGB)
#             assert data.shape[2] == data.shape[3]  # H == W

#             assert isinstance(labels, torch.Tensor)
#             assert labels.ndim == 1
#             assert labels.shape[0] == batch_size

#             break  # check only one sample
