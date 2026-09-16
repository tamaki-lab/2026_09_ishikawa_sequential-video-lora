import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from model.mae_lightning_model import (
    DEFAULT_CHECKPOINT_ID,
    MAELightningModule,
)


EXPECTED_PIXEL_VALUES_SHAPE = (1, 3, 224, 224)
OBSERVED_PARAMETER_NAME = (
    "model.vit.embeddings.patch_embeddings.projection.weight"
)


class SingleImageDataset(Dataset):

    def __init__(self, pixel_values: torch.Tensor):
        if tuple(pixel_values.shape) != EXPECTED_PIXEL_VALUES_SHAPE:
            raise AssertionError(
                "Unexpected pixel_values shape for single-image dataset: "
                f"expected {EXPECTED_PIXEL_VALUES_SHAPE}, "
                f"got {tuple(pixel_values.shape)}"
            )
        self.pixel_values = pixel_values

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"pixel_values": self.pixel_values[index]}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single-image, one-step MAE training smoke.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Path to a local image readable by Pillow.",
    )
    return parser.parse_args()


def _assert_shape(name, tensor, expected_shape):
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise AssertionError(
            f"Unexpected {name} shape: expected {expected_shape}, "
            f"got {actual_shape}"
        )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by this smoke spec, but no CUDA device is available."
        )

    pl.seed_everything(0, workers=True)

    if not args.image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    with Image.open(args.image_path) as input_image:
        image = input_image.convert("RGB")

    processor = AutoImageProcessor.from_pretrained(DEFAULT_CHECKPOINT_ID)
    pixel_values = processor(
        images=image,
        return_tensors="pt",
    )["pixel_values"]

    _assert_shape(
        "pixel_values",
        pixel_values,
        EXPECTED_PIXEL_VALUES_SHAPE,
    )
    if pixel_values.dtype != torch.float32:
        raise AssertionError(
            f"Expected FP32 pixel_values, got {pixel_values.dtype}"
        )

    dataset = SingleImageDataset(pixel_values)
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    if len(train_dataloader) != 1:
        raise AssertionError(
            f"Expected one training batch, got {len(train_dataloader)}"
        )

    mae_module = MAELightningModule(
        checkpoint_id=DEFAULT_CHECKPOINT_ID,
    )

    total_parameter_count = sum(
        parameter.numel() for parameter in mae_module.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in mae_module.parameters()
        if parameter.requires_grad
    )
    if trainable_parameter_count != total_parameter_count:
        raise AssertionError(
            "Expected all MAE parameters to be trainable: "
            f"trainable={trainable_parameter_count}, "
            f"total={total_parameter_count}"
        )

    try:
        target_parameter = mae_module.get_parameter(
            OBSERVED_PARAMETER_NAME,
        )
    except AttributeError as error:
        raise AssertionError(
            "Observed encoder parameter was not found: "
            f"{OBSERVED_PARAMETER_NAME}"
        ) from error
    if not target_parameter.requires_grad:
        raise AssertionError(
            f"Observed parameter is frozen: {OBSERVED_PARAMETER_NAME}"
        )
    before = target_parameter.detach().cpu().clone()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(
        mae_module,
        train_dataloaders=train_dataloader,
    )

    if trainer.global_step != 1:
        raise AssertionError(
            f"Expected trainer.global_step == 1, got {trainer.global_step}"
        )

    train_loss = trainer.callback_metrics.get("train_loss")
    if train_loss is None:
        raise AssertionError("train_loss was not recorded by Lightning")
    if train_loss.ndim != 0:
        raise AssertionError(
            f"Expected scalar train_loss, got shape {tuple(train_loss.shape)}"
        )
    loss_is_finite = bool(torch.isfinite(train_loss).item())
    if not loss_is_finite:
        raise AssertionError(
            f"Training loss is not finite: {train_loss.item()}"
        )

    after = target_parameter.detach().cpu()
    target_parameter_is_finite = bool(torch.isfinite(after).all().item())
    if not target_parameter_is_finite:
        raise AssertionError(
            f"Observed parameter contains NaN or Inf: {OBSERVED_PARAMETER_NAME}"
        )

    delta = after - before
    delta_norm = torch.linalg.vector_norm(delta)
    parameter_changed = not torch.equal(before, after)
    delta_norm_is_finite = bool(torch.isfinite(delta_norm).item())

    if not parameter_changed:
        raise AssertionError(
            f"Observed parameter did not change: {OBSERVED_PARAMETER_NAME}"
        )
    if not delta_norm_is_finite or delta_norm.item() <= 0:
        raise AssertionError(
            f"Invalid parameter delta norm: {delta_norm.item()}"
        )

    resolved_revision = getattr(
        mae_module.model.config,
        "_commit_hash",
        None,
    )
    resolved_device = trainer.strategy.root_device

    print(f"checkpoint: {mae_module.checkpoint_id}")
    print(f"checkpoint revision: {resolved_revision or 'unknown'}")
    print(f"transformers version: {transformers.__version__}")
    print(f"device: {resolved_device}")
    print(f"pixel_values shape: {list(pixel_values.shape)}")
    print(f"trainable parameter count: {trainable_parameter_count}")
    print(f"total parameter count: {total_parameter_count}")
    print(f"training loss: {train_loss.item()}")
    print(f"loss finite: {loss_is_finite}")
    print(f"trainer global_step: {trainer.global_step}")
    print(f"observed parameter name: {OBSERVED_PARAMETER_NAME}")
    print(f"parameter changed: {parameter_changed}")
    print(f"target parameter finite: {target_parameter_is_finite}")
    print(f"delta norm: {delta_norm.item()}")
    print(f"delta norm finite: {delta_norm_is_finite}")


if __name__ == "__main__":
    main()
