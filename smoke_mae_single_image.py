import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor

from model.mae_lightning_model import (
    DEFAULT_CHECKPOINT_ID,
    MAELightningModule,
)


EXPECTED_PIXEL_VALUES_SHAPE = (1, 3, 224, 224)
EXPECTED_LOGITS_SHAPE = (1, 196, 768)
EXPECTED_MASK_SHAPE = (1, 196)
EXPECTED_IDS_RESTORE_SHAPE = (1, 196)
EXPECTED_MASKED_PATCH_COUNT = 147
EXPECTED_VISIBLE_PATCH_COUNT = 49


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single-image MAE reconstruction forward smoke.",
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
            f"Unexpected {name} shape: expected {expected_shape}, got {actual_shape}"
        )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by this smoke spec, but no CUDA device is available."
        )

    pl.seed_everything(0, workers=True)
    device = torch.device("cuda:0")

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

    mae_module = MAELightningModule(
        checkpoint_id=DEFAULT_CHECKPOINT_ID,
    )
    mae_module = mae_module.to(device)
    mae_module.eval()
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = mae_module(
            pixel_values=pixel_values,
            noise=None,
        )

    loss = outputs.loss
    logits = outputs.logits
    mask = outputs.mask
    ids_restore = outputs.ids_restore

    if loss is None:
        raise AssertionError("outputs.loss is None")
    if loss.ndim != 0:
        raise AssertionError(
            f"Expected scalar loss, got shape {tuple(loss.shape)}"
        )

    loss_is_finite = bool(torch.isfinite(loss).item())
    if not loss_is_finite:
        raise AssertionError(f"Loss is not finite: {loss.item()}")

    _assert_shape("logits", logits, EXPECTED_LOGITS_SHAPE)
    _assert_shape("mask", mask, EXPECTED_MASK_SHAPE)
    _assert_shape("ids_restore", ids_restore, EXPECTED_IDS_RESTORE_SHAPE)

    masked_patch_count = int(mask.sum().item())
    visible_patch_count = mask.numel() - masked_patch_count

    if masked_patch_count != EXPECTED_MASKED_PATCH_COUNT:
        raise AssertionError(
            "Unexpected masked patch count: "
            f"expected {EXPECTED_MASKED_PATCH_COUNT}, got {masked_patch_count}"
        )
    if visible_patch_count != EXPECTED_VISIBLE_PATCH_COUNT:
        raise AssertionError(
            "Unexpected visible patch count: "
            f"expected {EXPECTED_VISIBLE_PATCH_COUNT}, got {visible_patch_count}"
        )

    resolved_revision = getattr(
        mae_module.model.config,
        "_commit_hash",
        None,
    )

    print(f"checkpoint: {mae_module.checkpoint_id}")
    print(f"checkpoint revision: {resolved_revision or 'unknown'}")
    print(f"transformers version: {transformers.__version__}")
    print(f"device: {next(mae_module.parameters()).device}")
    print(f"pixel_values shape: {list(pixel_values.shape)}")
    print(f"loss: {loss.item()}")
    print(f"loss finite: {loss_is_finite}")
    print(f"logits shape: {list(logits.shape)}")
    print(f"mask shape: {list(mask.shape)}")
    print(f"masked patch count: {masked_patch_count}")
    print(f"visible patch count: {visible_patch_count}")
    print(f"ids_restore shape: {list(ids_restore.shape)}")


if __name__ == "__main__":
    main()
