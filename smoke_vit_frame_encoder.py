"""Extract a frame feature from one local RGB image."""

import argparse

import torch
from PIL import Image
from transformers import AutoImageProcessor

from model import ViTFrameEncoder


CHECKPOINT_ID = "google/vit-base-patch16-224"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", help="Path to a local image")
    args = parser.parse_args()

    with Image.open(args.image_path) as source:
        image = source.convert("RGB")

    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    if tuple(pixel_values.shape) != (1, 3, 224, 224):
        raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

    encoder = ViTFrameEncoder(CHECKPOINT_ID)
    total_parameters = sum(parameter.numel() for parameter in encoder.vit.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in encoder.vit.parameters() if parameter.requires_grad)
    if trainable_parameters != 0:
        raise RuntimeError(f"Backbone has {trainable_parameters} trainable parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        frame_features = encoder(pixel_values.to(device))

    if tuple(frame_features.shape) != (1, 768):
        raise RuntimeError(f"Unexpected feature shape: {tuple(frame_features.shape)}")
    finite = bool(torch.isfinite(frame_features).all().item())
    if not finite:
        raise RuntimeError("Frame feature contains NaN or Inf")

    print(f"checkpoint ID: {CHECKPOINT_ID}")
    print(f"resolved device: {device}")
    print(f"pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"pixel_values dtype: {pixel_values.dtype}")
    print(f"feature shape: {tuple(frame_features.shape)}")
    print(f"feature dtype: {frame_features.dtype}")
    print(f"feature finite: {finite}")
    print(f"total backbone parameters: {total_parameters}")
    print(f"trainable backbone parameters: {trainable_parameters}")


if __name__ == "__main__":
    main()
