"""Encode one 50Salads train1 chunk with the frozen ViT frame encoder.

Usage:
    python smoke_50salads_vit_bridge.py /path/to/50Salads
"""

import argparse
import subprocess
from pathlib import Path

import sequential_loader as sl
import torch
from transformers import AutoImageProcessor

from model import ViTFrameEncoder


CHECKPOINT_ID = "google/vit-base-patch16-224"
EXPECTED_BRANCH = "feature-50salads-loder"
BASE_COMMIT = "e7e037a9191f36b26e87f80f48caccfb35b6166d"
LOADER_COMMIT = "cef09aa12560127451a5f569d86d5d51671e6986"
FRAMES_PER_CHUNK = 16
FEATURE_SIZE = 768


def git_output(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def encode_chunk(sample: sl.SequentialSample, processor, encoder, device: torch.device):
    """Encode valid frames in source order and restore their chunk positions."""
    valid_mask = sample.valid_mask
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        raise RuntimeError("SequentialSample has no valid frames")

    valid_frames = sample.frames[valid_mask]
    pixel_values = processor(images=valid_frames, return_tensors="pt")["pixel_values"]
    if tuple(pixel_values.shape) != (valid_count, 3, 224, 224):
        raise RuntimeError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

    valid_features = encoder(pixel_values.to(device))
    if tuple(valid_features.shape) != (valid_count, FEATURE_SIZE):
        raise RuntimeError(f"Unexpected valid feature shape: {tuple(valid_features.shape)}")
    if not torch.isfinite(valid_features).all().item():
        raise RuntimeError("Valid feature contains NaN or Inf")

    frame_features = torch.zeros(
        (sample.frames.shape[0], FEATURE_SIZE),
        dtype=valid_features.dtype,
        device=valid_features.device,
    )
    device_mask = valid_mask.to(frame_features.device)
    frame_features[device_mask] = valid_features
    if not torch.all(frame_features[~device_mask] == 0).item():
        raise RuntimeError("Padding feature rows are not zero")

    return pixel_values, valid_features, frame_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path to the 50Salads dataset root")
    args = parser.parse_args()

    implementation_root = Path(__file__).resolve().parent
    branch = git_output(implementation_root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH}, got {branch}")
    branch_base = git_output(implementation_root, "merge-base", "HEAD", "main")
    if branch_base != BASE_COMMIT:
        raise RuntimeError(f"Expected branch base {BASE_COMMIT}, got {branch_base}")

    loader_root = Path(sl.__file__).resolve().parent.parent
    loader_revision = git_output(loader_root, "rev-parse", "HEAD") if (loader_root / ".git").exists() else "unavailable"
    if loader_revision != "unavailable" and loader_revision != LOADER_COMMIT:
        raise RuntimeError(f"Expected sequential_loader revision {LOADER_COMMIT}, got {loader_revision}")

    adapter = sl.Salads50Adapter(dataset_root=args.dataset_root)
    sources = adapter.sequence_sources("train1")
    dataset = sl.SequentialDataset(
        sources=sources,
        reader=sl.SequentialVideoReader(),
        chunk_config=sl.FixedChunkConfig(frames_per_chunk=FRAMES_PER_CHUNK),
    )
    loader = sl.build_sequential_dataloader(dataset=dataset)
    with sl.sequential_sample_stream(loader) as samples:
        sample = next(samples)

    if not isinstance(sample, sl.SequentialSample):
        raise RuntimeError(f"Expected SequentialSample, got {type(sample).__name__}")
    if sample.sequence_id != sources[0].sequence_id or sample.sequence_index != 0 or not sample.is_first:
        raise RuntimeError("First sample does not match the first source and first chunk")
    if tuple(sample.frames.shape[:2]) != (FRAMES_PER_CHUNK, 3):
        raise RuntimeError(f"Unexpected frames shape: {tuple(sample.frames.shape)}")
    if sample.frames.device.type != "cpu" or sample.frames.dtype != torch.uint8:
        raise RuntimeError("frames must be CPU uint8")
    if tuple(sample.valid_mask.shape) != (FRAMES_PER_CHUNK,):
        raise RuntimeError(f"Unexpected valid_mask shape: {tuple(sample.valid_mask.shape)}")

    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ViTFrameEncoder(CHECKPOINT_ID).to(device)
    total_parameters = sum(parameter.numel() for parameter in encoder.vit.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in encoder.vit.parameters() if parameter.requires_grad)
    if trainable_parameters != 0:
        raise RuntimeError(f"Backbone has {trainable_parameters} trainable parameters")

    encoder.eval()
    with torch.no_grad():
        pixel_values, valid_features, frame_features = encode_chunk(sample, processor, encoder, device)

    feature_finite = bool(torch.isfinite(frame_features).all().item())
    padding_zero = bool(torch.all(frame_features[~sample.valid_mask.to(device)] == 0).item())
    if not feature_finite or not padding_zero:
        raise RuntimeError("Frame features failed finite or padding check")

    print(f"implementation branch: {branch}")
    print(f"branch base: {branch_base}")
    print(f"sequential_loader revision: {loader_revision} (expected {LOADER_COMMIT})")
    print(f"checkpoint ID: {CHECKPOINT_ID}")
    print(f"resolved device: {device}")
    print(f"sequence_id: {sample.sequence_id}")
    print(f"sequence_index: {sample.sequence_index}")
    print(f"is_first: {sample.is_first}")
    print(f"is_last: {sample.is_last}")
    print(f"frames shape: {tuple(sample.frames.shape)}")
    print(f"frames dtype: {sample.frames.dtype}")
    print(f"valid count / T: {int(sample.valid_mask.sum().item())}/{sample.frames.shape[0]}")
    print(f"valid_mask: {sample.valid_mask.tolist()}")
    print(f"frame_indices: {sample.frame_indices.tolist()}")
    print(f"timestamps: {sample.timestamps.tolist()}")
    print(f"pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"pixel_values dtype: {pixel_values.dtype}")
    print(f"valid feature shape: {tuple(valid_features.shape)}")
    print(f"frame_features shape: {tuple(frame_features.shape)}")
    print(f"frame_features dtype: {frame_features.dtype}")
    print(f"feature finite: {feature_finite}")
    print(f"padding rows zero: {padding_zero}")
    print(f"total backbone parameters: {total_parameters}")
    print(f"trainable backbone parameters: {trainable_parameters}")


if __name__ == "__main__":
    main()
