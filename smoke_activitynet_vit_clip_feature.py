"""Encode the first ActivityNet training chunk with frozen ViT and masked mean.

Usage:
    python smoke_activitynet_vit_clip_feature.py /path/to/ActivityNet
"""

import argparse
import subprocess
from pathlib import Path

import sequential_loader as sl
import torch
from transformers import AutoImageProcessor

from model import MaskedMeanClipAggregator, ViTFrameEncoder
from sequential_vit_bridge import encode_chunk


CHECKPOINT_ID = "google/vit-base-patch16-224"
EXPECTED_BRANCH = "dev"
BASE_COMMIT = "3c0e40e86924ceb38931c2d5214ecf3251a8f99d"
LOADER_COMMIT = "19a0ed7e4c00300214bc9a2fe12da8c72c0499c0"
FRAMES_PER_CHUNK = 16
FEATURE_SIZE = 768


def git_output(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path to the ActivityNet dataset root")
    args = parser.parse_args()

    implementation_root = Path(__file__).resolve().parent
    branch = git_output(implementation_root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH}, got {branch}")
    implementation_revision = git_output(implementation_root, "rev-parse", "HEAD")
    git_output(implementation_root, "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD")

    loader_root = Path(sl.__file__).resolve().parent.parent
    loader_revision = git_output(loader_root, "rev-parse", "HEAD")
    if loader_revision != LOADER_COMMIT:
        raise RuntimeError(f"Expected sequential_loader revision {LOADER_COMMIT}, got {loader_revision}")
    if git_output(loader_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Pinned sequential_loader checkout has tracked changes")

    adapter = sl.ActivityNetAdapter(dataset_root=args.dataset_root)
    sources = adapter.sequence_sources("training")
    if not sources:
        raise RuntimeError("ActivityNet training subset has no sources")
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
    if sample.frames.ndim != 4 or tuple(sample.frames.shape[:2]) != (FRAMES_PER_CHUNK, 3):
        raise RuntimeError(f"Unexpected frames shape: {tuple(sample.frames.shape)}")
    if sample.frames.device.type != "cpu" or sample.frames.dtype != torch.uint8:
        raise RuntimeError("frames must be CPU uint8")
    if tuple(sample.valid_mask.shape) != (FRAMES_PER_CHUNK,) or sample.valid_mask.dtype != torch.bool:
        raise RuntimeError("valid_mask must be bool [16]")
    valid_count = int(sample.valid_mask.sum().item())
    if valid_count == 0:
        raise RuntimeError("SequentialSample has no valid frames")
    if not torch.equal(sample.frame_indices[sample.valid_mask], torch.arange(valid_count)):
        raise RuntimeError("First chunk must contain contiguous frame indices starting at zero")
    original_indices = sample.frame_indices.clone()
    original_timestamps = sample.timestamps.clone()

    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ViTFrameEncoder(CHECKPOINT_ID).to(device)
    total_parameters = sum(parameter.numel() for parameter in encoder.vit.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in encoder.vit.parameters() if parameter.requires_grad)
    if trainable_parameters != 0:
        raise RuntimeError(f"Backbone has {trainable_parameters} trainable parameters")

    encoder.eval()
    aggregator = MaskedMeanClipAggregator()
    with torch.no_grad():
        pixel_values, valid_features, frame_features = encode_chunk(sample, processor, encoder, device)
        clip_feature = aggregator(frame_features, sample.valid_mask)

    if tuple(frame_features.shape) != (FRAMES_PER_CHUNK, FEATURE_SIZE):
        raise RuntimeError(f"Unexpected frame_features shape: {tuple(frame_features.shape)}")
    if tuple(clip_feature.shape) != (FEATURE_SIZE,):
        raise RuntimeError(f"Unexpected clip_feature shape: {tuple(clip_feature.shape)}")
    frame_finite = bool(torch.isfinite(frame_features).all().item())
    clip_finite = bool(torch.isfinite(clip_feature).all().item())
    padding_zero = bool(torch.all(frame_features[~sample.valid_mask.to(device)] == 0).item())
    if not frame_finite or not clip_finite or not padding_zero:
        raise RuntimeError("Clip features failed finite or padding check")
    if not torch.equal(sample.frame_indices, original_indices) or not torch.allclose(
        sample.timestamps, original_timestamps, rtol=0, atol=0, equal_nan=True
    ):
        raise RuntimeError("Frame metadata alignment changed during encoding")

    print(f"implementation branch: {branch}")
    print(f"implementation revision: {implementation_revision}")
    print(f"implementation base: {BASE_COMMIT}")
    print(f"sequential_loader revision: {loader_revision}")
    print(f"checkpoint ID: {CHECKPOINT_ID}")
    print(f"resolved device: {device}")
    print("dataset: ActivityNet v1.3")
    print("split: training")
    print(f"training sources: {len(sources)}")
    print(f"sequence_id: {sample.sequence_id}")
    print(f"sequence_index: {sample.sequence_index}")
    print(f"is_first: {sample.is_first}")
    print(f"is_last: {sample.is_last}")
    print(f"frames shape: {tuple(sample.frames.shape)}")
    print(f"frames dtype: {sample.frames.dtype}")
    print(f"valid count / T: {valid_count}/{sample.frames.shape[0]}")
    print(f"valid_mask: {sample.valid_mask.tolist()}")
    print(f"frame_indices: {sample.frame_indices.tolist()}")
    print(f"timestamps: {sample.timestamps.tolist()}")
    print(f"pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"valid_features shape: {tuple(valid_features.shape)}")
    print(f"frame_features shape: {tuple(frame_features.shape)}")
    print(f"clip_feature shape: {tuple(clip_feature.shape)}")
    print(f"frame feature finite: {frame_finite}")
    print(f"clip feature finite: {clip_finite}")
    print(f"padding rows zero: {padding_zero}")
    print(f"total backbone parameters: {total_parameters}")
    print(f"trainable backbone parameters: {trainable_parameters}")


if __name__ == "__main__":
    main()
