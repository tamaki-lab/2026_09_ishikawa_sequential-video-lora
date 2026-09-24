"""Verify one engineering-only Q/V LoRA update on the first ActivityNet chunk.

Usage:
    python -m scripts.smoke.smoke_activitynet_vit_lora_one_step /path/to/ActivityNet
"""

import argparse
import subprocess
from pathlib import Path

import peft
import sequential_loader as sl
import torch
import transformers
from transformers import AutoImageProcessor

from model.aggregators import MaskedMeanClipAggregator
from model.backbones.vit import ViTLoRAFrameEncoder
from integration.sequential_vit import encode_chunk
from training.moco_audit import audit_encoder_parameters as audit_parameters


CHECKPOINT_ID = "google/vit-base-patch16-224"
EXPECTED_BRANCH = "dev"
BASE_COMMIT = "b6385d87e2e3e82a719d8f4b686b44aa293b1135"
LOADER_BRANCH = "ActivityNet"
LOADER_COMMIT = "19a0ed7e4c00300214bc9a2fe12da8c72c0499c0"
FRAMES_PER_CHUNK = 16
FEATURE_SIZE = 768


def git_output(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def run_one_step(encoder, clip_feature):
    """Audit a single AdamW step; this loss has no scientific interpretation."""
    if tuple(clip_feature.shape) != (FEATURE_SIZE,) or not torch.isfinite(clip_feature).all().item():
        raise RuntimeError("Expected a finite clip feature with shape [768]")
    base, lora = audit_parameters(encoder)
    optimizer = torch.optim.AdamW(list(lora.values()), lr=1.0e-3, weight_decay=0.0)
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    if len(optimized) != len(lora) or {id(p) for p in optimized} != {id(p) for p in lora.values()}:
        raise RuntimeError("Optimizer parameters do not match the trainable LoRA parameters")
    print("optimizer matches LoRA trainable parameters: True")
    before = {name: p.detach().cpu().clone() for name, p in encoder.named_parameters()}
    loss = clip_feature.float().pow(2).mean()
    loss_finite = bool(torch.isfinite(loss).item())
    print(f"loss: {loss.item()}")
    print(f"loss finite: {loss_finite}")
    if not loss_finite:
        raise RuntimeError("Engineering loss contains NaN or Inf")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in lora.values() if p.grad is not None]
    finite_count = sum(bool(torch.isfinite(grad).all().item()) for grad in gradients)
    nonzero_count = sum(bool(torch.count_nonzero(grad).item()) for grad in gradients)
    base_grad_count = sum(p.grad is not None for p in base.values())
    print(f"LoRA finite gradient count: {finite_count}")
    print(f"LoRA nonzero gradient count: {nonzero_count}")
    print(f"base gradient count: {base_grad_count}")
    if finite_count != len(gradients) or nonzero_count == 0 or base_grad_count != 0:
        raise RuntimeError("Expected finite, nonzero LoRA gradients and no base gradients")

    optimizer.step()
    base_changed = sum(not torch.equal(before[name], p.detach().cpu()) for name, p in base.items())
    lora_changed = sum(not torch.equal(before[name], p.detach().cpu()) for name, p in lora.items())
    parameters_finite = all(bool(torch.isfinite(p).all().item()) for p in encoder.parameters())
    print(f"base changed parameter count: {base_changed}")
    print(f"LoRA changed parameter count: {lora_changed}")
    print(f"all changed parameters finite: {parameters_finite}")
    print(f"all parameters finite: {parameters_finite}")
    if base_changed != 0 or lora_changed == 0 or not parameters_finite:
        raise RuntimeError("Expected unchanged base, updated LoRA and finite parameters after the step")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path to the ActivityNet dataset root")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    implementation_root = Path(__file__).resolve().parent
    branch = git_output(implementation_root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH}, got {branch}")
    revision = git_output(implementation_root, "rev-parse", "HEAD")
    git_output(implementation_root, "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD")
    loader_root = Path(sl.__file__).resolve().parent.parent
    loader_branch = git_output(loader_root, "branch", "--show-current")
    loader_revision = git_output(loader_root, "rev-parse", "HEAD")
    if loader_branch != LOADER_BRANCH or loader_revision != LOADER_COMMIT:
        raise RuntimeError(f"Expected sequential_loader {LOADER_BRANCH}@{LOADER_COMMIT}, got {loader_branch}@{loader_revision}")
    if git_output(loader_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Pinned sequential_loader checkout has tracked changes")
    if transformers.__version__ != "5.17.0" or peft.__version__ != "0.21.0":
        raise RuntimeError("Stage 4 requires transformers==5.17.0 and peft==0.21.0")

    device = torch.device(args.device)
    print(f"implementation branch: {branch}")
    print(f"implementation revision: {revision}")
    print(f"implementation base: {BASE_COMMIT}")
    print(f"sequential_loader branch: {loader_branch}")
    print(f"sequential_loader revision: {loader_revision}")
    print(f"checkpoint ID: {CHECKPOINT_ID}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"PEFT version: {peft.__version__}")
    print(f"resolved device: {device}")

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
    encoder = ViTLoRAFrameEncoder(CHECKPOINT_ID).to(device)
    encoder.train()
    pixel_values, valid_features, frame_features = encode_chunk(sample, processor, encoder, device)
    clip_feature = MaskedMeanClipAggregator()(frame_features, sample.valid_mask)
    if tuple(frame_features.shape) != (FRAMES_PER_CHUNK, FEATURE_SIZE):
        raise RuntimeError(f"Unexpected frame_features shape: {tuple(frame_features.shape)}")
    frame_finite = bool(torch.isfinite(frame_features).all().item())
    padding_zero = bool(torch.all(frame_features[~sample.valid_mask.to(device)] == 0).item())
    if not frame_finite or not padding_zero:
        raise RuntimeError("Frame features failed finite or padding check")
    if not torch.equal(sample.frame_indices, original_indices) or not torch.allclose(
        sample.timestamps, original_timestamps, rtol=0, atol=0, equal_nan=True
    ):
        raise RuntimeError("Frame metadata alignment changed during encoding")

    print("dataset: ActivityNet v1.3")
    print("split: training")
    print(f"training sources: {len(sources)}")
    print(f"sequence_id: {sample.sequence_id}")
    print(f"sequence_index: {sample.sequence_index}")
    print(f"valid count / T: {valid_count}/{sample.frames.shape[0]}")
    print(f"frame_indices: {sample.frame_indices.tolist()}")
    print(f"timestamps: {sample.timestamps.tolist()}")
    print(f"pixel_values shape: {tuple(pixel_values.shape)}")
    print(f"valid_features shape: {tuple(valid_features.shape)}")
    print(f"frame_features shape: {tuple(frame_features.shape)}")
    print(f"clip_feature shape: {tuple(clip_feature.shape)}")
    print(f"frame feature finite: {frame_finite}")
    print(f"clip feature finite: {bool(torch.isfinite(clip_feature).all().item())}")
    print(f"padding rows zero: {padding_zero}")
    run_one_step(encoder, clip_feature)


if __name__ == "__main__":
    main()
