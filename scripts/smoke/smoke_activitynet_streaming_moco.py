"""Common Streaming MoCo smoke / canary, each invocation a fresh run.

Defaults to Stage 6B: strict single stream, GBR + horizontal flip, all past keys.

Usage:
    python -m scripts.smoke.smoke_activitynet_streaming_moco /path/to/ActivityNet --max-steps 10
    python -m scripts.smoke.smoke_activitynet_streaming_moco /path/to/ActivityNet --preset stage6a
    python -m scripts.smoke.smoke_activitynet_streaming_moco /path/to/ActivityNet --stream-mode round_robin
"""

import argparse
from dataclasses import replace
from pathlib import Path

import sequential_loader as sl
import torch
from transformers import AutoImageProcessor

from model.backbones.vit import ViTLoRAFrameEncoder
from self_supervised.moco import ViTLoRAMoCo
from integration.sequential_stream import validate_sources
from training.moco_canary import run_streaming_moco
from training.moco_protocol import STAGE6A_PROTOCOL, STAGE6B_PROTOCOL
from .smoke_activitynet_vit_lora_moco_one_step import audit_provenance
from .smoke_activitynet_vit_lora_one_step import CHECKPOINT_ID, git_output


BASE_COMMIT = '4835b5736f0b1dcc9962cbeffd85e880010311ea'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--max-steps', type=int, default=10, help='Training updates; excludes warm-up keys')
    parser.add_argument('--preset', choices=('stage6a', 'stage6b'), default='stage6b')
    parser.add_argument('--stream-mode', choices=('round_robin', 'strict_single'))
    parser.add_argument('--key-transform', choices=('horizontal_flip', 'gbr_horizontal_flip'))
    parser.add_argument('--negative-policy', choices=('different_sequence', 'all_past'))
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error('--max-steps must be positive')
    preset = STAGE6A_PROTOCOL if args.preset == 'stage6a' else STAGE6B_PROTOCOL
    overrides = {axis: getattr(args, axis) for axis in ('stream_mode', 'key_transform', 'negative_policy')
                 if getattr(args, axis) is not None}
    protocol = replace(preset, **overrides)
    audit_provenance()
    git_output(Path(__file__).resolve().parent, 'merge-base', '--is-ancestor', BASE_COMMIT, 'HEAD')
    print(f'Streaming MoCo implementation base: {BASE_COMMIT}')
    print(f'protocol preset: {args.preset}; stream_mode: {protocol.stream_mode}; '
          f'key_transform: {protocol.key_transform}; negative_policy: {protocol.negative_policy}; '
          f'source_count: {protocol.source_count}; warmup_count: {protocol.warmup_count}')
    device = torch.device(args.device)
    print(f'device: {device}; training target: {args.max_steps}; fresh state: True')
    sources = sl.ActivityNetAdapter(dataset_root=args.dataset_root).sequence_sources('training')[:protocol.source_count]
    validate_sources(
        sources, stream_mode=protocol.stream_mode,
        round_robin_stream_count=protocol.round_robin_stream_count,
    )
    print(f'ActivityNet training first {protocol.source_count} sources: {[source.sequence_id for source in sources]}')
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    moco = ViTLoRAMoCo(ViTLoRAFrameEncoder(CHECKPOINT_ID)).to(device).train()
    run_streaming_moco(moco, processor, sources, device, args.max_steps, protocol=protocol)
    print(f'Streaming MoCo {args.max_steps}-step mechanics: PASS')


if __name__ == '__main__':
    main()
