"""Stage 6A: four-stream MoCo smoke / canary, each invocation a fresh run.

Usage:
    python smoke_activitynet_vit_lora_moco_multistep.py /path/to/ActivityNet --max-steps 10
    python smoke_activitynet_vit_lora_moco_multistep.py /path/to/ActivityNet --max-steps 100
"""

import argparse
from pathlib import Path

import sequential_loader as sl
import torch
from transformers import AutoImageProcessor

from model.backbones.vit import ViTLoRAFrameEncoder
from self_supervised.moco import ViTLoRAMoCo
from smoke_activitynet_vit_lora_moco_one_step import audit_provenance
from smoke_activitynet_vit_lora_one_step import CHECKPOINT_ID, git_output
from training.moco_canary import run_canary, validate_sources


BASE_COMMIT = '1cbaa4a0fde2fd196a36feb3eeb0074709b52cd9'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--max-steps', type=int, default=10, help='Training updates; excludes four warm-up keys')
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 4096 - 4:
        parser.error('--max-steps must be positive and leave room for four warm-up keys (maximum 4092)')
    audit_provenance()
    git_output(Path(__file__).resolve().parent, 'merge-base', '--is-ancestor', BASE_COMMIT, 'HEAD')
    print(f'Stage 6A implementation base: {BASE_COMMIT}')
    device = torch.device(args.device)
    print(f'device: {device}; training target: {args.max_steps}; fresh state: True')
    sources = sl.ActivityNetAdapter(dataset_root=args.dataset_root).sequence_sources('training')[:4]
    validate_sources(sources)
    print(f'ActivityNet training first four sources: {[source.sequence_id for source in sources]}')
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT_ID)
    moco = ViTLoRAMoCo(ViTLoRAFrameEncoder(CHECKPOINT_ID)).to(device).train()
    run_canary(moco, processor, sources, device, args.max_steps)
    print(f'Stage 6A {args.max_steps}-step mechanics: PASS')


if __name__ == '__main__':
    main()
