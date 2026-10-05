"""Raw pre-projector [768] segment features for Base ViT and Query LoRA.

valid RGB frames -> frame CLS [T, 768] -> Masked Mean -> chunk [768]
-> simple mean over the manifest chunks -> segment [768] float32.
No normalization, centering or other post-processing for either condition.
"""

import io
from pathlib import Path

import torch

from integration.sequential_vit import encode_chunk
from model.aggregators import MaskedMeanClipAggregator
from integration.sequential_stream import ordered_samples
from utils.artifact_io import read_json, sha256_file, write_bytes_atomic, write_json_atomic
from .activitynet_manifest import SPLITS, chunk_in_segment


FEATURE_SCHEMA = 'activitynet-segment-features/v1'
FEATURE_SIZE = 768
CONDITIONS = ('base_vit', 'moco_query_lora_final')
FEATURE_FILES = tuple(f'{split}/features.pt' for split in SPLITS)


@torch.no_grad()
def chunk_feature(sample, processor, encoder, device):
    frames = encode_chunk(sample, processor, encoder, device)[2]
    feature = MaskedMeanClipAggregator()(frames, sample.valid_mask).float().cpu()
    if tuple(feature.shape) != (FEATURE_SIZE,) or not torch.isfinite(feature).all().item():
        raise RuntimeError('Expected a finite chunk feature [768]')
    return feature


def video_segment_features(source, rows, processor, encoder, device):
    """Encode each needed chunk of one video once, then average per segment."""
    needed = {index for row in rows for index in row['chunk_indices']}
    last = max(needed)
    chunks = {}
    with ordered_samples((source,), stream_mode='strict_single') as samples:
        for sample in samples:
            index = sample.sequence_index
            if index in needed:
                for row in rows:
                    if index in row['chunk_indices'] and not chunk_in_segment(
                        sample, row['segment_start'], row['segment_end']
                    ):
                        raise RuntimeError(f'Chunk {index} no longer lies in segment {row["segment_id"]}')
                chunks[index] = chunk_feature(sample, processor, encoder, device)
            if index >= last:
                break
    if set(chunks) != needed:
        raise RuntimeError(f'Missing manifest chunks for {source.sequence_id}: {sorted(needed - set(chunks))}')
    return [torch.stack([chunks[index] for index in row['chunk_indices']]).mean(dim=0) for row in rows]


def extract_split(rows, sources, processor, encoder, device, progress=None):
    """Features in manifest row order for one split."""
    by_id = {source.sequence_id: source for source in sources}
    features, videos = [], []
    for row in rows:
        if not videos or videos[-1][0] != row['video_id']:
            videos.append((row['video_id'], []))
        videos[-1][1].append(row)
    if len({video for video, _ in videos}) != len(videos):
        raise RuntimeError('Manifest rows of one video must be contiguous')
    for position, (video, video_rows) in enumerate(videos):
        features.extend(video_segment_features(by_id[video], video_rows, processor, encoder, device))
        if progress:
            progress(position + 1, len(videos))
    return {
        'features': torch.stack(features).float() if features else torch.empty(0, FEATURE_SIZE),
        'labels': torch.tensor([row['label_id'] for row in rows], dtype=torch.int64),
        'segment_ids': [row['segment_id'] for row in rows],
    }


def save_split(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(path, buffer.getvalue())


def verify_split(value, rows, split):
    """Shape, dtype, device and exact manifest order of one split."""
    expected = [row for row in rows if row['split'] == split]
    features, labels = value['features'], value['labels']
    if value['segment_ids'] != [row['segment_id'] for row in expected]:
        raise RuntimeError(f'{split} segment IDs differ from the manifest order or count')
    if features.dtype != torch.float32 or features.device.type != 'cpu' or tuple(features.shape) != (
        len(expected), FEATURE_SIZE
    ) or features.requires_grad or not torch.isfinite(features).all().item():
        raise RuntimeError(f'{split} features must be finite float32 CPU [N, 768]')
    if labels.dtype != torch.int64 or labels.tolist() != [row['label_id'] for row in expected]:
        raise RuntimeError(f'{split} labels differ from the manifest')


def load_features(directory, rows, manifest_metadata, condition=None):
    """Load a feature artifact and verify hashes and manifest alignment."""
    directory = Path(directory)
    metadata = read_json(directory / 'metadata.json')
    if metadata.get('schema') != FEATURE_SCHEMA:
        raise RuntimeError(f'Unsupported feature schema: {metadata.get("schema")}')
    if condition is not None and metadata['condition'] != condition:
        raise RuntimeError(f'Expected {condition} features, got {metadata["condition"]}')
    for key in ('segment_manifest_sha256', 'label_mapping_sha256'):
        if metadata['manifest'][key] != manifest_metadata[key]:
            raise RuntimeError(f'Feature artifact was extracted from a different manifest ({key})')
    if {file: sha256_file(directory / file) for file in FEATURE_FILES} != metadata['files']:
        raise RuntimeError('Feature files differ from metadata hashes')
    splits = {}
    for split in SPLITS:
        splits[split] = torch.load(directory / split / 'features.pt', map_location='cpu', weights_only=True)
        verify_split(splits[split], rows, split)
    return splits, metadata


def write_feature_artifact(directory, splits, rows, metadata):
    directory = Path(directory)
    for split in SPLITS:
        verify_split(splits[split], rows, split)
        save_split(directory / split / 'features.pt', splits[split])
    metadata = {**metadata, 'schema': FEATURE_SCHEMA,
                'files': {file: sha256_file(directory / file) for file in FEATURE_FILES}}
    write_json_atomic(directory / 'metadata.json', metadata)
    return metadata
