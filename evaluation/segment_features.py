"""Raw pre-projector segment features for Base ViT and Query LoRA.

valid RGB frames -> frame CLS [T, D] -> Masked Mean -> chunk [D]
-> simple mean over the manifest chunks -> segment [D] float32.
No normalization, centering or other post-processing for either condition.
"""

import io
from pathlib import Path

import torch

from integration.sequential_vit import encode_chunk
from model.aggregators import MaskedMeanClipAggregator
from integration.sequential_stream import ordered_samples
from utils.artifact_io import canonical_json_bytes, read_json, sha256_bytes, sha256_file
from utils.artifact_io import write_bytes_atomic, write_json_atomic
from utils.configuration import load_config_group
from .activitynet_manifest import SPLITS, chunk_in_segment


FEATURE_SCHEMA = 'activitynet-segment-features/v3'
FEATURE_SIZE = load_config_group('encoder', 'vit_base_patch16_224')['feature_size']
FRAMES_PER_CHUNK = load_config_group('sequential', 'default')['frames_per_chunk']
CONDITIONS = ('base_vit', 'moco_query_lora_final')
FEATURE_FILES = tuple(f'{split}/features.pt' for split in SPLITS)
# The only definition `extract_split` implements. A Hydra preset declares it
# for the science identity, but cannot change it without a code change.
FEATURE_DEFINITION = {
    'schema': 'activitynet-segment-feature-definition/v2',
    'frame_feature': 'ViT CLS without pooler', 'chunk_aggregation': 'masked mean over valid frames',
    'segment_aggregation': 'mean over manifest chunks', 'dtype': 'float32', 'normalization': None,
}


def validate_feature_definition(definition):
    if definition != FEATURE_DEFINITION:
        raise ValueError(f'Only the implemented feature definition is supported: {FEATURE_DEFINITION}')
    return definition


def shared_contract_sha256(contract):
    return sha256_bytes(canonical_json_bytes(contract))


def validate_shared_contract(metadata):
    contract = metadata.get('shared_feature_contract')
    recorded = metadata.get('shared_feature_contract_sha256')
    if not isinstance(contract, dict) or shared_contract_sha256(contract) != recorded:
        raise RuntimeError('Feature shared contract hash does not match its metadata')
    if 'science' in metadata and metadata['science'] != contract.get('science'):
        raise RuntimeError('Feature science metadata differs from the shared contract')
    if metadata.get('selection_sha256') != contract.get('selection_sha256'):
        raise RuntimeError('Feature selection metadata differs from the shared contract')
    return contract


def validate_feature_pair(base_metadata, lora_metadata):
    """Require every comparison axis except the intended LoRA condition to match."""
    base = validate_shared_contract(base_metadata)
    lora = validate_shared_contract(lora_metadata)
    if base != lora or base_metadata['shared_feature_contract_sha256'] != lora_metadata[
        'shared_feature_contract_sha256'
    ]:
        raise RuntimeError('Base / LoRA shared feature contracts differ')
    return base


def feature_size_from_metadata(metadata):
    size = (metadata.get('feature_definition') or {}).get('size')
    if type(size) is not int or size < 1:
        raise RuntimeError('Feature metadata must define a positive integer size')
    return size


@torch.no_grad()
def chunk_feature(sample, processor, encoder, device, feature_size=None):
    feature_size = getattr(encoder, 'feature_size', FEATURE_SIZE) if feature_size is None else feature_size
    frames = encode_chunk(sample, processor, encoder, device, feature_size=feature_size)[2]
    feature = MaskedMeanClipAggregator()(frames, sample.valid_mask).float().cpu()
    if tuple(feature.shape) != (feature_size,) or not torch.isfinite(feature).all().item():
        raise RuntimeError(f'Expected a finite chunk feature [{feature_size}]')
    return feature


def video_segment_features(
    source, rows, processor, encoder, device, feature_size=None, frames_per_chunk=None,
):
    """Encode each needed chunk of one video once, then average per segment."""
    feature_size = getattr(encoder, 'feature_size', FEATURE_SIZE) if feature_size is None else feature_size
    frames_per_chunk = FRAMES_PER_CHUNK if frames_per_chunk is None else frames_per_chunk
    needed = {index for row in rows for index in row['chunk_indices']}
    last = max(needed)
    chunks = {}
    with ordered_samples(
        (source,), stream_mode='strict_single', frames_per_chunk=frames_per_chunk,
    ) as samples:
        for sample in samples:
            index = sample.sequence_index
            if index in needed:
                for row in rows:
                    if index in row['chunk_indices'] and not chunk_in_segment(
                        sample, row['segment_start'], row['segment_end']
                    ):
                        raise RuntimeError(f'Chunk {index} no longer lies in segment {row["segment_id"]}')
                chunks[index] = chunk_feature(sample, processor, encoder, device, feature_size)
            if index >= last:
                break
    if set(chunks) != needed:
        raise RuntimeError(f'Missing manifest chunks for {source.sequence_id}: {sorted(needed - set(chunks))}')
    return [torch.stack([chunks[index] for index in row['chunk_indices']]).mean(dim=0) for row in rows]


def extract_split(
    rows, sources, processor, encoder, device, feature_size=None, frames_per_chunk=None, progress=None,
):
    """Features in manifest row order for one split."""
    feature_size = getattr(encoder, 'feature_size', FEATURE_SIZE) if feature_size is None else feature_size
    frames_per_chunk = FRAMES_PER_CHUNK if frames_per_chunk is None else frames_per_chunk
    by_id = {source.sequence_id: source for source in sources}
    features, videos = [], []
    for row in rows:
        if not videos or videos[-1][0] != row['video_id']:
            videos.append((row['video_id'], []))
        videos[-1][1].append(row)
    if len({video for video, _ in videos}) != len(videos):
        raise RuntimeError('Manifest rows of one video must be contiguous')
    for position, (video, video_rows) in enumerate(videos):
        features.extend(video_segment_features(
            by_id[video], video_rows, processor, encoder, device, feature_size, frames_per_chunk,
        ))
        if progress:
            progress(position + 1, len(videos))
    return {
        'features': torch.stack(features).float() if features else torch.empty(0, feature_size),
        'labels': torch.tensor([row['label_id'] for row in rows], dtype=torch.int64),
        'segment_ids': [row['segment_id'] for row in rows],
    }


def save_split(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(path, buffer.getvalue())


def verify_split(value, rows, split, feature_size=None):
    """Shape, dtype, device and exact manifest order of one split."""
    feature_size = FEATURE_SIZE if feature_size is None else feature_size
    expected = [row for row in rows if row['split'] == split]
    features, labels = value['features'], value['labels']
    if value['segment_ids'] != [row['segment_id'] for row in expected]:
        raise RuntimeError(f'{split} segment IDs differ from the manifest order or count')
    if features.dtype != torch.float32 or features.device.type != 'cpu' or tuple(features.shape) != (
        len(expected), feature_size
    ) or features.requires_grad or not torch.isfinite(features).all().item():
        raise RuntimeError(f'{split} features must be finite float32 CPU [N, {feature_size}]')
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
    feature_size = feature_size_from_metadata(metadata)
    contract = validate_shared_contract(metadata)
    for key in ('segment_manifest_sha256', 'label_mapping_sha256'):
        if metadata['manifest'][key] != manifest_metadata[key]:
            raise RuntimeError(f'Feature artifact was extracted from a different manifest ({key})')
        if contract['manifest'][key] != manifest_metadata[key]:
            raise RuntimeError(f'Feature shared contract differs from the manifest ({key})')
    if {file: sha256_file(directory / file) for file in FEATURE_FILES} != metadata['files']:
        raise RuntimeError('Feature files differ from metadata hashes')
    splits = {}
    for split in SPLITS:
        splits[split] = torch.load(directory / split / 'features.pt', map_location='cpu', weights_only=True)
        verify_split(splits[split], rows, split, feature_size)
    return splits, metadata


def write_feature_artifact(directory, splits, rows, metadata):
    directory = Path(directory)
    validate_shared_contract(metadata)
    feature_size = feature_size_from_metadata(metadata)
    for split in SPLITS:
        verify_split(splits[split], rows, split, feature_size)
        save_split(directory / split / 'features.pt', splits[split])
    metadata = {**metadata, 'schema': FEATURE_SCHEMA,
                'files': {file: sha256_file(directory / file) for file in FEATURE_FILES}}
    write_json_atomic(directory / 'metadata.json', metadata)
    return metadata
