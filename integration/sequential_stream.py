"""Chronological SequentialSample streams, shared by training and evaluation."""

from contextlib import ExitStack, contextmanager

import sequential_loader as sl
import torch

from utils.configuration import load_config_group


_SEQUENTIAL_CONFIG = load_config_group('sequential', 'default')
FRAMES_PER_CHUNK = _SEQUENTIAL_CONFIG['frames_per_chunk']
STREAM_COUNT = _SEQUENTIAL_CONFIG['round_robin_stream_count']
STREAM_MODES = ('round_robin', 'strict_single')


def source_count(stream_mode, round_robin_stream_count=STREAM_COUNT):
    if stream_mode not in STREAM_MODES:
        raise ValueError(f'stream_mode must be one of {STREAM_MODES}')
    if type(round_robin_stream_count) is not int or round_robin_stream_count <= 0:
        raise ValueError('round_robin_stream_count must be a positive integer')
    return round_robin_stream_count if stream_mode == 'round_robin' else 1


def validate_sources(sources, stream_mode='round_robin', *, round_robin_stream_count=STREAM_COUNT):
    count = source_count(stream_mode, round_robin_stream_count)
    if len(sources) != count or len({source.sequence_id for source in sources}) != count:
        description = ('four distinct sources' if count == 4 else f'{count} distinct sources') \
            if stream_mode == 'round_robin' else 'one source for strict_single'
        raise ValueError(f'Expected exactly {description}')


@contextmanager
def ordered_samples(
    sources, stream_mode='round_robin', *, frames_per_chunk=FRAMES_PER_CHUNK,
    round_robin_stream_count=STREAM_COUNT,
):
    """Yield one or four chronological streams without prefetch; always close readers.

    A final sample is consumed normally, then the whole iterator ends, even if
    the other streams still have samples. Timestamps retain their raw ordering.
    """
    if type(frames_per_chunk) is not int or frames_per_chunk <= 0:
        raise ValueError('frames_per_chunk must be a positive integer')
    validate_sources(sources, stream_mode, round_robin_stream_count=round_robin_stream_count)
    with ExitStack() as stack:
        streams = []
        for source in sources:
            dataset = sl.SequentialDataset(
                sources=(source,), reader=sl.SequentialVideoReader(),
                chunk_config=sl.FixedChunkConfig(frames_per_chunk=frames_per_chunk),
            )
            loader = sl.build_sequential_dataloader(dataset=dataset)
            streams.append(stack.enter_context(sl.sequential_sample_stream(loader)))

        def iterate():
            index = 0
            while True:
                for source, stream in zip(sources, streams):
                    sample = next(stream, None)
                    if sample is None:
                        return
                    if not isinstance(sample, sl.SequentialSample):
                        raise RuntimeError('Expected SequentialSample')
                    if (sample.sequence_id, sample.sequence_index, sample.is_first) != (
                        source.sequence_id, index, index == 0
                    ):
                        raise RuntimeError('Stream sequence identity or chronological chunk order changed')
                    if sample.frames.ndim != 4 or tuple(sample.frames.shape[:2]) != (frames_per_chunk, 3):
                        raise RuntimeError(f'Expected {frames_per_chunk} RGB frames including padding')
                    if sample.frames.device.type != 'cpu' or sample.frames.dtype != torch.uint8:
                        raise RuntimeError('Expected CPU uint8 frames')
                    if sample.valid_mask.shape != (frames_per_chunk,) or sample.valid_mask.dtype != torch.bool:
                        raise RuntimeError(f'Expected bool valid_mask [{frames_per_chunk}]')
                    count = int(sample.valid_mask.sum().item())
                    if count == 0 and sample.is_last:
                        return
                    start = source.start_frame + index * frames_per_chunk
                    if not count or not torch.equal(
                        sample.valid_mask, torch.arange(frames_per_chunk) < count
                    ) or not torch.equal(
                        sample.frame_indices[sample.valid_mask], torch.arange(start, start + count)
                    ) or (count != frames_per_chunk and not sample.is_last):
                        raise RuntimeError('Expected contiguous absolute frame indices and terminal-only padding')
                    yield sample
                    if sample.is_last:
                        return
                index += 1

        samples = iterate()
        try:
            yield samples
        finally:
            samples.close()
