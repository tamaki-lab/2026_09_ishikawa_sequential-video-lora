"""Chronological SequentialSample streams, shared by training and evaluation."""

from contextlib import ExitStack, contextmanager

import sequential_loader as sl
import torch


FRAMES_PER_CHUNK = 16
STREAM_COUNT = 4
STREAM_MODES = ('round_robin', 'strict_single')


def source_count(stream_mode):
    if stream_mode not in STREAM_MODES:
        raise ValueError(f'stream_mode must be one of {STREAM_MODES}')
    return STREAM_COUNT if stream_mode == 'round_robin' else 1


def validate_sources(sources, stream_mode='round_robin'):
    count = source_count(stream_mode)
    if len(sources) != count or len({source.sequence_id for source in sources}) != count:
        description = 'four distinct sources' if count == 4 else 'one source for strict_single'
        raise ValueError(f'Expected exactly {description}')


@contextmanager
def ordered_samples(sources, stream_mode='round_robin'):
    """Yield one or four chronological streams without prefetch; always close readers.

    A final sample is consumed normally, then the whole iterator ends, even if
    the other streams still have samples. Timestamps retain their raw ordering.
    """
    validate_sources(sources, stream_mode)
    with ExitStack() as stack:
        streams = []
        for source in sources:
            dataset = sl.SequentialDataset(
                sources=(source,), reader=sl.SequentialVideoReader(),
                chunk_config=sl.FixedChunkConfig(frames_per_chunk=FRAMES_PER_CHUNK),
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
                    if sample.frames.ndim != 4 or tuple(sample.frames.shape[:2]) != (16, 3):
                        raise RuntimeError('Expected 16 RGB frames including padding')
                    if sample.frames.device.type != 'cpu' or sample.frames.dtype != torch.uint8:
                        raise RuntimeError('Expected CPU uint8 frames')
                    if sample.valid_mask.shape != (16,) or sample.valid_mask.dtype != torch.bool:
                        raise RuntimeError('Expected bool valid_mask [16]')
                    count = int(sample.valid_mask.sum().item())
                    if count == 0 and sample.is_last:
                        return
                    start = source.start_frame + index * FRAMES_PER_CHUNK
                    if not count or not torch.equal(sample.valid_mask, torch.arange(16) < count) or not torch.equal(
                        sample.frame_indices[sample.valid_mask], torch.arange(start, start + count)
                    ) or (count != FRAMES_PER_CHUNK and not sample.is_last):
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
