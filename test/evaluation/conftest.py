"""Synthetic ActivityNet-like videos, annotations and a deterministic encoder."""

from contextlib import contextmanager
import json

import pytest
import sequential_loader as sl
import torch


@pytest.fixture
def videos(monkeypatch):
    state = {'timestamps': {}, 'reads': [], 'opened': [], 'closed': []}

    class Reader:
        @contextmanager
        def open(self, source):
            self.source = source
            state['opened'].append(source.sequence_id)
            try:
                yield self
            finally:
                state['closed'].append(source.sequence_id)

        def read(self, chunk):
            video = self.source.sequence_id
            state['reads'].append((video, chunk.chunk_index))
            times = state['timestamps'][video][chunk.chunk_index]
            count = len(times)
            seed = sum(map(ord, video)) * 11 + chunk.chunk_index * 5
            frames = ((torch.arange(count * 12).reshape(count, 3, 2, 2) * 3 + seed) % 251).to(torch.uint8)
            return sl.DecodedChunk(
                frames=frames, frame_indices=torch.arange(chunk.start_frame, chunk.start_frame + count),
                timestamps=torch.tensor(times, dtype=torch.float64),
                reached_eof=chunk.chunk_index == len(state['timestamps'][video]) - 1,
            )

    monkeypatch.setattr(sl, 'SequentialVideoReader', Reader)
    return state


@pytest.fixture
def dataset(tmp_path, videos):
    """Write a v1.3-shaped annotation file and build sources per split."""
    def make(database, timestamps):
        path = tmp_path / 'activity_net.v1-3.min.json'
        path.write_text(json.dumps({'version': 'VERSION 1.3', 'database': database}))
        videos['timestamps'].update(timestamps)
        return {split: tuple(sl.SequenceSource(
            sequence_id=video, source_id=video, source=tmp_path / f'v_{video}.mp4', start_frame=0, stop_frame=None,
            evaluation_reference=sl.ActivityNetAnnotationReference(annotation_path=path, video_id=video, subset=split),
        ) for video in sorted(database) if database[video]['subset'] == split) for split in ('training', 'validation')}
    return make
