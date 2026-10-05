"""Independent protocol axes and historical presets for streaming MoCo."""

from dataclasses import dataclass

from integration.sequential_stream import STREAM_MODES, source_count


@dataclass(frozen=True)
class StreamingMoCoProtocol:
    stream_mode: str = 'round_robin'
    key_transform: str = 'horizontal_flip'
    negative_policy: str = 'different_sequence'

    def __post_init__(self):
        for name, allowed in (
            ('stream_mode', STREAM_MODES),
            ('key_transform', ('horizontal_flip', 'gbr_horizontal_flip')),
            ('negative_policy', ('different_sequence', 'all_past')),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f'{name} must be one of {allowed}')

    @property
    def source_count(self):
        return source_count(self.stream_mode)

    @property
    def warmup_count(self):
        return self.source_count


STAGE6A_PROTOCOL = StreamingMoCoProtocol()
STAGE6B_PROTOCOL = StreamingMoCoProtocol('strict_single', 'gbr_horizontal_flip', 'all_past')
