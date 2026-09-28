"""Independent protocol axes and historical presets for streaming MoCo."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamingMoCoProtocol:
    stream_mode: str = 'round_robin'
    key_transform: str = 'horizontal_flip'
    negative_policy: str = 'different_sequence'

    def __post_init__(self):
        for name, allowed in (
            ('stream_mode', ('round_robin', 'strict_single')),
            ('key_transform', ('horizontal_flip', 'gbr_horizontal_flip')),
            ('negative_policy', ('different_sequence', 'all_past')),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f'{name} must be one of {allowed}')

    @property
    def source_count(self):
        return 4 if self.stream_mode == 'round_robin' else 1

    @property
    def warmup_count(self):
        return self.source_count


STAGE6A_PROTOCOL = StreamingMoCoProtocol()
STAGE6B_PROTOCOL = StreamingMoCoProtocol('strict_single', 'gbr_horizontal_flip', 'all_past')
