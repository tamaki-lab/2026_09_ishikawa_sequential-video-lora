"""Independent protocol axes and historical presets for streaming MoCo."""

from dataclasses import dataclass

from utils.configuration import load_config_group


STREAM_MODES = ('round_robin', 'strict_single')
_SEQUENTIAL = load_config_group('sequential', 'default')
_STAGE6B = load_config_group('moco', 'stage6b_v2')['protocol']


@dataclass(frozen=True)
class StreamingMoCoProtocol:
    stream_mode: str = 'round_robin'
    key_transform: str = 'horizontal_flip'
    negative_policy: str = 'different_sequence'
    round_robin_stream_count: int = _SEQUENTIAL['round_robin_stream_count']

    def __post_init__(self):
        for name, allowed in (
            ('stream_mode', STREAM_MODES),
            ('key_transform', ('horizontal_flip', 'gbr_horizontal_flip')),
            ('negative_policy', ('different_sequence', 'all_past')),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f'{name} must be one of {allowed}')
        if type(self.round_robin_stream_count) is not int or self.round_robin_stream_count <= 0:
            raise ValueError('round_robin_stream_count must be a positive integer')

    @property
    def source_count(self):
        return self.round_robin_stream_count if self.stream_mode == 'round_robin' else 1

    @property
    def warmup_count(self):
        return self.source_count


STAGE6A_PROTOCOL = StreamingMoCoProtocol(round_robin_stream_count=_SEQUENTIAL['round_robin_stream_count'])
STAGE6B_PROTOCOL = StreamingMoCoProtocol(
    _STAGE6B['stream_mode'], _STAGE6B['key_transform'], _STAGE6B['negative_policy'],
    round_robin_stream_count=_SEQUENTIAL['round_robin_stream_count'],
)
