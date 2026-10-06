"""Typed resolved configuration for Streaming MoCo.

Hydra owns composition at the command boundary.  Training code receives these
immutable values, while the canonical production contract is always loaded
independently from the repository group YAMLs.
"""

from dataclasses import dataclass
from pathlib import Path

from training.moco_protocol import StreamingMoCoProtocol
from utils.configuration import load_config_group, validate_path_component


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


@dataclass(frozen=True)
class ActivityNetConfig:
    name: str
    version: str
    annotation_version: str
    splits: dict
    expected_source_counts: dict
    class_count: int

    @classmethod
    def from_mapping(cls, value):
        result = cls(
            name=str(value['name']), version=str(value['version']),
            annotation_version=str(value['annotation_version']), splits=dict(value['splits']),
            expected_source_counts={key: int(count) for key, count in value['expected_source_counts'].items()},
            class_count=int(value['class_count']),
        )
        if set(result.splits) != {'training', 'validation'} or set(result.expected_source_counts) != set(
            result.splits
        ):
            raise ValueError('ActivityNet config requires training and validation split contracts')
        _positive_int('activitynet.class_count', result.class_count)
        for split, count in result.expected_source_counts.items():
            _positive_int(f'activitynet.expected_source_counts.{split}', count)
        return result


@dataclass(frozen=True)
class LoRAConfig:
    target_modules: tuple
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str

    @classmethod
    def from_mapping(cls, value):
        result = cls(
            target_modules=tuple(value['target_modules']), r=value['r'], lora_alpha=value['lora_alpha'],
            lora_dropout=float(value['lora_dropout']), bias=str(value['bias']),
        )
        if not result.target_modules or any(not isinstance(name, str) or not name for name in result.target_modules):
            raise ValueError('encoder.lora.target_modules must contain non-empty names')
        _positive_int('encoder.lora.r', result.r)
        _positive_int('encoder.lora.lora_alpha', result.lora_alpha)
        if not 0.0 <= result.lora_dropout < 1.0:
            raise ValueError('encoder.lora.lora_dropout must be in [0, 1)')
        return result

    def metadata(self):
        return {
            'target_modules': sorted(self.target_modules), 'r': self.r, 'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout, 'bias': self.bias,
        }


@dataclass(frozen=True)
class EncoderConfig:
    checkpoint_id: str
    feature_size: int
    image_size: int
    channels: int
    lora: LoRAConfig

    @classmethod
    def from_mapping(cls, value):
        result = cls(
            checkpoint_id=str(value['checkpoint_id']), feature_size=value['feature_size'],
            image_size=value['image_size'], channels=value['channels'], lora=LoRAConfig.from_mapping(value['lora']),
        )
        if not result.checkpoint_id:
            raise ValueError('encoder.checkpoint_id must be non-empty')
        for name in ('feature_size', 'image_size', 'channels'):
            _positive_int(f'encoder.{name}', getattr(result, name))
        return result


@dataclass(frozen=True)
class SequentialConfig:
    frames_per_chunk: int
    round_robin_stream_count: int

    @classmethod
    def from_mapping(cls, value):
        result = cls(value['frames_per_chunk'], value['round_robin_stream_count'])
        _positive_int('sequential.frames_per_chunk', result.frames_per_chunk)
        _positive_int('sequential.round_robin_stream_count', result.round_robin_stream_count)
        return result


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    betas: tuple
    eps: float

    @classmethod
    def from_mapping(cls, value):
        result = cls(
            str(value['name']), float(value['lr']), float(value['weight_decay']),
            tuple(float(beta) for beta in value['betas']), float(value['eps']),
        )
        if result.name != 'AdamW':
            raise ValueError('Only the audited AdamW optimizer is supported')
        if result.lr <= 0 or result.weight_decay < 0:
            raise ValueError('optimizer lr must be positive and weight_decay non-negative')
        if len(result.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in result.betas) or result.eps <= 0:
            raise ValueError('optimizer betas must be two values in [0, 1) and eps positive')
        return result


@dataclass(frozen=True)
class IntervalConfig:
    resume_videos: int
    snapshot_videos: int
    comet_metric_updates: int

    @classmethod
    def from_mapping(cls, value):
        result = cls(value['resume_videos'], value['snapshot_videos'], value['comet_metric_updates'])
        for name in ('resume_videos', 'snapshot_videos', 'comet_metric_updates'):
            _positive_int(f'moco.intervals.{name}', getattr(result, name))
        return result


@dataclass(frozen=True)
class MoCoConfig:
    name: str
    protocol_version: str
    protocol: StreamingMoCoProtocol
    projection_size: int
    queue_capacity: int
    momentum: float
    temperature: float
    optimizer: OptimizerConfig
    intervals: IntervalConfig

    @classmethod
    def from_mapping(cls, value, *, round_robin_stream_count=None):
        if round_robin_stream_count is None:
            round_robin_stream_count = load_config_group(
                'sequential', 'default',
            )['round_robin_stream_count']
        protocol = value['protocol']
        result = cls(
            name=str(value['name']), protocol_version=str(value['protocol_version']),
            protocol=StreamingMoCoProtocol(
                str(protocol['stream_mode']), str(protocol['key_transform']), str(protocol['negative_policy']),
                round_robin_stream_count=round_robin_stream_count,
            ),
            projection_size=value['projection_size'], queue_capacity=value['queue_capacity'],
            momentum=float(value['momentum']), temperature=float(value['temperature']),
            optimizer=OptimizerConfig.from_mapping(value['optimizer']),
            intervals=IntervalConfig.from_mapping(value['intervals']),
        )
        _positive_int('moco.projection_size', result.projection_size)
        _positive_int('moco.queue_capacity', result.queue_capacity)
        if not 0.0 <= result.momentum < 1.0:
            raise ValueError('moco.momentum must be in [0, 1)')
        if result.temperature <= 0:
            raise ValueError('moco.temperature must be positive')
        return result


@dataclass(frozen=True)
class RuntimeConfig:
    dataset_root: Path
    run_id: str
    seed: int
    resume: bool
    device: str
    stop_after_videos: int | None
    output_root: Path

    @classmethod
    def from_mapping(cls, value):
        result = cls(
            dataset_root=Path(value['dataset_root']), run_id=str(value['run_id']), seed=value['seed'],
            resume=value['resume'], device=str(value['device']), stop_after_videos=value['stop_after_videos'],
            output_root=Path(value['output_root']),
        )
        validate_path_component('runtime.run_id', result.run_id)
        if type(result.seed) is not int or not 0 <= result.seed < 2 ** 32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if type(result.resume) is not bool or result.device not in ('cpu', 'cuda'):
            raise ValueError('resume must be boolean and device must be cpu or cuda')
        if result.stop_after_videos is not None:
            _positive_int('runtime.stop_after_videos', result.stop_after_videos)
        return result


@dataclass(frozen=True)
class FullMoCoConfig:
    activitynet: ActivityNetConfig
    encoder: EncoderConfig
    sequential: SequentialConfig
    moco: MoCoConfig
    provenance: dict
    tracking: dict
    runtime: RuntimeConfig
    disable_comet: bool

    @classmethod
    def from_mapping(cls, value):
        sequential = SequentialConfig.from_mapping(value['sequential'])
        logging = value['logging']
        if type(logging['disable_comet']) is not bool:
            raise ValueError('logging.disable_comet must be boolean')
        tracking = dict(value['tracking'])
        artifacts = tracking.get('artifacts') or {}
        if not isinstance(tracking.get('comet_project'), str) or not tracking['comet_project']:
            raise ValueError('tracking.comet_project must be a non-empty string')
        if not isinstance(artifacts.get('moco_snapshot'), str) or not artifacts['moco_snapshot']:
            raise ValueError('tracking.artifacts.moco_snapshot must be a non-empty string')
        return cls(
            activitynet=ActivityNetConfig.from_mapping(value['activitynet']),
            encoder=EncoderConfig.from_mapping(value['encoder']), sequential=sequential,
            moco=MoCoConfig.from_mapping(
                value['moco'], round_robin_stream_count=sequential.round_robin_stream_count,
            ),
            provenance=dict(value['provenance']), tracking=tracking,
            runtime=RuntimeConfig.from_mapping(value['runtime']),
            disable_comet=logging['disable_comet'],
        )

    def production_groups_match(self):
        """Compare against repository canonical groups, never against this config itself."""
        canonical_sequential = SequentialConfig.from_mapping(load_config_group('sequential', 'default'))
        return (
            self.activitynet == ActivityNetConfig.from_mapping(load_config_group('activitynet', 'v1_3'))
            and self.encoder == EncoderConfig.from_mapping(load_config_group('encoder', 'vit_base_patch16_224'))
            and self.sequential == canonical_sequential
            and self.moco == MoCoConfig.from_mapping(
                load_config_group('moco', 'stage6b_v2'),
                round_robin_stream_count=canonical_sequential.round_robin_stream_count,
            )
            and self.provenance == load_config_group('provenance', 'research_v1')
        )


def canonical_activitynet_config():
    return ActivityNetConfig.from_mapping(load_config_group('activitynet', 'v1_3'))


def canonical_encoder_config():
    return EncoderConfig.from_mapping(load_config_group('encoder', 'vit_base_patch16_224'))


def canonical_sequential_config():
    return SequentialConfig.from_mapping(load_config_group('sequential', 'default'))


def canonical_moco_config():
    sequential = canonical_sequential_config()
    return MoCoConfig.from_mapping(
        load_config_group('moco', 'stage6b_v2'),
        round_robin_stream_count=sequential.round_robin_stream_count,
    )
