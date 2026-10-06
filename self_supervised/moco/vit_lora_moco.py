"""Single-clip MoCo v2-style mechanics for the Stage 5 ViT-LoRA smoke."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model.backbones.vit import ViTLoRAFrameEncoder
from utils.configuration import load_config_group
from .negative_selection import select_negatives


_ENCODER_CONFIG = load_config_group('encoder', 'vit_base_patch16_224')
_MOCO_CONFIG = load_config_group('moco', 'stage6b_v2')
FEATURE_SIZE = _ENCODER_CONFIG['feature_size']
PROJECTION_SIZE = _MOCO_CONFIG['projection_size']
QUEUE_CAPACITY = _MOCO_CONFIG['queue_capacity']
MOMENTUM = _MOCO_CONFIG['momentum']
TEMPERATURE = _MOCO_CONFIG['temperature']


def lora_parameters(encoder):
    return {
        name: parameter for name, parameter in encoder.named_parameters()
        if '.lora_A.' in name or '.lora_B.' in name
    }


def require_normalized(vector, projection_size=PROJECTION_SIZE):
    if tuple(vector.shape) != (projection_size,) or not torch.isfinite(vector).all().item():
        raise RuntimeError(f'Expected a finite projected vector [{projection_size}]')
    if not torch.allclose(vector.norm(), vector.new_tensor(1.), rtol=1e-5, atol=1e-6):
        raise RuntimeError('Expected an L2-normalized projected vector')


@dataclass(frozen=True)
class QueueEntry:
    key: torch.Tensor
    sequence_id: str
    sequence_index: int


class MetadataQueue:
    """FIFO storage of detached keys and their source metadata."""

    def __init__(self, capacity=QUEUE_CAPACITY, projection_size=PROJECTION_SIZE):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError('Queue capacity must be a positive integer')
        if not isinstance(projection_size, int) or projection_size <= 0:
            raise ValueError('Queue projection size must be a positive integer')
        self.capacity = capacity
        self.projection_size = projection_size
        self._entries = deque(maxlen=capacity)

    def __len__(self):
        return len(self._entries)

    @property
    def entries(self):
        return tuple(self._entries)

    def enqueue(self, key, sequence_id, sequence_index):
        require_normalized(key, self.projection_size)
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError('sequence_id must be non-empty')
        if type(sequence_index) is not int or sequence_index < 0:
            raise ValueError('sequence_index must be a non-negative integer')
        self._entries.append(QueueEntry(key.detach().clone(), sequence_id, sequence_index))

    def negatives(self, sequence_id):
        """Legacy Stage 5 convenience; streaming policy lives in the selector."""
        return select_negatives(self.entries, sequence_id)


class ViTLoRAMoCo(nn.Module):
    """Independent Query/Key branches; optimizer, EMA and enqueue stay explicit.

    Only Query LoRA and Query projector belong to the optimizer. The caller
    must compute loss from the old queue, step the optimizer, update the key
    branch, then enqueue the already computed positive key, in that order.
    Queue entries are runtime state only; checkpoint/resume is outside Stage 5.
    """

    def __init__(
        self, query_encoder=None, *, config=None, feature_size=None, projection_size=PROJECTION_SIZE,
        queue_capacity=QUEUE_CAPACITY, momentum=MOMENTUM, temperature=TEMPERATURE,
    ):
        super().__init__()
        if config is not None:
            projection_size, queue_capacity = config.projection_size, config.queue_capacity
            momentum, temperature = config.momentum, config.temperature
        self.query_encoder = query_encoder if query_encoder is not None else ViTLoRAFrameEncoder()
        self.feature_size = getattr(self.query_encoder, 'feature_size', FEATURE_SIZE) \
            if feature_size is None else feature_size
        if type(self.feature_size) is not int or self.feature_size <= 0:
            raise ValueError('feature_size must be a positive integer')
        if type(projection_size) is not int or projection_size <= 0:
            raise ValueError('projection_size must be a positive integer')
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise ValueError('queue_capacity must be a positive integer')
        if not 0.0 <= momentum < 1.0 or temperature <= 0:
            raise ValueError('momentum must be in [0, 1) and temperature must be positive')
        self.projection_size = projection_size
        self.key_encoder = deepcopy(self.query_encoder).requires_grad_(False)
        self.query_projector = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.projection_size),
        )
        self.key_projector = deepcopy(self.query_projector).requires_grad_(False)
        self.queue = MetadataQueue(queue_capacity, self.projection_size)
        self.momentum = momentum
        self.temperature = temperature

    def query_parameters(self):
        return list(lora_parameters(self.query_encoder).values()) + list(self.query_projector.parameters())

    def project_query(self, clip_feature):
        q = F.normalize(self.query_projector(clip_feature), dim=-1)
        require_normalized(q, self.projection_size)
        return q

    @torch.no_grad()
    def project_key(self, clip_feature):
        k = F.normalize(self.key_projector(clip_feature), dim=-1)
        require_normalized(k, self.projection_size)
        return k

    def contrastive_loss(self, query, positive_key, sequence_id, *, negatives=None):
        require_normalized(query, self.projection_size)
        require_normalized(positive_key, self.projection_size)
        entries = self.queue.negatives(sequence_id) if negatives is None else tuple(negatives)
        if not entries:
            raise RuntimeError('No valid negatives supplied for InfoNCE')
        negative_keys = torch.stack([entry.key.detach() for entry in entries])
        similarities = torch.cat(((query @ positive_key.detach()).reshape(1), negative_keys @ query))
        logits = similarities.unsqueeze(0) / self.temperature
        loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=query.device))
        if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
            raise RuntimeError('InfoNCE logits or loss contain NaN or Inf')
        return loss, logits, entries

    @torch.no_grad()
    def update_key(self):
        pairs = (
            (lora_parameters(self.query_encoder), lora_parameters(self.key_encoder)),
            (dict(self.query_projector.named_parameters()), dict(self.key_projector.named_parameters())),
        )
        for query_parameters, key_parameters in pairs:
            for name, key in key_parameters.items():
                key.mul_(self.momentum).add_(query_parameters[name], alpha=1. - self.momentum)
