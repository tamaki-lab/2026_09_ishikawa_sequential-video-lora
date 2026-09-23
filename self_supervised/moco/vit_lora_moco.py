"""Single-clip MoCo v2-style mechanics for the Stage 5 ViT-LoRA smoke."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model.backbones.vit import ViTLoRAFrameEncoder


FEATURE_SIZE = 768
PROJECTION_SIZE = 128
QUEUE_CAPACITY = 4096
MOMENTUM = 0.999
TEMPERATURE = 0.07


def lora_parameters(encoder):
    return {
        name: parameter for name, parameter in encoder.named_parameters()
        if '.lora_A.' in name or '.lora_B.' in name
    }


def require_normalized(vector):
    if tuple(vector.shape) != (PROJECTION_SIZE,) or not torch.isfinite(vector).all().item():
        raise RuntimeError('Expected a finite projected vector [128]')
    if not torch.allclose(vector.norm(), vector.new_tensor(1.), rtol=1e-5, atol=1e-6):
        raise RuntimeError('Expected an L2-normalized projected vector')


@dataclass(frozen=True)
class QueueEntry:
    key: torch.Tensor
    sequence_id: str
    sequence_index: int


class MetadataQueue:
    """Empty FIFO of detached keys; filter same-video entries at read time."""

    def __init__(self, capacity=QUEUE_CAPACITY):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError('Queue capacity must be a positive integer')
        self.capacity = capacity
        self._entries = deque(maxlen=capacity)

    def __len__(self):
        return len(self._entries)

    @property
    def entries(self):
        return tuple(self._entries)

    def enqueue(self, key, sequence_id, sequence_index):
        require_normalized(key)
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError('sequence_id must be non-empty')
        if type(sequence_index) is not int or sequence_index < 0:
            raise ValueError('sequence_index must be a non-negative integer')
        self._entries.append(QueueEntry(key.detach().clone(), sequence_id, sequence_index))

    def negatives(self, sequence_id):
        entries = tuple(entry for entry in self._entries if entry.sequence_id != sequence_id)
        if not entries:
            raise RuntimeError('No valid different-sequence negatives in the queue')
        return entries


class ViTLoRAMoCo(nn.Module):
    """Independent Query/Key branches; optimizer, EMA and enqueue stay explicit.

    Only Query LoRA and Query projector belong to the optimizer. The caller
    must compute loss from the old queue, step the optimizer, update the key
    branch, then enqueue the already computed positive key, in that order.
    Queue entries are runtime state only; checkpoint/resume is outside Stage 5.
    """

    def __init__(self, query_encoder=None):
        super().__init__()
        self.query_encoder = query_encoder if query_encoder is not None else ViTLoRAFrameEncoder()
        self.key_encoder = deepcopy(self.query_encoder).requires_grad_(False)
        self.query_projector = nn.Sequential(
            nn.Linear(FEATURE_SIZE, FEATURE_SIZE),
            nn.ReLU(),
            nn.Linear(FEATURE_SIZE, PROJECTION_SIZE),
        )
        self.key_projector = deepcopy(self.query_projector).requires_grad_(False)
        self.queue = MetadataQueue()
        self.momentum = MOMENTUM
        self.temperature = TEMPERATURE

    def query_parameters(self):
        return list(lora_parameters(self.query_encoder).values()) + list(self.query_projector.parameters())

    def project_query(self, clip_feature):
        q = F.normalize(self.query_projector(clip_feature), dim=-1)
        require_normalized(q)
        return q

    @torch.no_grad()
    def project_key(self, clip_feature):
        k = F.normalize(self.key_projector(clip_feature), dim=-1)
        require_normalized(k)
        return k

    def contrastive_loss(self, query, positive_key, sequence_id):
        require_normalized(query)
        require_normalized(positive_key)
        entries = self.queue.negatives(sequence_id)
        negatives = torch.stack([entry.key for entry in entries])
        similarities = torch.cat(((query @ positive_key.detach()).reshape(1), negatives @ query))
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
