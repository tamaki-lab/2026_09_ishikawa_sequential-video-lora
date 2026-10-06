"""Frozen-feature Linear Probe: fixed hyperparameters, final-epoch evaluation.

The seed only sets the classifier initialization and the training shuffle
order. Validation is evaluated once after the last epoch and never selects
an epoch. Metrics: Top-1 (primary) and Macro class accuracy (secondary).
"""

import csv
from dataclasses import dataclass, replace
import io
import math
from pathlib import Path
from typing import Mapping

import numpy
import torch
from torch import nn
from torch.nn import functional as F

from utils.artifact_io import read_json, sha256_file, write_bytes_atomic, write_json_atomic
from utils.configuration import load_config_group


RESULT_SCHEMA = 'activitynet-linear-probe-result/v2'
AGGREGATE_SCHEMA = 'activitynet-linear-probe-aggregate/v2'
RESULT_FILES = ('probe_classifier.pt', 'summary.json', 'history.csv', 'per_class_accuracy.csv',
                'confusion_matrix.npy')


@dataclass(frozen=True)
class ProbeConfig:
    """Resolved scientific probe config shared by execution and metadata."""

    protocol: str
    seeds: tuple
    classifier_bias: bool
    loss: str
    class_weight: object
    optimizer: str
    lr: float
    weight_decay: float
    betas: tuple
    eps: float
    batch_size: int
    epochs: int
    scheduler: object
    train_shuffle: bool
    validation_shuffle: bool
    early_stopping: bool

    @classmethod
    def from_mapping(cls, protocol, value: Mapping):
        classifier = value['classifier']
        loss = value['loss']
        optimizer = value['optimizer']
        return cls(
            protocol=protocol,
            seeds=tuple(value['seeds']),
            classifier_bias=classifier['bias'],
            loss=loss['name'],
            class_weight=loss['class_weight'],
            optimizer=optimizer['name'],
            lr=optimizer['lr'],
            weight_decay=optimizer['weight_decay'],
            betas=tuple(optimizer['betas']),
            eps=optimizer['eps'],
            batch_size=value['batch_size'],
            epochs=value['epochs'],
            scheduler=value['scheduler'],
            train_shuffle=value['train_shuffle'],
            validation_shuffle=value['validation_shuffle'],
            early_stopping=value['early_stopping'],
        )

    def __post_init__(self):
        if not isinstance(self.protocol, str) or not self.protocol:
            raise ValueError('protocol must be a non-empty string')
        if not self.seeds or any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError('seeds must be non-empty non-negative integers')
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError('seeds must be unique')
        if self.classifier_bias is not True:
            raise ValueError('Only a biased linear classifier is implemented')
        if self.loss != 'cross_entropy' or self.class_weight is not None:
            raise ValueError('Only unweighted cross_entropy is implemented')
        if self.optimizer != 'AdamW':
            raise ValueError('Only AdamW is implemented')
        if not isinstance(self.lr, (int, float)) or self.lr <= 0:
            raise ValueError('lr must be positive')
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0:
            raise ValueError('weight_decay must be non-negative')
        if len(self.betas) != 2 or any(not isinstance(beta, float) or not 0 <= beta < 1 for beta in self.betas):
            raise ValueError('betas must be two floats in [0, 1)')
        if not isinstance(self.eps, float) or self.eps <= 0:
            raise ValueError('eps must be a positive float')
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if type(self.epochs) is not int or self.epochs < 0:
            raise ValueError('epochs must be a non-negative integer')
        if self.scheduler is not None or self.train_shuffle is not True or self.validation_shuffle is not False:
            raise ValueError('Only shuffled training without a scheduler or validation shuffle is implemented')
        if self.early_stopping is not False:
            raise ValueError('Early stopping is not implemented')

    def with_epochs(self, epochs):
        return replace(self, epochs=epochs)

    def hyperparameters(self, feature_size, class_count):
        return {
            'classifier': {
                'class': 'Linear', 'in_features': feature_size, 'out_features': class_count,
                'bias': self.classifier_bias,
            },
            'loss': {'name': self.loss, 'class_weight': self.class_weight},
            'optimizer': {'name': self.optimizer, 'lr': self.lr, 'weight_decay': self.weight_decay,
                          'betas': list(self.betas), 'eps': self.eps},
            'batch_size': self.batch_size, 'epochs': self.epochs, 'scheduler': self.scheduler,
            'train_shuffle': self.train_shuffle, 'validation_shuffle': self.validation_shuffle,
            'early_stopping': self.early_stopping,
        }


_PROBE_PRESET = load_config_group('linear_probe', 'lp_v1')
_ENCODER_PRESET = load_config_group('encoder', 'vit_base_patch16_224')
_ACTIVITYNET_PRESET = load_config_group('activitynet', 'v1_3')
DEFAULT_CONFIG = ProbeConfig.from_mapping(_PROBE_PRESET['id'], _PROBE_PRESET['probe'])
# Backward-compatible library aliases. They are views of the Hydra presets,
# not independent defaults.
PROTOCOL = DEFAULT_CONFIG.protocol
FEATURE_SIZE = _ENCODER_PRESET['feature_size']
CLASS_COUNT = _ACTIVITYNET_PRESET['class_count']
SEEDS = DEFAULT_CONFIG.seeds
HYPERPARAMETERS = DEFAULT_CONFIG.hyperparameters(FEATURE_SIZE, CLASS_COUNT)


def experiment_name(protocol, condition=None, seed=None):
    if seed is None:
        # Legacy two-argument form: experiment_name(condition, seed).
        protocol, condition, seed = PROTOCOL, protocol, condition
    return f'{protocol}__{condition.replace("_", "-")}__seed-{seed}'


def train_probe(features, labels, seed, config=None, class_count=None, *, on_epoch=None):
    """Train a fresh classifier; features are never modified.

    Every hyperparameter comes from `config`, the same object callers record.
    """
    config = DEFAULT_CONFIG if config is None else config
    if not isinstance(config, ProbeConfig):
        raise TypeError('config must be a ProbeConfig')
    class_count = CLASS_COUNT if class_count is None else class_count
    if features.ndim != 2 or not features.shape[1] or features.requires_grad or features.dtype != torch.float32:
        raise ValueError('Expected non-empty-width frozen float32 features [N, D]')
    if labels.ndim != 1 or labels.dtype != torch.int64 or len(labels) != len(features) or not len(labels):
        raise ValueError('Expected non-empty int64 labels [N] aligned with features')
    if type(class_count) is not int or class_count < 1 or labels.min().item() < 0 or labels.max().item() >= class_count:
        raise ValueError('Labels must be in [0, class_count)')
    if seed not in config.seeds:
        raise ValueError(f'seed {seed} is not in the configured seeds')
    torch.manual_seed(seed)
    # Initialized on CPU from the seed, then moved, so the device never changes the init.
    classifier = nn.Linear(features.shape[1], class_count, bias=config.classifier_bias).to(features.device)
    shuffle = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=config.betas, eps=config.eps,
    )
    history = []
    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(len(features), generator=shuffle)
        loss_sum, correct = 0., 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            logits = classifier(features[batch])
            loss = F.cross_entropy(logits, labels[batch])
            if not torch.isfinite(loss).item():
                raise RuntimeError('Probe loss is not finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
            correct += (logits.argmax(dim=1) == labels[batch]).sum().item()
        row = {'epoch': epoch, 'train_loss': loss_sum / len(order), 'train_top1': 100. * correct / len(order)}
        history.append(row)
        if on_epoch:
            on_epoch(row)
    return classifier, history


@torch.no_grad()
def evaluate(classifier, features, labels):
    if features.ndim != 2 or features.shape[1] != classifier.in_features or not len(features):
        raise ValueError('Evaluation features do not match the classifier')
    if labels.ndim != 1 or labels.dtype != torch.int64 or len(labels) != len(features):
        raise ValueError('Evaluation labels must be int64 [N] aligned with features')
    predictions = classifier(features).argmax(dim=1).cpu()
    labels = labels.cpu()
    class_count = classifier.out_features
    if labels.min().item() < 0 or labels.max().item() >= class_count:
        raise ValueError('Evaluation labels are outside the classifier classes')
    confusion = torch.zeros(class_count, class_count, dtype=torch.int64)
    confusion.index_put_((labels, predictions), torch.ones_like(labels), accumulate=True)
    support = confusion.sum(dim=1)
    correct = confusion.diagonal()
    per_class = [100. * c / s if s else None for c, s in zip(correct.tolist(), support.tolist())]
    supported = [value for value in per_class if value is not None]
    return {
        'top1': 100. * correct.sum().item() / len(labels),
        'macro_class_accuracy': sum(supported) / len(supported),
        'macro_class_count': len(supported),
        'per_class_accuracy': per_class,
        'support': support.tolist(),
        'confusion_matrix': confusion.numpy(),
    }


def mean_std(values):
    """Arithmetic mean and sample standard deviation (ddof=1)."""
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else None
    return mean, std


def write_result(directory, classifier, history, metrics, summary, label_names):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save({'schema': RESULT_SCHEMA, 'state_dict': classifier.state_dict(),
                'condition': summary['condition'], 'seed': summary['seed']}, buffer)
    write_bytes_atomic(directory / 'probe_classifier.pt', buffer.getvalue())
    write_bytes_atomic(directory / 'history.csv', _csv(['epoch', 'train_loss', 'train_top1'], history))
    write_bytes_atomic(directory / 'per_class_accuracy.csv', _csv(
        ['label_id', 'label', 'accuracy', 'support'],
        [{'label_id': index, 'label': label_names.get(index, ''), 'accuracy': '' if accuracy is None else accuracy,
          'support': support}
         for index, (accuracy, support) in enumerate(zip(metrics['per_class_accuracy'], metrics['support']))]))
    matrix = io.BytesIO()
    numpy.save(matrix, metrics['confusion_matrix'])
    write_bytes_atomic(directory / 'confusion_matrix.npy', matrix.getvalue())
    summary = {**summary, 'files': {name: sha256_file(directory / name) for name in RESULT_FILES
                                    if name != 'summary.json'}}
    write_json_atomic(directory / 'summary.json', summary)
    return summary


def _csv(fields, rows):
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode()


def load_summary(directory):
    summary = read_json(Path(directory) / 'summary.json')
    if summary.get('schema') != RESULT_SCHEMA:
        raise RuntimeError(f'Unsupported probe result schema: {summary.get("schema")}')
    if {name: sha256_file(Path(directory) / name) for name in summary['files']} != summary['files']:
        raise RuntimeError(f'Probe result files differ from summary hashes: {directory}')
    return summary


def aggregate(summaries):
    """Per-condition mean +/- sample std over seeds, keeping every seed row."""
    conditions = {}
    for summary in summaries:
        conditions.setdefault(summary['condition'], []).append(summary)
    result = {}
    for condition, rows in conditions.items():
        rows = sorted(rows, key=lambda row: row['seed'])
        top1 = mean_std([row['top1'] for row in rows])
        macro = mean_std([row['macro_class_accuracy'] for row in rows])
        result[condition] = {
            'seeds': [row['seed'] for row in rows],
            'top1_mean': top1[0], 'top1_std': top1[1],
            'macro_class_accuracy_mean': macro[0], 'macro_class_accuracy_std': macro[1],
            'per_seed': [{key: row[key] for key in ('seed', 'top1', 'macro_class_accuracy', 'comet_experiment_key')}
                         for row in rows],
        }
    return result


def aggregate_csv(result):
    rows = []
    for condition, values in result.items():
        for row in values['per_seed']:
            rows.append({'condition': condition, 'seed': row['seed'], 'top1': row['top1'],
                         'macro_class_accuracy': row['macro_class_accuracy']})
        rows.append({'condition': condition, 'seed': 'mean', 'top1': values['top1_mean'],
                     'macro_class_accuracy': values['macro_class_accuracy_mean']})
        rows.append({'condition': condition, 'seed': 'sample_std_ddof1',
                     'top1': '' if values['top1_std'] is None else values['top1_std'],
                     'macro_class_accuracy': '' if values['macro_class_accuracy_std'] is None
                     else values['macro_class_accuracy_std']})
    return _csv(['condition', 'seed', 'top1', 'macro_class_accuracy'], rows)
