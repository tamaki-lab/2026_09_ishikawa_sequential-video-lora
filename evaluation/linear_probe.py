"""Frozen-feature Linear Probe: fixed hyperparameters, final-epoch evaluation.

The seed only sets the classifier initialization and the training shuffle
order. Validation is evaluated once after the last epoch and never selects
an epoch. Metrics: Top-1 (primary) and Macro class accuracy (secondary).
"""

import csv
import io
import math
from pathlib import Path

import numpy
import torch
from torch import nn
from torch.nn import functional as F

from utils.artifact_io import read_json, sha256_file, write_bytes_atomic, write_json_atomic


RESULT_SCHEMA = 'activitynet-linear-probe-result/v1'
AGGREGATE_SCHEMA = 'activitynet-linear-probe-aggregate/v1'
PROTOCOL = 'lp-v1'
FEATURE_SIZE = 768
CLASS_COUNT = 200
SEEDS = (0, 1, 2)
HYPERPARAMETERS = {
    'classifier': 'Linear(768, 200, bias=True)', 'loss': 'cross_entropy', 'class_weight': None,
    'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 256, 'epochs': 100,
    'scheduler': None, 'train_shuffle': True, 'validation_shuffle': False, 'early_stopping': False,
}
RESULT_FILES = ('probe_classifier.pt', 'summary.json', 'history.csv', 'per_class_accuracy.csv',
                'confusion_matrix.npy')


def experiment_name(condition, seed):
    return f'{PROTOCOL}__{condition.replace("_", "-")}__seed-{seed}'


def train_probe(features, labels, seed, epochs, *, batch_size=256, lr=1e-3, weight_decay=1e-4, on_epoch=None):
    """Train a fresh classifier; features are never modified."""
    if features.requires_grad or features.dtype != torch.float32 or tuple(features.shape[1:]) != (FEATURE_SIZE,):
        raise ValueError('Expected frozen float32 features [N, 768]')
    torch.manual_seed(seed)
    # Initialized on CPU from the seed, then moved, so the device never changes the init.
    classifier = nn.Linear(FEATURE_SIZE, CLASS_COUNT, bias=True).to(features.device)
    shuffle = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(features), generator=shuffle)
        loss_sum, correct = 0., 0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
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
    predictions = classifier(features).argmax(dim=1)
    confusion = torch.zeros(CLASS_COUNT, CLASS_COUNT, dtype=torch.int64)
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
