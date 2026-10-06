"""Linear Probe: fixed classifier, seed scope, metrics, aggregation and artifacts."""

import csv
from dataclasses import replace
import io
import json

import numpy
import pytest
import torch

from evaluation import linear_probe as probe


def epochs(count):
    return probe.DEFAULT_CONFIG.with_epochs(count)


def data(count=300, seed=0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, probe.CLASS_COUNT, (count,), generator=generator)
    features = torch.randn(count, 768, generator=generator)
    features[torch.arange(count), labels] += 4.
    return features, labels


def test_only_classifier_is_trained_and_features_are_untouched():
    features, labels = data()
    before = features.clone()
    classifier, history = probe.train_probe(features, labels, seed=0, config=epochs(3))
    assert isinstance(classifier, torch.nn.Linear)
    assert (classifier.in_features, classifier.out_features, classifier.bias is not None) == (768, 200, True)
    assert torch.equal(features, before) and not features.requires_grad and features.grad is None
    assert [row['epoch'] for row in history] == [1, 2, 3]
    assert history[-1]['train_loss'] < history[0]['train_loss']
    with pytest.raises(ValueError, match='frozen'):
        probe.train_probe(features.requires_grad_(), labels, seed=0, config=epochs(1))


def test_seed_controls_only_init_and_shuffle():
    features, labels = data()
    init = {seed: probe.train_probe(features, labels, seed=seed, config=epochs(0))[0].weight.detach()
            for seed in (0, 1)}
    torch.manual_seed(1234)
    torch.rand(10)  # Unrelated global RNG use must not change a seeded run.
    again = probe.train_probe(features, labels, seed=0, config=epochs(2))
    reference = probe.train_probe(features, labels, seed=0, config=epochs(2))
    assert torch.equal(again[0].weight, reference[0].weight) and again[1] == reference[1]
    assert not torch.equal(init[0], init[1])
    orders = []
    original = torch.randperm

    def record(*args, **kwargs):
        orders.append(original(*args, **kwargs))
        return orders[-1]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, 'randperm', record)
        probe.train_probe(features, labels, seed=0, config=epochs(1))
        probe.train_probe(features, labels, seed=1, config=epochs(1))
    assert len(orders) == 2 and not torch.equal(orders[0], orders[1])


def test_metrics_confusion_and_macro_accuracy():
    classifier = torch.nn.Linear(768, 200)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.bias.zero_()
        classifier.weight[:200, :200] = torch.eye(200)
    # Class 0: 2/2 correct, class 1: 1/2 (one predicted as 3), class 2: 0/1.
    labels = torch.tensor([0, 0, 1, 1, 2])
    predicted = torch.tensor([0, 0, 1, 3, 5])
    features = torch.zeros(5, 768)
    features[torch.arange(5), predicted] = 1.
    metrics = probe.evaluate(classifier, features, labels)
    assert metrics['top1'] == pytest.approx(60.)
    assert metrics['macro_class_accuracy'] == pytest.approx((100. + 50. + 0.) / 3)
    assert metrics['macro_class_count'] == 3
    assert metrics['support'][:4] == [2, 2, 1, 0] and metrics['per_class_accuracy'][3] is None
    confusion = metrics['confusion_matrix']
    assert confusion.shape == (200, 200) and confusion.sum() == 5
    assert confusion[0, 0] == 2 and confusion[1, 3] == 1 and confusion[2, 5] == 1


def test_mean_and_sample_std_and_aggregate():
    assert probe.mean_std([1., 2., 3.]) == (2., 1.)
    summaries = [{'condition': condition, 'seed': seed, 'top1': top1, 'macro_class_accuracy': top1 - 1,
                  'comet_experiment_key': None}
                 for condition, values in (('base_vit', (10., 12., 14.)), ('moco_query_lora_final', (11., 11., 11.)))
                 for seed, top1 in zip(probe.SEEDS, values)]
    result = probe.aggregate(summaries[::-1])
    assert result['base_vit']['seeds'] == [0, 1, 2]
    assert result['base_vit']['top1_mean'] == 12. and result['base_vit']['top1_std'] == 2.
    assert result['moco_query_lora_final']['top1_std'] == 0.
    rows = list(csv.DictReader(io.StringIO(probe.aggregate_csv(result).decode())))
    assert [row['seed'] for row in rows[:5]] == ['0', '1', '2', 'mean', 'sample_std_ddof1']
    assert probe.experiment_name('lp-v1', 'base_vit', 0) == 'lp-v1__probe__base-vit__seed-0'
    assert probe.experiment_name('lp-v1', 'moco_query_lora_final', 2) == 'lp-v1__probe__moco-query-lora-final__seed-2'
    assert probe.experiment_name('moco_query_lora_final', 2) == 'lp-v1__probe__moco-query-lora-final__seed-2'


def test_result_files_round_trip_and_tamper_detection(tmp_path):
    features, labels = data(64)
    classifier, history = probe.train_probe(features, labels, seed=0, config=epochs(2))
    metrics = probe.evaluate(classifier, features, labels)
    summary = probe.write_result(tmp_path, classifier, history, metrics, {
        'schema': probe.RESULT_SCHEMA, 'condition': 'base_vit', 'seed': 0, 'top1': metrics['top1'],
    }, {index: f'label-{index}' for index in range(200)})
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(probe.RESULT_FILES)
    assert probe.load_summary(tmp_path) == summary
    assert numpy.load(tmp_path / 'confusion_matrix.npy').shape == (200, 200)
    assert json.loads((tmp_path / 'summary.json').read_text())['files'].keys() == set(probe.RESULT_FILES) - {
        'summary.json'}
    (tmp_path / 'history.csv').write_text('changed')
    with pytest.raises(RuntimeError, match='hashes'):
        probe.load_summary(tmp_path)


def test_optimizer_is_built_from_the_recorded_config(monkeypatch):
    features, labels = data(32)
    config = replace(epochs(1), lr=5e-4, weight_decay=0.0, betas=(0.8, 0.99), eps=1e-6)
    created = []
    factory = torch.optim.AdamW

    def record(parameters, **kwargs):
        created.append(kwargs)
        return factory(parameters, **kwargs)
    monkeypatch.setattr(torch.optim, 'AdamW', record)
    probe.train_probe(features, labels, seed=0, config=config)
    recorded = config.hyperparameters(768, probe.CLASS_COUNT)['optimizer']
    assert created == [{key: recorded[key] if key != 'betas' else tuple(recorded[key])
                        for key in ('lr', 'weight_decay', 'betas', 'eps')}]
    with pytest.raises(ValueError, match='betas'):
        replace(config, betas=(0.9,))
