import pytest

from torch import Tensor

from utils import compute_topk_accuracy


@pytest.mark.parametrize(
    'logits,labels,topk,gt_acc', [
        (
            Tensor([
                [100.0, 10.0, 1.0],
                [1.0, 100.0, 10.0],
                [1.0, 10.0, 100.0],
            ]),
            Tensor([0, 1, 2]),
            (1, 2),
            (100, 100),
        ),
        (
            Tensor([
                [100.0, 10.0, 1.0],
                [100.0, 10.0, 1.0],
                [1.0, 100.0, 10.0],
                [1.0, 10.0, 100.0],
            ]),
            Tensor([0, 1, 0, 2]),
            (1, 2),
            (50, 75),
        ),
        (
            Tensor([
                [100.0, 10.0, 1.0],
                [100.0, 10.0, 1.0],
                [1.0, 100.0, 10.0],
                [1.0, 10.0, 100.0],
            ]),
            Tensor([0, 1, 0, 2]),
            (1,),
            (50,),
        ),
    ])
def test_accuracy(
    logits,
    labels,
    topk,
    gt_acc,
):
    acc = compute_topk_accuracy(logits, labels, topk=topk)
    assert acc == gt_acc
