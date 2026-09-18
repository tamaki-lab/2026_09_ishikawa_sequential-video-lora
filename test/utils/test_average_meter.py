import pytest

from torch import Tensor

from utils import (
    AverageMeter,
    AvgMeterLossTopk,
)


@pytest.mark.parametrize(
    'values,avg', [
        ((1.0, 2.0, 3.0), 2.0),
        ((3.0, 2.0, 4.0), 3.0),
        (Tensor([1.0, 2.0, 3.0]), 2.0),
        (Tensor([3.0, 2.0, 4.0]), 3.0),
    ])
def test_average_meter(
    values,
    avg,
):
    meter = AverageMeter()
    for value in values:
        meter.update(value)

    assert meter.avg == avg
    assert isinstance(meter.avg, float)
    assert meter.count == len(values)


@pytest.mark.parametrize(
    'loss_list,topk_list,loss_avg,topk_avg_list', [
        (
            (9.0, 10.0, 11.0),
            (
                (1.0, 3.0),
                (2.0, 2.0),
                (3.0, 4.0),
            ),
            10.0,
            (2.0, 3.0),
        )
    ])
def test_average_meters_loss_topk(
    loss_list,
    topk_list,
    loss_avg,
    topk_avg_list,
):
    meters = AvgMeterLossTopk('train')
    for loss, topk in zip(loss_list, topk_list):
        top1, top5 = topk
        meters.update(loss, (top1, top5))

    assert meters.loss_meter.avg == loss_avg

    for meter, avg in zip(meters.topk_meters, topk_avg_list):
        assert meter.avg == avg
