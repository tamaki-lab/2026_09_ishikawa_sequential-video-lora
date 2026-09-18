import pytest
from torch.optim.lr_scheduler import LRScheduler

from setup import configure_optimizer, configure_scheduler


@pytest.fixture(scope='session')
def optimizer(
    model,
    optimizer_name="SGD",
    lr=1e-5,
    weight_decay=1e-5,
    momentum=0.99,
):
    optimizer_for_test = configure_optimizer(
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        model_params=model.parameters()
    )
    return optimizer_for_test


@pytest.mark.parametrize('use_scheduler', [True, False])
def test_scheduler(
    optimizer,  # pylint: disable=redefined-outer-name
    use_scheduler: bool,
):

    scheduler = configure_scheduler(
        optimizer=optimizer,
        use_scheduler=use_scheduler
    )
    assert isinstance(scheduler, LRScheduler)
