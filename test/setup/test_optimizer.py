import pytest
from torch.optim import Optimizer

from setup import configure_optimizer


@pytest.mark.parametrize('optimizer_name', ["SGD", "Adam"])
@pytest.mark.parametrize('lr', [0.01, 1e-5])
@pytest.mark.parametrize('weight_decay', [0.01, 1e-5])
@pytest.mark.parametrize('momentum', [1.0, 0.99])
def test_optimizer(
    optimizer_name,
    lr,
    weight_decay,
    momentum,
    model,
):

    optimizer = configure_optimizer(
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        model_params=model.parameters()
    )
    assert isinstance(optimizer, Optimizer)
