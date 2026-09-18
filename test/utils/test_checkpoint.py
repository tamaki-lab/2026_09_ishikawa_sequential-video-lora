from dataclasses import dataclass
import pytest


import torch
from torch import nn


from model import (
    configure_model,
    ModelConfig,
)
from utils import (
    save_to_checkpoint,
    load_from_checkpoint,
)
from setup import configure_optimizer, configure_scheduler


@dataclass
class DummyLogger:
    project_name: str = "test_checkpoint_project"
    name: str = "experiment_name"


@pytest.mark.parametrize(
    'model_name',
    ["resnet18", "resnet50", "abn_r50", "vit_b", "x3d", "zero_output_dummy"]
)
@pytest.mark.parametrize('use_dp_when_save', [True, False])
@pytest.mark.parametrize('use_dp_when_load', [True, False])
def test_checkpoint_save_load(  # noqa: FNE003 FNE004
    model_name,
    use_dp_when_save,
    use_dp_when_load,
    tmp_path,  # https://docs.pytest.org/en/stable/how-to/tmp_path.html
    n_classes=2,
    use_pretrained=False,
    current_epoch=10,
    current_train_step=100,
    current_val_step=5,
    val_top1=99.99,
    optimizer_name="SGD",
    lr=1e-5,
    weight_decay=1e-5,
    momentum=0.99,
    use_scheduler=True,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    # prepare model, optimzer, scheduler to save
    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
    ))
    model.to(device)
    if use_dp_when_save:
        model = nn.DataParallel(model)  # type: ignore[assignment]

    optimizer = configure_optimizer(
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        model_params=model.parameters()
    )
    scheduler = configure_scheduler(
        optimizer=optimizer,
        use_scheduler=use_scheduler
    )

    # save
    _, checkpoint_filename = save_to_checkpoint(
        save_checkpoint_dir=str(tmp_path),
        current_epoch=current_epoch,
        current_train_step=current_train_step,
        current_val_step=current_val_step,
        acc=val_top1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        experiment_logger=DummyLogger(),
    )

    # prepare another model, optimzer, scheduler to load
    another_model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
    ))
    another_model.to(device)
    if use_dp_when_load:
        another_model = nn.DataParallel(another_model)  # type: ignore[assignment]

    another_optimizer = configure_optimizer(
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        model_params=another_model.parameters()
    )
    another_scheduler = configure_scheduler(
        optimizer=another_optimizer,
        use_scheduler=use_scheduler
    )

    # load
    (
        loaded_current_epoch,
        loaded_current_train_step,
        loaded_current_val_step,
        loaded_model,
        loaded_optimizer,
        loaded_scheduler,
    ) = load_from_checkpoint(
        checkpoint_to_resume=checkpoint_filename,
        model=another_model,
        optimizer=another_optimizer,
        scheduler=another_scheduler,
        device=device
    )

    # check
    assert loaded_current_epoch == current_epoch
    assert loaded_current_train_step == current_train_step
    assert loaded_current_val_step == current_val_step

    assert id(loaded_optimizer) != id(optimizer)
    assert loaded_optimizer.state_dict() == optimizer.state_dict()

    assert id(loaded_scheduler) != id(scheduler)
    assert loaded_scheduler.state_dict() == scheduler.state_dict()

    assert id(loaded_model) != id(model)

    # below is "assert loaded_model == model"
    for p1, p2 in zip(loaded_model.named_parameters(), model.named_parameters()):

        # instead of assert p1[0] == p2[0], below compares "conv1" == "module.conv1"
        assert p1[0] in p2[0] or p2[0] in p1[0], "layer names are different"

        assert torch.eq(p1[1], p2[1]).all().item(), "layer parameters are different"
