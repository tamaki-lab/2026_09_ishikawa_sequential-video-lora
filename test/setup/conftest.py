import pytest
import torch

from model import configure_model, ModelConfig


@pytest.fixture(scope='session')
def model(
    model_name="resnet18",
    n_classes=10,
    use_pretrained=True,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model_for_test = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
    ))
    model_for_test.to(device)
    return model_for_test
