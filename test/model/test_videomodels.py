import pytest

import torch

from model import (
    configure_model,
    ModelConfig,
    ModelOutput,
    ClassificationBaseModel,
)


@pytest.mark.parametrize(
    ['model_name', 'frames_per_clip'],
    [
        ('x3d', 16),
    ]
)
@pytest.mark.parametrize('use_pretrained', [True, False])
@pytest.mark.parametrize('n_classes', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('crop_size', [224])
def test_video_model_output(
    model_name,
    use_pretrained,
    n_classes,
    batch_size,
    crop_size,
    frames_per_clip,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
    ))
    model.to(device)
    assert isinstance(model, ClassificationBaseModel)

    data = torch.rand(batch_size, 3, frames_per_clip, crop_size, crop_size, device=device)  # BCTHW
    labels = torch.randint(0, n_classes - 1, (batch_size, ), device=device)

    output = model(data, labels)
    assert isinstance(output, ModelOutput)

    assert isinstance(output.logits, torch.Tensor)
    assert output.logits.shape == (batch_size, n_classes)

    assert isinstance(output.loss, torch.Tensor)
    assert output.loss.ndim == 0

    # no labels
    output = model(data)
    assert isinstance(output, ModelOutput)

    assert isinstance(output.logits, torch.Tensor)
    assert output.logits.shape == (batch_size, n_classes)

    assert output.loss is None


@pytest.mark.parametrize(
    ['model_name', 'frames_per_clip'],
    [
        ('x3d', 16),
    ]
)
@pytest.mark.parametrize('use_pretrained', [True, False])
@pytest.mark.parametrize('n_classes', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('crop_size', [224])
def test_video_model_loss_backward(
    model_name,
    use_pretrained,
    n_classes,
    batch_size,
    crop_size,
    frames_per_clip,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
    ))
    model.to(device)
    assert isinstance(model, ClassificationBaseModel)

    data = torch.rand(batch_size, 3, frames_per_clip, crop_size, crop_size, device=device)  # BCTHW
    labels = torch.randint(0, n_classes - 1, (batch_size, ), device=device)

    output = model(data, labels)
    output.loss.backward()
