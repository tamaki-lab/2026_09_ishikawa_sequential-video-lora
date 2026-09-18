import os

from model import (
    ClassificationBaseModel,
    ModelConfig,
    ResNet50,
    ResNet18,
    ABNResNet50,
    X3DM,
    ViTb,
    ZeroOutputModel,
)


def set_torch_home(
    model_info: ModelConfig
) -> None:
    """Specity the directory where a pre-trained model is stored.
    Otherwise, by default, models are stored in users home dir `~/.torch`
    """
    os.environ['TORCH_HOME'] = model_info.torch_home


def configure_model(
        model_info: ModelConfig
) -> ClassificationBaseModel:
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        ClassificationBaseModel: model
    """

    if model_info.use_pretrained:
        set_torch_home(model_info)

    if model_info.model_name == 'resnet18':
        model = ResNet18(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'resnet50':
        model = ResNet50(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'abn_r50':
        model = ABNResNet50(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'vit_b':
        model = ViTb(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'x3d':
        model = X3DM(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'zero_output_dummy':
        model = ZeroOutputModel(model_info)  # type: ignore[assignment]

    else:
        raise ValueError('invalid model_info.model_name')

    return model
