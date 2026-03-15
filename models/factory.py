"""Model factory for creating fighting detection models."""

import logging

import torch.nn as nn

from utils.config import Config

logger = logging.getLogger(__name__)


def create_model(config: Config) -> nn.Module:
    """Create a model based on the configuration.

    Args:
        config: Configuration object with model settings.

    Returns:
        Initialized model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    model_name = config.model.name
    num_classes = config.model.num_classes
    pretrained = config.model.pretrained
    dropout_rate = config.model.get("dropout_rate", 0.5)

    logger.info("Creating model: %s", model_name)

    if model_name in ("x3d_s", "x3d_xs"):
        from models.x3d import X3DModel
        return X3DModel(
            model_variant=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    elif model_name == "r2plus1d_18":
        from models.r2plus1d import R2Plus1DModel
        return R2Plus1DModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: x3d_s, x3d_xs, r2plus1d_18"
        )
