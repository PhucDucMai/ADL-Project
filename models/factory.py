"""Model factory for creating fighting detection models.

Supports multiple models from various sources:
- PyTorch Hub (X3D, I3D)
- Torchvision (R(2+1)D, SlowFast)
- HuggingFace (VideoMAE, VAD-CLIP, RTFM)
"""

import logging

import torch.nn as nn

from utils.config import Config

logger = logging.getLogger(__name__)

# Supported models with their default source
SUPPORTED_MODELS = {
    # Original models
    "x3d_s": {"source": "torch_hub", "module": "models.x3d"},
    "x3d_xs": {"source": "torch_hub", "module": "models.x3d"},
    "r2plus1d_18": {"source": "torchvision", "module": "models.r2plus1d"},
    # New models
    "i3d": {"source": "torch_hub", "module": "models.i3d"},
    "slowfast": {"source": "torchvision", "module": "models.slowfast"},
    "videomae": {"source": "huggingface", "module": "models.videomae"},
    "rtfm": {"source": "huggingface", "module": "models.rtfm"},
    "vad_clip": {"source": "huggingface", "module": "models.vad_clip"},
}


def create_model(config: Config) -> nn.Module:
    """Create a model based on the configuration.

    Supports multiple model sources:
    - torch_hub: PyTorch Hub models (X3D, I3D)
    - torchvision: Torchvision models (R(2+1)D, SlowFast)
    - huggingface: HuggingFace transformers (VideoMAE, VAD-CLIP, RTFM)

    Args:
        config: Configuration object with model settings.
               config.model.name: Model architecture ID
               config.model.source: (optional) Override default source
               config.model.num_classes: Number of output classes
               config.model.pretrained: Load pretrained weights
               config.model.dropout_rate: Dropout rate in head

    Returns:
        Initialized model (nn.Module).

    Raises:
        ValueError: If model name is not recognized or invalid source.
    """
    model_name = config.model.name.lower()
    num_classes = config.model.num_classes
    pretrained = config.model.pretrained
    dropout_rate = config.model.get("dropout_rate", 0.5)

    # Get model source (default or override from config)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS.keys())}"
        )

    source = config.model.get("source", SUPPORTED_MODELS[model_name]["source"])
    logger.info(
        "Creating model: %s (source: %s, pretrained: %s)",
        model_name,
        source,
        pretrained,
    )

    # Create model based on source
    if source == "torch_hub":
        return _create_torch_hub_model(
            model_name, num_classes, pretrained, dropout_rate
        )
    elif source == "torchvision":
        return _create_torchvision_model(
            model_name, num_classes, pretrained, dropout_rate
        )
    elif source == "huggingface":
        return _create_huggingface_model(
            model_name, num_classes, pretrained, dropout_rate
        )
    else:
        raise ValueError(
            f"Unknown model source: {source}. "
            f"Supported sources: torch_hub, torchvision, huggingface"
        )


def _create_torch_hub_model(
    model_name: str, num_classes: int, pretrained: bool, dropout_rate: float
) -> nn.Module:
    """Create model from PyTorch Hub.

    Args:
        model_name: Model name (x3d_s, x3d_xs, i3d)
        num_classes: Number of output classes
        pretrained: Load pretrained weights
        dropout_rate: Dropout rate

    Returns:
        Initialized model
    """
    if model_name in ("x3d_s", "x3d_xs"):
        from models.x3d import X3DModel

        return X3DModel(
            model_variant=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    elif model_name == "i3d":
        from models.i3d import I3DModel

        return I3DModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown torch_hub model: {model_name}")


def _create_torchvision_model(
    model_name: str, num_classes: int, pretrained: bool, dropout_rate: float
) -> nn.Module:
    """Create model from torchvision.models.video.

    Args:
        model_name: Model name (r2plus1d_18, slowfast)
        num_classes: Number of output classes
        pretrained: Load pretrained weights
        dropout_rate: Dropout rate

    Returns:
        Initialized model
    """
    if model_name == "r2plus1d_18":
        from models.r2plus1d import R2Plus1DModel

        return R2Plus1DModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    elif model_name == "slowfast":
        from models.slowfast import SlowFastModel

        return SlowFastModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown torchvision model: {model_name}")


def _create_huggingface_model(
    model_name: str, num_classes: int, pretrained: bool, dropout_rate: float
) -> nn.Module:
    """Create model from HuggingFace transformers.

    Args:
        model_name: Model name (videomae, rtfm, vad_clip)
        num_classes: Number of output classes
        pretrained: Load pretrained weights
        dropout_rate: Dropout rate

    Returns:
        Initialized model
    """
    if model_name == "videomae":
        from models.videomae import VideoMAEModel

        return VideoMAEModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    elif model_name == "rtfm":
        from models.rtfm import RTFMModel

        return RTFMModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    elif model_name == "vad_clip":
        from models.vad_clip import VADCLIPModel

        return VADCLIPModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unknown HuggingFace model: {model_name}")


def list_available_models():
    """Return list of available models."""
    return list(SUPPORTED_MODELS.keys())

