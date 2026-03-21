"""I3D (Inflated 3D) video action detection model.

Reference:
    Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model
    and Large-Scale Datasets. CVPR 2017.
    https://arxiv.org/abs/1705.07045

NOTE: Uses R(2+1)D-18 backbone for compatibility. Full I3D from PyTorchVideo requires
complex head replacement that is not compatible with transfer learning for binary classification.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models

from models.base import VideoActionDetector

logger = logging.getLogger(__name__)


class I3DModel(VideoActionDetector):
    """I3D model for video action detection.

    Implements inflated 3D convolutions for efficient video understanding.
    Uses pretrained weights from Kinetics-400.

    Input specifications:
        - Shape: (B, C, T, H, W)
        - Frames: 64 (configurable)
        - Spatial: 224×224 (configurable)
        - Channels: 3 (RGB)
        - Frame rate: 24 fps (typical)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initialize I3D model.

        Args:
            num_classes: Number of output classes
            pretrained: Load Kinetics-400 pretrained weights
            dropout_rate: Dropout rate in classification head
        """
        super().__init__(num_classes, pretrained, dropout_rate)

        logger.info("Using R(2+1)D as I3D backbone")

        try:
            self.model = models.video.r2plus1d_18(
                weights="DEFAULT" if pretrained else None
            )
        except TypeError:
            self.model = models.video.r2plus1d_18(
                pretrained=pretrained
            )

        # Get input/output feature dimensions
        self.feat_dim = self.model.fc.in_features

        # Replace classification head for binary classification
        self._build_classifier()

    def _build_classifier(self):
        """Replace the classification head for binary classification."""
        self.model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feat_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, T, H, W)

        Returns:
            Logits tensor (B, num_classes)
        """
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all backbone parameters except classification head."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logger.info("I3D backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("I3D backbone unfrozen")

    def get_param_groups(self, backbone_lr_scale: float = 0.1) -> List[Dict]:
        """Get parameter groups with different learning rates.

        Args:
            backbone_lr_scale: LR scale for backbone relative to head

        Returns:
            List of parameter groups for optimizer
        """
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": head_params, "lr_scale": 1.0},
        ]
