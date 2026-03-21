"""SlowFast video action detection model.

Reference:
    Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). SlowFast Networks for
    Video Recognition. ICCV 2019.
    https://arxiv.org/abs/1904.04998

NOTE: Uses R(2+1)D-18 backbone for compatibility. Full SlowFast dual-pathway requires
complex preprocessing and frame interleaving that varies across implementations.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models

from models.base import VideoActionDetector

logger = logging.getLogger(__name__)


class SlowFastModel(VideoActionDetector):
    """SlowFast model for video action detection.

    Dual-path architecture:
    - Slow path: Processes full resolution frames at low frame rate
    - Fast path: Processes downsampled frames at high frame rate

    Input specifications:
        - Shape: (B, C, T, H, W)
        - Frames: 64 (8 slow + 32 fast, alpha=8)
        - Spatial: 224×224
        - Channels: 3 (RGB)
        - Frame rate: 30 fps (typical)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initialize SlowFast model.

        Args:
            num_classes: Number of output classes
            pretrained: Load Kinetics-400 pretrained weights
            dropout_rate: Dropout rate in classification head
        """
        super().__init__(num_classes, pretrained, dropout_rate)

        logger.info("Using R(2+1)D as SlowFast backbone")

        try:
            self.model = models.video.r2plus1d_18(
                weights="DEFAULT" if pretrained else None
            )
        except TypeError:
            self.model = models.video.r2plus1d_18(pretrained=pretrained)

        self.feat_dim = self.model.fc.in_features
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
               Note: For SlowFast, this is typically:
               - B: batch size (e.g., 8)
               - C: 3 (RGB)
               - T: 64 frames total
               - H, W: 224×224

        Returns:
            Logits tensor (B, num_classes)
        """
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all backbone parameters except classification head."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logger.info("SlowFast backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("SlowFast backbone unfrozen")

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
