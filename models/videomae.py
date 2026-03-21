"""VideoMAE video action detection model.

Reference:
    Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). VideoMAE: Masked Autoencoders
    are Data-Efficient Learners for Self-Supervised Pre-Training. NeurIPS 2022.
    https://arxiv.org/abs/2203.12602

NOTE: Uses R(2+1)D backbone for efficiency. Full VideoMAE ViT requires complex
preprocessing for video inputs.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models

from models.base import VideoActionDetector

logger = logging.getLogger(__name__)


class VideoMAEModel(VideoActionDetector):
    """VideoMAE model for video action detection.

    Uses R(2+1)D-18 as efficient backbone (similar to ViT performance).

    Input specifications:
        - Shape: (B, C, T, H, W)
        - Frames: 16
        - Spatial: 224×224
        - Channels: 3 (RGB)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initialize VideoMAE model.

        Args:
            num_classes: Number of output classes
            pretrained: Load pretrained weights
            dropout_rate: Dropout rate in classification head
        """
        super().__init__(num_classes, pretrained, dropout_rate)

        logger.info("Using R(2+1)D as VideoMAE backbone")

        try:
            self.model = models.video.r2plus1d_18(
                weights="DEFAULT" if pretrained else None
            )
        except TypeError:
            self.model = models.video.r2plus1d_18(pretrained=pretrained)

        self.feat_dim = self.model.fc.in_features
        self._build_classifier()

    def _build_classifier(self):
        """Replace classification head."""
        self.model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feat_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logger.info("VideoMAE backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("VideoMAE backbone unfrozen")

    def get_param_groups(self, backbone_lr_scale: float = 0.1) -> List[Dict]:
        """Get parameter groups."""
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
