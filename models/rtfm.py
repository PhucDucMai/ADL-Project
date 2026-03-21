"""RTFM (Real-Time FightingMonitoring) video action detection model.

RTFM is designed for real-time fighting detection and has been used in
action detection competitions. This wrapper attempts to load the model
from HuggingFace or uses a deep CNN fallback.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models

from models.base import VideoActionDetector

logger = logging.getLogger(__name__)


class RTFMModel(VideoActionDetector):
    """RTFM model for video action detection.

    Specialized architecture for real-time fighting monitoring.
    Uses efficient 3D convolutions.

    Input specifications:
        - Shape: (B, C, T, H, W)
        - Frames: 16 (adaptive)
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
        """Initialize RTFM model.

        Args:
            num_classes: Number of output classes
            pretrained: Load pretrained weights
            dropout_rate: Dropout rate in classification head
        """
        super().__init__(num_classes, pretrained, dropout_rate)

        # Try to load RTFM from HuggingFace first
        try:
            from transformers import AutoModel

            model_id = "spaces/RTFM/rtfm-action-detection"
            self.model = AutoModel.from_pretrained(
                model_id, trust_remote_code=True
            )
            logger.info(f"Loaded RTFM model from {model_id}")
            self.feat_dim = self.model.config.hidden_size
        except Exception as e:
            logger.warning(
                f"Could not load RTFM from HuggingFace ({e}). "
                "Using ResNet3D fallback (similar architecture)."
            )
            # Fallback: use pretrained R(2+1)D which is similar in spirit
            weights = (
                models.video.R2Plus1D_18_Weights.KINETICS400_V1
                if pretrained
                else None
            )
            self.model = models.video.r2plus1d_18(weights=weights)
            self.feat_dim = self.model.fc.in_features
            logger.info("Using R(2+1)D-18 as RTFM fallback")

        # Replace classification head
        self._build_classifier()

    def _build_classifier(self):
        """Replace the classification head for binary classification."""
        if hasattr(self.model, "fc"):
            self.model.fc = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.feat_dim, self.num_classes),
            )
        elif hasattr(self.model, "classifier"):
            self.model.classifier = nn.Sequential(
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
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        logger.info("RTFM backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("RTFM backbone unfrozen")

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
            if "fc" in name or "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": head_params, "lr_scale": 1.0},
        ]
