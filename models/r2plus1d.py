"""R(2+1)D model wrapper for fighting detection.

R(2+1)D decomposes 3D convolutions into separate spatial (2D) and
temporal (1D) convolutions. This decomposition doubles the number of
nonlinearities in the network and makes optimization easier.

Reference:
    Tran, D. et al. (2018). A Closer Look at Spatiotemporal Convolutions
    for Action Recognition. CVPR 2018.

R(2+1)D-18 specifications:
    - Input: 16 frames at 112x112 spatial resolution (default)
    - Parameters: ~31.5M
    - Pretrained on Kinetics-400
"""

import logging

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

logger = logging.getLogger(__name__)


class R2Plus1DModel(nn.Module):
    """R(2+1)D-18 model wrapper for binary fighting detection.

    Uses torchvision's pretrained R(2+1)D-18 model and replaces the
    classification head for binary classification.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initialize the R(2+1)D model.

        Args:
            num_classes: Number of output classes.
            pretrained: Whether to load pretrained Kinetics-400 weights.
            dropout_rate: Dropout rate before the final FC layer.
        """
        super().__init__()
        self.num_classes = num_classes

        logger.info("Loading R(2+1)D-18 model (pretrained=%s)", pretrained)

        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.model = r2plus1d_18(weights=weights)

        # Replace the classification head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info("Model loaded: total params=%d", total_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, T, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all layers except the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        logger.info("Backbone frozen, only head is trainable")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen for full fine-tuning")
