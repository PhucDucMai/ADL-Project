"""Base class for video action detection models.

Provides common interface for all video model architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class VideoActionDetector(nn.Module, ABC):
    """Abstract base class for video action detection models.

    Defines the interface that all video detection models must implement.
    Supports:
    - Flexible input shapes (B, C, T, H, W)
    - Transfer learning with backbone freezing
    - Per-layer learning rate scaling
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initialize video action detector.

        Args:
            num_classes: Number of output classes (default: 2 for fight/normal)
            pretrained: Whether to load pretrained weights
            dropout_rate: Dropout rate in classification head
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for video classification.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
               B = batch size
               C = channels (typically 3 for RGB)
               T = temporal dimension (frames)
               H, W = spatial dimensions

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        raise NotImplementedError

    def freeze_backbone(self):
        """Freeze all backbone (feature extraction) parameters.

        Only classification head remains trainable.
        Used for transfer learning with frozen pretrained features.
        """
        # Default implementation: freeze all but last layer
        for name, param in self.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters.

        Used for full fine-tuning after initial training phase.
        """
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(
        self, backbone_lr_scale: float = 0.1
    ) -> List[Dict]:
        """Get parameter groups with different learning rates.

        Useful for transfer learning where pretrained backbone gets
        a lower learning rate than new classification head.

        Args:
            backbone_lr_scale: Multiplier for backbone LR relative to head LR
                             (e.g., 0.1 means backbone uses 0.1x the head LR)

        Returns:
            List of parameter groups suitable for torch.optim.Optimizer

        Example:
            >>> param_groups = model.get_param_groups(backbone_lr_scale=0.1)
            >>> optimizer = torch.optim.SGD(param_groups, lr=0.01)
            # Backbone learns at 0.001, head learns at 0.01
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if "classifier" in name or "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": head_params, "lr_scale": 1.0},
        ]

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters.

        Returns:
            Dictionary with 'total' and 'trainable' parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
