"""X3D model wrapper for fighting detection.

X3D (Expanded 3D) is an efficient video recognition architecture that
progressively expands a tiny 2D image classification architecture along
multiple network axes (temporal, spatial, width, depth, etc.).

Reference:
    Feichtenhofer, C. (2020). X3D: Expanding Architectures for Efficient
    Video Recognition. CVPR 2020.

X3D-S specifications:
    - Input: 13 frames at 182x182 spatial resolution
    - Parameters: ~3.8M
    - FLOPs: ~2.96G
    - Top-1 accuracy on Kinetics-400: ~73.3%
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class X3DModel(nn.Module):
    """X3D-S/XS model wrapper for binary fighting detection.

    Loads a pretrained X3D model from pytorchvideo and replaces the
    classification head for binary classification (normal vs fight).
    """

    def __init__(
        self,
        model_variant: str = "x3d_s",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initialize the X3D model.

        Args:
            model_variant: Which X3D variant to use ("x3d_s" or "x3d_xs").
            num_classes: Number of output classes.
            pretrained: Whether to load pretrained weights from Kinetics-400.
            dropout_rate: Dropout rate before the final classification layer.
        """
        super().__init__()
        self.model_variant = model_variant
        self.num_classes = num_classes

        logger.info("Loading %s model (pretrained=%s)", model_variant, pretrained)

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            model_variant,
            pretrained=pretrained,
        )

        # Replace the classification head
        # X3D model structure: model.blocks[5] is the head block
        # The head contains: pool -> dropout -> proj -> activation
        head_block = self.model.blocks[5]
        in_features = head_block.proj.in_features

        head_block.dropout = nn.Dropout(p=dropout_rate)
        head_block.proj = nn.Linear(in_features, num_classes)
        # Remove the softmax activation since we use CrossEntropyLoss
        head_block.activation = None

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Model loaded: total params=%d, trainable params=%d",
            total_params,
            trainable_params,
        )

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
        # Unfreeze the head
        for param in self.model.blocks[5].parameters():
            param.requires_grad = True
        logger.info("Backbone frozen, only head is trainable")

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen for full fine-tuning")

    def get_param_groups(self, backbone_lr_scale: float = 0.1):
        """Get parameter groups with different learning rates.

        Args:
            backbone_lr_scale: Scale factor for backbone learning rate
                relative to the head learning rate.

        Returns:
            List of parameter group dictionaries.
        """
        head_params = list(self.model.blocks[5].parameters())
        head_param_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in self.model.parameters()
            if id(p) not in head_param_ids and p.requires_grad
        ]

        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": head_params, "lr_scale": 1.0},
        ]
