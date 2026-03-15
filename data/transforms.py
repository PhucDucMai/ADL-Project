"""Video transforms for training and evaluation.

Implements spatial and temporal augmentations for video clips.
All transforms operate on tensors of shape (T, H, W, C) or (C, T, H, W).
"""

import logging
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.45, 0.45, 0.45]
IMAGENET_STD = [0.225, 0.225, 0.225]


class VideoTransform:
    """Composable video transform pipeline.

    Applies spatial transforms consistently across all frames in a clip.
    """

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """Apply transforms to a video clip.

        Args:
            frames: Array of shape (T, H, W, C) in uint8 RGB format.

        Returns:
            Tensor of shape (C, T, H, W) normalized and transformed.
        """
        # Convert to float tensor: (T, H, W, C) -> (T, C, H, W)
        clip = torch.from_numpy(frames).float() / 255.0
        clip = clip.permute(0, 3, 1, 2)  # (T, C, H, W)

        for transform in self.transforms:
            clip = transform(clip)

        # Rearrange to (C, T, H, W) for model input
        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
        return clip


class ResizeVideo:
    """Resize all frames in a clip to the target size."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """Resize clip frames.

        Args:
            clip: Tensor of shape (T, C, H, W).

        Returns:
            Resized tensor.
        """
        T_dim = clip.shape[0]
        # Resize each frame
        resized = []
        for t in range(T_dim):
            frame = F.resize(clip[t], [self.size, self.size], antialias=True)
            resized.append(frame)
        return torch.stack(resized)


class RandomCropVideo:
    """Randomly crop all frames in a clip with the same spatial crop."""

    def __init__(self, size: int, scale: Tuple[float, float] = (0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        _, _, H, W = clip.shape
        # Random scale
        scale = np.random.uniform(self.scale[0], self.scale[1])
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)

        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)

        clip = clip[:, :, top:top + crop_h, left:left + crop_w]

        # Resize to target size
        resized = []
        for t in range(clip.shape[0]):
            frame = F.resize(clip[t], [self.size, self.size], antialias=True)
            resized.append(frame)
        return torch.stack(resized)


class CenterCropVideo:
    """Center crop all frames in a clip."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        _, _, H, W = clip.shape
        crop_h = min(self.size, H)
        crop_w = min(self.size, W)
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2

        clip = clip[:, :, top:top + crop_h, left:left + crop_w]

        if crop_h != self.size or crop_w != self.size:
            resized = []
            for t in range(clip.shape[0]):
                frame = F.resize(clip[t], [self.size, self.size], antialias=True)
                resized.append(frame)
            return torch.stack(resized)
        return clip


class RandomHorizontalFlipVideo:
    """Randomly flip all frames horizontally with same decision."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.p:
            return clip.flip(-1)
        return clip


class ColorJitterVideo:
    """Apply color jitter consistently across all frames."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        # Get transform parameters once and apply to all frames
        params = self.jitter.get_params(
            brightness=self.jitter.brightness,
            contrast=self.jitter.contrast,
            saturation=self.jitter.saturation,
            hue=self.jitter.hue,
        )

        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params

        result = []
        for t in range(clip.shape[0]):
            frame = clip[t]
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    frame = F.adjust_brightness(frame, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    frame = F.adjust_contrast(frame, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    frame = F.adjust_saturation(frame, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    frame = F.adjust_hue(frame, hue_factor)
            result.append(frame)
        return torch.stack(result)


class NormalizeVideo:
    """Normalize all frames with given mean and std."""

    def __init__(self, mean=None, std=None):
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=clip.dtype, device=clip.device)
        std = torch.tensor(self.std, dtype=clip.dtype, device=clip.device)
        # clip shape: (T, C, H, W)
        clip = (clip - mean[None, :, None, None]) / std[None, :, None, None]
        return clip


def get_train_transforms(config) -> VideoTransform:
    """Build training transforms from config.

    Args:
        config: Configuration object with data augmentation settings.

    Returns:
        VideoTransform pipeline for training.
    """
    spatial_size = config.data.spatial_size
    transforms_list = []

    # Resize to slightly larger than target for cropping
    resize_size = int(spatial_size * 1.15)
    transforms_list.append(ResizeVideo(resize_size))

    # Random crop
    transforms_list.append(RandomCropVideo(
        size=spatial_size,
        scale=(
            config.data.get("random_crop_scale_min", 0.8),
            config.data.get("random_crop_scale_max", 1.0),
        ),
    ))

    # Random horizontal flip
    if config.data.get("random_horizontal_flip", True):
        transforms_list.append(RandomHorizontalFlipVideo(p=0.5))

    # Color jitter
    if config.data.get("color_jitter", False):
        transforms_list.append(ColorJitterVideo(
            brightness=config.data.get("color_jitter_brightness", 0.2),
            contrast=config.data.get("color_jitter_contrast", 0.2),
            saturation=config.data.get("color_jitter_saturation", 0.2),
            hue=config.data.get("color_jitter_hue", 0.1),
        ))

    # Normalize
    transforms_list.append(NormalizeVideo())

    return VideoTransform(transforms_list)


def get_val_transforms(config) -> VideoTransform:
    """Build validation/inference transforms from config.

    Args:
        config: Configuration object with data settings.

    Returns:
        VideoTransform pipeline for validation/inference.
    """
    spatial_size = config.data.spatial_size
    transforms_list = [
        ResizeVideo(spatial_size),
        CenterCropVideo(spatial_size),
        NormalizeVideo(),
    ]

    return VideoTransform(transforms_list)
