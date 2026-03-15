"""Dataset class for fighting detection training.

Expects a directory structure:
    data/raw/train/
        fight/
            video001.avi
            video002.mp4
            ...
        normal/
            video001.avi
            video002.mp4
            ...
    data/raw/val/
        fight/
            ...
        normal/
            ...

Supported video formats: .avi, .mp4, .mkv, .mov, .wmv
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.video_reader import read_video_pyav, read_video_uniform
from data.transforms import VideoTransform

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


class FightDataset(Dataset):
    """PyTorch Dataset for fight/normal video classification.

    Each sample is a video clip loaded using PyAV (FFmpeg) and
    transformed into a tensor suitable for 3D CNN input.
    """

    def __init__(
        self,
        data_dir: str,
        clip_length: int = 13,
        frame_stride: int = 2,
        transform: Optional[VideoTransform] = None,
        classes: Optional[List[str]] = None,
        sampling_mode: str = "random",
    ):
        """Initialize the dataset.

        Args:
            data_dir: Root directory containing class subdirectories.
            clip_length: Number of frames per clip.
            frame_stride: Temporal stride between frames.
            transform: Video transform pipeline.
            classes: List of class names. Defaults to ["normal", "fight"].
            sampling_mode: Frame sampling strategy ("random" or "uniform").
        """
        self.data_dir = Path(data_dir)
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.transform = transform
        self.sampling_mode = sampling_mode

        if classes is None:
            classes = ["normal", "fight"]
        self.classes = classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Discover video files
        self.samples: List[Tuple[str, int]] = []
        self._discover_samples()

        logger.info(
            "Dataset initialized: %d samples from %s (%s)",
            len(self.samples),
            data_dir,
            ", ".join(f"{cls}={self._count_class(idx)}" for cls, idx in self.class_to_idx.items()),
        )

    def _discover_samples(self):
        """Scan the data directory for video files organized by class."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning("Class directory not found: %s", class_dir)
                continue

            for video_path in sorted(class_dir.iterdir()):
                if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                    self.samples.append((str(video_path), class_idx))

        if not self.samples:
            logger.warning("No video samples found in %s", self.data_dir)

    def _count_class(self, class_idx: int) -> int:
        """Count the number of samples for a given class index."""
        return sum(1 for _, idx in self.samples if idx == class_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load and transform a video clip.

        Args:
            index: Sample index.

        Returns:
            Tuple of (clip_tensor, label) where clip_tensor has shape (C, T, H, W).
        """
        video_path, label = self.samples[index]

        try:
            if self.sampling_mode == "uniform":
                frames = read_video_uniform(video_path, self.clip_length)
            else:
                frames = read_video_pyav(
                    video_path,
                    num_frames=self.clip_length,
                    frame_stride=self.frame_stride,
                )

            if self.transform is not None:
                clip = self.transform(frames)
            else:
                # Default: convert to tensor (C, T, H, W)
                clip = torch.from_numpy(frames).float() / 255.0
                clip = clip.permute(3, 0, 1, 2)

            return clip, label

        except Exception as e:
            logger.error("Error loading video %s: %s", video_path, str(e))
            # Return a zero tensor and the label on error
            if self.transform is not None:
                h = w = 182
            else:
                h = w = 182
            clip = torch.zeros(3, self.clip_length, h, w)
            return clip, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training.

        Returns:
            Tensor of class weights.
        """
        counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1

        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(self.classes)

        return torch.tensor(weights, dtype=torch.float32)
