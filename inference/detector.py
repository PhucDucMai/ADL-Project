"""Fighting behavior detector using trained model.

Loads a trained model checkpoint and performs inference on video clips.
Supports temporal smoothing to reduce false positives.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.transforms import NormalizeVideo, ResizeVideo, CenterCropVideo, VideoTransform
from models.factory import create_model
from utils.config import Config

logger = logging.getLogger(__name__)


class FightDetector:
    """Inference wrapper for the fighting detection model.

    Loads a pretrained model and provides methods for single-clip
    and streaming inference with temporal smoothing.
    """

    CLASS_NAMES = ["normal", "fight"]

    def __init__(
        self,
        config: Config,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the detector.

        Args:
            config: Configuration object.
            model_path: Path to the model checkpoint. Overrides config if set.
            device: Device to run inference on. Overrides config if set.
        """
        self.config = config
        self.device = torch.device(
            device or config.inference.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.clip_length = config.inference.get("clip_length", config.model.clip_length)
        self.spatial_size = config.inference.get("spatial_size", config.model.spatial_size)
        self.confidence_threshold = config.inference.get("confidence_threshold", 0.6)

        # Temporal smoothing
        smoothing_window = config.inference.get("temporal_smoothing_window", 3)
        self.prediction_history: deque = deque(maxlen=smoothing_window)

        # Build inference transforms
        self.transform = VideoTransform([
            ResizeVideo(self.spatial_size),
            CenterCropVideo(self.spatial_size),
            NormalizeVideo(),
        ])

        # Load model
        model_path = model_path or config.inference.get("model_path", "checkpoints/best_model.pth")
        self.model = self._load_model(model_path)

        logger.info(
            "Detector initialized: device=%s, threshold=%.2f, clip_length=%d",
            self.device, self.confidence_threshold, self.clip_length,
        )

    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint.

        Args:
            model_path: Path to the checkpoint file.

        Returns:
            Loaded model in eval mode.
        """
        model = create_model(self.config)

        checkpoint_path = Path(model_path)
        if checkpoint_path.exists():
            logger.info("Loading model weights from: %s", model_path)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            logger.info("Model weights loaded successfully")
        else:
            logger.warning(
                "Checkpoint not found at %s. Using model with pretrained backbone only.",
                model_path,
            )

        model = model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_clip(self, frames: np.ndarray) -> Dict:
        """Run inference on a single video clip.

        Args:
            frames: Array of shape (T, H, W, 3) in uint8 RGB format.

        Returns:
            Dictionary with:
                - label: Predicted class name ("normal" or "fight")
                - label_idx: Predicted class index
                - confidence: Prediction confidence (0-1)
                - probabilities: Per-class probabilities
                - is_fight: Boolean indicating fighting detection
        """
        # Apply transforms: (T, H, W, C) -> (C, T, H, W)
        clip = self.transform(frames)
        # Add batch dimension: (1, C, T, H, W)
        clip = clip.unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(clip)
        probs = F.softmax(logits, dim=1)

        # Get prediction
        confidence, pred_idx = probs.max(dim=1)
        confidence = confidence.item()
        pred_idx = pred_idx.item()
        label = self.CLASS_NAMES[pred_idx]

        return {
            "label": label,
            "label_idx": pred_idx,
            "confidence": confidence,
            "probabilities": {
                name: probs[0, i].item()
                for i, name in enumerate(self.CLASS_NAMES)
            },
            "is_fight": (
                pred_idx == 1 and confidence >= self.confidence_threshold
            ),
        }

    def predict_with_smoothing(self, frames: np.ndarray) -> Dict:
        """Run inference with temporal smoothing.

        Maintains a history of recent predictions and averages their
        probabilities to produce a smoother detection signal.

        Args:
            frames: Array of shape (T, H, W, 3) in uint8 RGB format.

        Returns:
            Smoothed prediction dictionary (same format as predict_clip).
        """
        result = self.predict_clip(frames)
        self.prediction_history.append(result["probabilities"])

        # Average probabilities over the smoothing window
        if len(self.prediction_history) > 1:
            avg_probs = {}
            for name in self.CLASS_NAMES:
                avg_probs[name] = np.mean([
                    p[name] for p in self.prediction_history
                ])

            # Determine smoothed prediction
            smoothed_idx = max(range(len(self.CLASS_NAMES)),
                               key=lambda i: avg_probs[self.CLASS_NAMES[i]])
            smoothed_label = self.CLASS_NAMES[smoothed_idx]
            smoothed_conf = avg_probs[smoothed_label]

            result = {
                "label": smoothed_label,
                "label_idx": smoothed_idx,
                "confidence": smoothed_conf,
                "probabilities": avg_probs,
                "is_fight": (
                    smoothed_idx == 1 and smoothed_conf >= self.confidence_threshold
                ),
            }

        return result

    def reset(self):
        """Reset the prediction history for temporal smoothing."""
        self.prediction_history.clear()
