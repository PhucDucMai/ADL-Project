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
        checkpoint_candidates = self._resolve_checkpoint_candidates(model_path)
        model_pretrained = bool(self.config.model.get("pretrained", True))

        if checkpoint_candidates and model_pretrained:
            logger.info(
                "Checkpoint detected. Building model with pretrained=False to avoid "
                "external weight downloads during inference."
            )
            self.config.model.pretrained = False
            try:
                model = create_model(self.config)
            finally:
                self.config.model.pretrained = model_pretrained
        else:
            try:
                model = create_model(self.config)
            except Exception as e:
                if not model_pretrained:
                    raise
                logger.warning(
                    "Pretrained model initialization failed (%s). "
                    "Retrying with pretrained=False.",
                    e,
                )
                self.config.model.pretrained = False
                try:
                    model = create_model(self.config)
                finally:
                    self.config.model.pretrained = model_pretrained

        loaded = False
        for checkpoint_path in checkpoint_candidates:
            logger.info("Loading model weights from: %s", checkpoint_path)
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                state_dict = self._extract_state_dict(checkpoint)
                filtered_state_dict, load_stats = self._adapt_state_dict_for_model(model, state_dict)

                if load_stats["matched"] == 0:
                    logger.warning(
                        "Skipping incompatible checkpoint (0 matched tensors): %s",
                        checkpoint_path,
                    )
                    continue

                load_result = model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(
                    "Model weights loaded: matched=%d/%d (missing=%d, unexpected=%d)",
                    load_stats["matched"],
                    load_stats["target_total"],
                    len(load_result.missing_keys),
                    len(load_result.unexpected_keys),
                )
                if load_stats["matched"] < max(1, int(0.5 * load_stats["target_total"])):
                    logger.warning(
                        "Only partial checkpoint load was possible (%d/%d tensors). "
                        "This often means model mismatch or classifier-head size mismatch.",
                        load_stats["matched"],
                        load_stats["target_total"],
                    )
                loaded = True
                break
            except Exception as e:
                logger.warning("Failed to load checkpoint %s: %s", checkpoint_path, e)

        if checkpoint_candidates and not loaded:
            logger.warning(
                "No compatible checkpoint could be loaded from %d candidate(s). "
                "Continuing with model initialized from current config.",
                len(checkpoint_candidates),
            )
        elif not checkpoint_candidates:
            logger.warning(
                "Checkpoint not found at %s (or discoverable alternatives). "
                "Using model with pretrained backbone only.",
                model_path,
            )

        model = model.to(self.device)
        model.eval()
        return model

    def _resolve_checkpoint_candidates(self, model_path: str) -> list[Path]:
        """Resolve checkpoint candidates, with automatic fallback discovery.

        Supports explicit files and run-organized training outputs like:
        checkpoints/<model_name>/<run_id>/best_model.pth
        """
        explicit_path = Path(model_path)
        explicit_candidates = []
        if explicit_path.exists() and explicit_path.is_file():
            explicit_candidates.append(explicit_path)

        model_name = str(self.config.model.name)
        search_roots = []

        training_cfg = getattr(self.config, "training", None)
        if training_cfg is not None:
            checkpoint_root = training_cfg.get("checkpoint_dir", None)
            if checkpoint_root:
                search_roots.append(Path(checkpoint_root))

        search_roots.append(Path("checkpoints"))

        candidates = []
        for root in search_roots:
            if not root.exists():
                continue

            patterns = [
                f"{model_name}/**/best_model.pth",
                f"{model_name}/**/final_model.pth",
                f"{model_name}/best_model.pth",
                f"{model_name}/final_model.pth",
                "best_model.pth",
                "final_model.pth",
            ]

            for pattern in patterns:
                for p in root.glob(pattern):
                    if p.is_file():
                        candidates.append(p)

        if not candidates:
            return explicit_candidates

        candidates = sorted(
            set(candidates),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        combined = []
        seen = set()
        for p in explicit_candidates + candidates:
            if p not in seen:
                combined.append(p)
                seen.add(p)

        if combined and not explicit_candidates:
            logger.warning(
                "Configured checkpoint '%s' not found. Trying auto-discovered candidates "
                "starting with latest: %s",
                model_path,
                combined[0],
            )
        return combined

    @staticmethod
    def _extract_state_dict(checkpoint) -> Dict:
        """Extract state dict from common checkpoint formats."""
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            # Raw state dict case
            if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint
        raise RuntimeError(
            "Unsupported checkpoint format. Expected one of keys: "
            "'model_state_dict', 'state_dict', or raw tensor state dict."
        )

    @staticmethod
    def _adapt_state_dict_for_model(model: nn.Module, state_dict: Dict) -> Tuple[Dict, Dict[str, int]]:
        """Map checkpoint tensors to current model keys with shape validation."""
        target_state = model.state_dict()

        def strip_prefix(sd: Dict, prefix: str) -> Dict:
            return {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in sd.items()
            }

        variants = []
        variants.append(("raw", state_dict))

        no_module = strip_prefix(state_dict, "module.")
        variants.append(("no_module", no_module))

        no_module_no_model = strip_prefix(no_module, "model.")
        variants.append(("no_module_no_model", no_module_no_model))

        add_model_prefix = {f"model.{k}": v for k, v in no_module.items()}
        variants.append(("add_model_prefix", add_model_prefix))

        best_name = "raw"
        best_match_count = -1
        best_matched = {}

        for variant_name, sd_variant in variants:
            matched = {}
            for k, v in sd_variant.items():
                if k not in target_state:
                    continue
                if target_state[k].shape != v.shape:
                    continue
                matched[k] = v

            if len(matched) > best_match_count:
                best_match_count = len(matched)
                best_name = variant_name
                best_matched = matched

        logger.info(
            "Checkpoint key adaptation selected variant '%s' (%d matched tensors)",
            best_name,
            len(best_matched),
        )

        stats = {
            "matched": len(best_matched),
            "source_total": len(state_dict),
            "target_total": len(target_state),
        }
        return best_matched, stats

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
