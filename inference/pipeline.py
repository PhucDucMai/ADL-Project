"""Real-time inference pipeline for fighting detection.

Combines the StreamReader and FightDetector to provide a complete
inference pipeline that reads frames from a video source, runs
detection, and overlays results on the displayed frames.
"""

import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

from inference.detector import FightDetector
from inference.stream_reader import StreamReader
from utils.config import Config

logger = logging.getLogger(__name__)

# Colors in BGR for OpenCV overlay
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


class InferencePipeline:
    """Complete inference pipeline for real-time fighting detection.

    Orchestrates the stream reader and detector to process video
    frames and produce annotated output frames.
    """

    def __init__(
        self,
        config: Config,
        source: str,
        model_path: Optional[str] = None,
    ):
        """Initialize the inference pipeline.

        Args:
            config: Configuration object.
            source: Video source (RTSP URL or file path).
            model_path: Optional path to model checkpoint.
        """
        self.config = config
        self.source = source

        # Initialize components
        buffer_size = config.inference.get("buffer_size", 64)
        self.reader = StreamReader(
            source=source,
            buffer_size=buffer_size,
        )
        self.detector = FightDetector(
            config=config,
            model_path=model_path,
        )

        # Pipeline settings
        self.clip_length = config.inference.get("clip_length", config.model.clip_length)
        self.frame_stride = config.inference.get("frame_stride", 2)
        self.inference_interval = config.inference.get("inference_interval", 8)
        self.warning_frames = config.inference.get("warning_display_frames", 30)

        # State
        self.frame_counter = 0
        self.last_result = None
        self.warning_countdown = 0
        self.running = False

        logger.info(
            "Pipeline initialized: source=%s, clip_length=%d, "
            "inference_interval=%d",
            source, self.clip_length, self.inference_interval,
        )

    def start(self):
        """Start the pipeline (begins reading frames)."""
        self.reader.start()
        self.running = True
        logger.info("Inference pipeline started")

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        self.reader.stop()
        self.detector.reset()
        logger.info("Inference pipeline stopped")

    def process_frame(self) -> Optional[np.ndarray]:
        """Process one iteration of the pipeline.

        Reads the current frame, runs inference if needed,
        and returns an annotated frame.

        Returns:
            Annotated frame as numpy array (H, W, 3) in BGR format
            for display, or None if no frame is available.
        """
        frame = self.reader.get_frame()
        if frame is None:
            return None

        self.frame_counter += 1

        # Run inference at configured interval
        if self.frame_counter % self.inference_interval == 0:
            required_frames = self.clip_length * self.frame_stride
            frames = self.reader.get_frames(required_frames)

            if frames is not None:
                # Subsample frames according to stride
                indices = list(range(0, required_frames, self.frame_stride))[:self.clip_length]
                clip = frames[indices]

                result = self.detector.predict_with_smoothing(clip)
                self.last_result = result

                if result["is_fight"]:
                    self.warning_countdown = self.warning_frames
                    logger.warning(
                        "FIGHTING DETECTED - confidence: %.2f",
                        result["confidence"],
                    )

        # Annotate frame
        display_frame = self._annotate_frame(frame)

        if self.warning_countdown > 0:
            self.warning_countdown -= 1

        return display_frame

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add detection overlay to a frame.

        Args:
            frame: RGB frame array (H, W, 3).

        Returns:
            Annotated frame in BGR format (H, W, 3).
        """
        # Convert RGB to BGR for OpenCV
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = display.shape[:2]

        if self.last_result is not None:
            result = self.last_result
            prob_normal = result["probabilities"].get("normal", 0)
            prob_fight = result["probabilities"].get("fight", 0)

            # Status bar at the top
            bar_height = 40
            cv2.rectangle(display, (0, 0), (w, bar_height), COLOR_BLACK, -1)

            status_text = (
                f"Normal: {prob_normal:.1%}  |  Fight: {prob_fight:.1%}  |  "
                f"FPS: {self.reader.fps:.0f}"
            )
            cv2.putText(
                display, status_text,
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLOR_WHITE, 1, cv2.LINE_AA,
            )

            # Warning overlay when fighting detected
            if self.warning_countdown > 0:
                # Semi-transparent red overlay
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
                alpha = 0.15
                cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)

                # Red border
                border_thickness = 6
                cv2.rectangle(
                    display,
                    (0, 0), (w - 1, h - 1),
                    COLOR_RED, border_thickness,
                )

                # Warning text
                warning_text = "WARNING: FIGHTING DETECTED"
                text_size = cv2.getTextSize(
                    warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3,
                )[0]
                text_x = (w - text_size[0]) // 2
                text_y = h - 50

                # Background for text
                cv2.rectangle(
                    display,
                    (text_x - 15, text_y - text_size[1] - 15),
                    (text_x + text_size[0] + 15, text_y + 15),
                    COLOR_RED, -1,
                )
                cv2.putText(
                    display, warning_text,
                    (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    COLOR_WHITE, 3, cv2.LINE_AA,
                )

                # Confidence text
                conf_text = f"Confidence: {result['confidence']:.1%}"
                conf_size = cv2.getTextSize(
                    conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2,
                )[0]
                conf_x = (w - conf_size[0]) // 2
                conf_y = text_y + 40
                cv2.putText(
                    display, conf_text,
                    (conf_x, conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    COLOR_RED, 2, cv2.LINE_AA,
                )

        return display

    def run_display(self, window_name: str = "Fighting Detection"):
        """Run the pipeline with OpenCV window display.

        Press 'q' or ESC to quit.

        Args:
            window_name: Name of the display window.
        """
        self.start()

        try:
            while self.running:
                display = self.process_frame()
                if display is not None:
                    cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

                # Small sleep to prevent CPU saturation
                time.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            self.stop()
            cv2.destroyAllWindows()


def run_inference(config: Config, source: str, model_path: Optional[str] = None):
    """Run the inference pipeline from command line.

    Args:
        config: Configuration object.
        source: Video source (RTSP URL or file path).
        model_path: Optional path to model checkpoint.
    """
    pipeline = InferencePipeline(config, source, model_path)
    pipeline.run_display()
