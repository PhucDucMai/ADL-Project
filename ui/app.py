"""Streamlit-based user interface for fighting detection.

Provides a web interface for:
    - Selecting input source (RTSP stream or video file)
    - Starting/stopping detection
    - Viewing real-time results with warning overlays

Usage:
    streamlit run ui/app.py -- --config configs/default.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.detector import FightDetector
from inference.stream_reader import StreamReader
from utils.config import load_config
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def get_config():
    """Load configuration, checking command line args first."""
    config_path = "configs/default.yaml"
    # Check for --config argument passed after '--' in streamlit command
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    return load_config(config_path)


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "running": False,
        "source": "",
        "source_type": "Video File",
        "reader": None,
        "detector": None,
        "frame_counter": 0,
        "last_result": None,
        "warning_countdown": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_detection(config, source: str):
    """Initialize and start the detection components.

    Args:
        config: Configuration object.
        source: Video source path or RTSP URL.
    """
    # Stop existing components
    stop_detection()

    # Initialize stream reader
    reader = StreamReader(
        source=source,
        buffer_size=config.inference.get("buffer_size", 64),
    )
    reader.start()

    # Initialize detector
    detector = FightDetector(config=config)

    st.session_state.reader = reader
    st.session_state.detector = detector
    st.session_state.running = True
    st.session_state.frame_counter = 0
    st.session_state.last_result = None
    st.session_state.warning_countdown = 0

    logger.info("Detection started with source: %s", source)


def stop_detection():
    """Stop all detection components."""
    if st.session_state.reader is not None:
        st.session_state.reader.stop()
        st.session_state.reader = None

    if st.session_state.detector is not None:
        st.session_state.detector.reset()
        st.session_state.detector = None

    st.session_state.running = False
    logger.info("Detection stopped")


def process_and_annotate(config) -> np.ndarray:
    """Process the current frame and return an annotated image.

    Args:
        config: Configuration object.

    Returns:
        Annotated frame in RGB format, or None.
    """
    reader = st.session_state.reader
    detector = st.session_state.detector

    if reader is None or detector is None:
        return None

    frame = reader.get_frame()
    if frame is None:
        return None

    st.session_state.frame_counter += 1

    clip_length = config.inference.get("clip_length", config.model.clip_length)
    frame_stride = config.inference.get("frame_stride", 2)
    inference_interval = config.inference.get("inference_interval", 8)

    # Run inference at the configured interval
    if st.session_state.frame_counter % inference_interval == 0:
        required = clip_length * frame_stride
        frames = reader.get_frames(required)

        if frames is not None:
            indices = list(range(0, required, frame_stride))[:clip_length]
            clip = frames[indices]
            result = detector.predict_with_smoothing(clip)
            st.session_state.last_result = result

            warning_frames = config.inference.get("warning_display_frames", 30)
            if result["is_fight"]:
                st.session_state.warning_countdown = warning_frames

    # Annotate
    display = frame.copy()
    h, w = display.shape[:2]

    result = st.session_state.last_result
    if result is not None:
        prob_fight = result["probabilities"].get("fight", 0)

        # Warning overlay
        if st.session_state.warning_countdown > 0:
            # Red tint overlay
            red_overlay = np.zeros_like(display)
            red_overlay[:, :, 0] = 255  # Red channel
            alpha = 0.15
            display = cv2.addWeighted(display, 1 - alpha, red_overlay, alpha, 0)

            # Red border
            display[:6, :] = [255, 0, 0]
            display[-6:, :] = [255, 0, 0]
            display[:, :6] = [255, 0, 0]
            display[:, -6:] = [255, 0, 0]

            st.session_state.warning_countdown -= 1

    return display


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Fighting Detection System",
        layout="wide",
    )

    setup_logger("root")
    config = get_config()
    init_session_state()

    # Header
    st.title("Fighting Behavior Detection System")
    st.markdown("Real-time abnormal behavior detection using deep learning")

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")

        source_type = st.radio(
            "Input Source",
            options=["Video File", "RTSP Stream"],
            index=0,
        )
        st.session_state.source_type = source_type

        if source_type == "Video File":
            video_path = st.text_input(
                "Video file path",
                value="",
                placeholder="/path/to/video.mp4",
            )
            uploaded = st.file_uploader(
                "Or upload a video file",
                type=["mp4", "avi", "mkv", "mov"],
            )
            if uploaded is not None:
                # Save uploaded file temporarily
                temp_path = Path("data/temp_upload.mp4")
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())
                source = str(temp_path)
            else:
                source = video_path
        else:
            source = st.text_input(
                "RTSP URL",
                value="",
                placeholder="rtsp://192.168.1.100:554/stream",
            )

        st.session_state.source = source

        st.markdown("---")

        # Detection settings
        st.subheader("Detection Settings")
        threshold = st.slider(
            "Confidence Threshold",
            min_value=0.3,
            max_value=0.95,
            value=config.inference.get("confidence_threshold", 0.6),
            step=0.05,
        )
        config.inference.confidence_threshold = threshold

        st.markdown("---")

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button(
                "Start Detection",
                disabled=st.session_state.running or not source,
                use_container_width=True,
            )
        with col2:
            stop_button = st.button(
                "Stop",
                disabled=not st.session_state.running,
                use_container_width=True,
            )

        if start_button and source:
            start_detection(config, source)
            st.rerun()

        if stop_button:
            stop_detection()
            st.rerun()

        # Status
        st.markdown("---")
        st.subheader("Status")
        if st.session_state.running:
            st.success("Detection is running")
            reader = st.session_state.reader
            if reader is not None:
                st.text(f"Source: {reader.source}")
                st.text(f"Buffer: {reader.get_buffer_size()} frames")
                st.text(f"Processed: {st.session_state.frame_counter} frames")
        else:
            st.info("Detection stopped")

    # Main content area
    if st.session_state.running:
        # Create placeholders
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        while st.session_state.running:
            display = process_and_annotate(config)

            if display is not None:
                frame_placeholder.image(
                    display,
                    channels="RGB",
                    use_container_width=True,
                )

                result = st.session_state.last_result
                if result is not None:
                    if result["is_fight"] or st.session_state.warning_countdown > 0:
                        status_placeholder.error(
                            "WARNING: FIGHTING DETECTED "
                            f"(confidence: {result['confidence']:.1%})"
                        )
                    else:
                        status_placeholder.success(
                            f"Status: Normal "
                            f"(fight probability: {result['probabilities'].get('fight', 0):.1%})"
                        )

            time.sleep(0.03)  # ~30 FPS display rate

    else:
        st.info(
            "Select a video source and click 'Start Detection' to begin. "
            "The system will analyze video frames for fighting behavior "
            "and display a red warning when fighting is detected."
        )

        # Show model info
        with st.expander("Model Information"):
            st.markdown(f"""
            - **Architecture**: {config.model.name}
            - **Input Resolution**: {config.model.spatial_size}x{config.model.spatial_size}
            - **Clip Length**: {config.model.clip_length} frames
            - **Classes**: Normal, Fight
            - **Confidence Threshold**: {config.inference.get('confidence_threshold', 0.6):.0%}
            """)


if __name__ == "__main__":
    main()
