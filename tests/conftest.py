"""Pytest configuration and shared fixtures for fighting detection tests.

Provides fixtures for:
- Device selection (CPU/GPU)
- Dummy video data for testing
- Model instances
- Configuration objects
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import av


@pytest.fixture(scope="session")
def device():
    """Get device for testing (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def tmp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_video_tensor():
    """Create dummy video tensor for testing.

    Returns:
        Tensor of shape (B, C, T, H, W) = (2, 3, 16, 224, 224)
    """
    batch_size = 2
    channels = 3
    frames = 16
    height = 224
    width = 224
    return torch.randn(batch_size, channels, frames, height, width)


@pytest.fixture
def dummy_labels():
    """Create dummy labels for testing.

    Returns:
        Tensor of shape (B,) with binary labels
    """
    return torch.tensor([0, 1], dtype=torch.long)


@pytest.fixture
def sample_config():
    """Load sample configuration for testing."""
    from utils.config import load_config

    return load_config("configs/default.yaml")


@pytest.fixture
def test_video_path(tmp_data_dir):
    """Create a small test video file.

    Returns:
        Path to test video file
    """
    video_path = tmp_data_dir / "test_video.mp4"

    # Create a simple test video with 10 frames
    container = av.open(str(video_path), mode="w")
    stream = container.add_stream("libx264", rate=24)
    stream.width = 224
    stream.height = 224
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "fast"}

    for i in range(10):
        # Create random frame
        frame_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return video_path


@pytest.fixture
def test_data_structure(tmp_data_dir):
    """Create test dataset structure with sample videos.

    Creates:
        train/fight/, train/normal/, val/fight/, val/normal/
        Each with a small test video

    Returns:
        Dictionary with paths to train and val directories
    """
    data_dir = tmp_data_dir / "data"

    for split in ["train", "val"]:
        for class_name in ["fight", "normal"]:
            class_dir = data_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Create a test video in each class
            video_path = class_dir / f"{class_name}_sample.mp4"

            container = av.open(str(video_path), mode="w")
            stream = container.add_stream("libx264", rate=24)
            stream.width = 224
            stream.height = 224
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "fast"}

            for i in range(10):
                frame_array = (
                    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                )
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

            container.close()

    return {
        "train_dir": str(data_dir / "train"),
        "val_dir": str(data_dir / "val"),
    }


@pytest.fixture
def model_configs():
    """Return list of all available model names for testing."""
    from models.factory import list_available_models

    return list_available_models()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
