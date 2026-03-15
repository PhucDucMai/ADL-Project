"""FFmpeg-based video reader using PyAV for efficient video decoding.

PyAV is a Pythonic binding for FFmpeg. It provides direct access to
FFmpeg's libraries, enabling efficient video decoding without the
overhead of OpenCV's VideoCapture.
"""

import logging
from typing import List, Optional, Tuple

import av
import numpy as np

logger = logging.getLogger(__name__)


def read_video_pyav(
    video_path: str,
    num_frames: int,
    frame_stride: int = 1,
    start_frame: Optional[int] = None,
) -> np.ndarray:
    """Read frames from a video file using PyAV (FFmpeg).

    Samples `num_frames` frames with a temporal stride of `frame_stride`.
    If start_frame is None, a random starting position is chosen.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        frame_stride: Temporal stride between sampled frames.
        start_frame: Starting frame index. If None, chosen randomly.

    Returns:
        Array of shape (num_frames, H, W, 3) in RGB format, dtype uint8.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames

    # If total frames is unknown or zero, count them
    if total_frames == 0:
        total_frames = 0
        for _ in container.decode(video=0):
            total_frames += 1
        container.close()
        container = av.open(video_path)

    # Calculate the span of frames needed
    span = num_frames * frame_stride
    max_start = max(0, total_frames - span)

    if start_frame is None:
        start_frame = np.random.randint(0, max(1, max_start + 1))
    else:
        start_frame = min(start_frame, max_start)

    # Determine which frame indices to sample
    frame_indices = set()
    for i in range(num_frames):
        idx = start_frame + i * frame_stride
        idx = min(idx, total_frames - 1)
        frame_indices.add(idx)

    # Ordered list of (original_order, frame_index)
    ordered_indices = []
    for i in range(num_frames):
        idx = start_frame + i * frame_stride
        idx = min(idx, total_frames - 1)
        ordered_indices.append(idx)

    # Decode frames
    frames_dict = {}
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count in frame_indices:
            img = frame.to_ndarray(format="rgb24")
            frames_dict[frame_count] = img
        frame_count += 1
        if frame_count > max(frame_indices):
            break

    container.close()

    # Assemble frames in order
    frames = []
    for idx in ordered_indices:
        if idx in frames_dict:
            frames.append(frames_dict[idx])
        elif frames:
            # Repeat last available frame if index is out of range
            frames.append(frames[-1])

    # Handle case where we got fewer frames than expected
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
            raise RuntimeError(f"Could not read any frames from {video_path}")

    return np.stack(frames[:num_frames])


def read_video_uniform(
    video_path: str,
    num_frames: int,
) -> np.ndarray:
    """Read uniformly sampled frames from a video file.

    Divides the video into `num_frames` equal segments and takes the
    middle frame of each segment. This provides consistent temporal
    coverage regardless of video length.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        Array of shape (num_frames, H, W, 3) in RGB format, dtype uint8.
    """
    container = av.open(video_path)

    # Collect all frames
    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_ndarray(format="rgb24"))

    container.close()

    total = len(all_frames)
    if total == 0:
        raise RuntimeError(f"No frames found in {video_path}")

    # Uniformly sample frame indices
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        # Repeat frames if video is shorter than num_frames
        indices = np.arange(num_frames) % total

    frames = [all_frames[idx] for idx in indices]
    return np.stack(frames)


def get_video_info(video_path: str) -> dict:
    """Get video metadata using PyAV.

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary with video info (width, height, fps, duration, num_frames).
    """
    container = av.open(video_path)
    stream = container.streams.video[0]

    info = {
        "width": stream.codec_context.width,
        "height": stream.codec_context.height,
        "fps": float(stream.average_rate) if stream.average_rate else 0.0,
        "duration": float(stream.duration * stream.time_base) if stream.duration else 0.0,
        "num_frames": stream.frames,
        "codec": stream.codec_context.name,
    }

    container.close()
    return info
