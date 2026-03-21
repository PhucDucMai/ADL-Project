"""FFmpeg-based stream reader for RTSP streams and video files.

Uses PyAV (FFmpeg bindings) for efficient video decoding with support for:
    - RTSP streams (e.g., rtsp://192.168.1.100:554/stream)
    - Video files (e.g., /path/to/video.mp4)
    - Threaded frame reading for non-blocking operation
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

import av
import numpy as np

logger = logging.getLogger(__name__)


class StreamReader:
    """Threaded video stream reader using PyAV (FFmpeg).

    Continuously reads frames from a video source in a background thread
    and stores them in a thread-safe buffer. Supports both RTSP streams
    and local video files.
    """

    def __init__(
        self,
        source: str,
        buffer_size: int = 64,
        rtsp_transport: str = "tcp",
        reconnect_delay: float = 2.0,
        loop_file: bool = True,
    ):
        """Initialize the stream reader.

        Args:
            source: RTSP URL or path to a video file.
            buffer_size: Maximum number of frames to buffer.
            rtsp_transport: RTSP transport protocol ("tcp" or "udp").
            reconnect_delay: Seconds to wait before reconnecting on error.
            loop_file: Whether to loop video files when they end.
        """
        self.source = source
        self.buffer_size = buffer_size
        self.rtsp_transport = rtsp_transport
        self.reconnect_delay = reconnect_delay
        self.loop_file = loop_file

        self.is_rtsp = source.lower().startswith("rtsp://")
        self.buffer: deque = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.fps: float = 30.0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.frame_count: int = 0

    def start(self):
        """Start the background frame reading thread."""
        if self.running:
            logger.warning("Stream reader already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("Stream reader started: %s", self.source)

    def stop(self):
        """Stop the background frame reading thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None
        with self.lock:
            self.buffer.clear()
        logger.info("Stream reader stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the most recent frame from the buffer.

        Returns:
            Frame as numpy array (H, W, 3) in RGB format, or None if buffer is empty.
        """
        with self.lock:
            if len(self.buffer) > 0:
                return self.buffer[-1]
        return None

    def get_frames(self, count: int) -> Optional[np.ndarray]:
        """Get the most recent N frames from the buffer.

        Args:
            count: Number of frames to retrieve.

        Returns:
            Array of shape (count, H, W, 3) or None if not enough frames.
        """
        with self.lock:
            if len(self.buffer) >= count:
                frames = list(self.buffer)[-count:]
                return np.stack(frames)
        return None

    def get_buffer_size(self) -> int:
        """Get the current number of frames in the buffer."""
        with self.lock:
            return len(self.buffer)

    def _open_container(self) -> av.container.InputContainer:
        """Open a video container with appropriate options."""
        options = {}
        if self.is_rtsp:
            options = {
                "rtsp_transport": self.rtsp_transport,
                "stimeout": "5000000",  # 5 second timeout
                "max_delay": "500000",
                "fflags": "nobuffer",
                "flags": "low_delay",
            }

        container = av.open(self.source, options=options)
        stream = container.streams.video[0]

        # Enable threading for faster decoding
        stream.thread_type = "AUTO"

        self.fps = float(stream.average_rate) if stream.average_rate else 30.0
        self.frame_width = stream.codec_context.width
        self.frame_height = stream.codec_context.height

        logger.info(
            "Opened stream: %dx%d @ %.1f fps",
            self.frame_width, self.frame_height, self.fps,
        )
        return container

    def _read_loop(self):
        """Main frame reading loop running in a background thread."""
        while self.running:
            try:
                container = self._open_container()

                for frame in container.decode(video=0):
                    if not self.running:
                        break

                    img = frame.to_ndarray(format="rgb24")
                    with self.lock:
                        self.buffer.append(img)
                    self.frame_count += 1

                container.close()

                # End of file handling
                if not self.is_rtsp:
                    if self.loop_file:
                        logger.info("Video file ended, restarting")
                        continue
                    else:
                        logger.info("Video file ended")
                        self.running = False
                        break

            except av.error.InvalidDataError as e:
                logger.error("Stream decode error: %s", str(e))
                if not self.is_rtsp:
                    # For files, corrupt data means the file is damaged past
                    # this point — stop instead of re-opening from the start.
                    logger.warning(
                        "Corrupt data in video file at frame %d, stopping",
                        self.frame_count,
                    )
                    self.running = False
                    break
                if self.running:
                    time.sleep(self.reconnect_delay)

            except av.error.EOFError:
                if not self.is_rtsp and self.loop_file:
                    logger.info("EOF reached, restarting video")
                    continue
                elif self.is_rtsp:
                    logger.warning("RTSP stream ended, reconnecting")
                    time.sleep(self.reconnect_delay)
                else:
                    self.running = False
                    break

            except Exception as e:
                logger.error("Stream reader error: %s", str(e))
                if self.running:
                    time.sleep(self.reconnect_delay)

        logger.info("Stream reader loop exited (total frames: %d)", self.frame_count)

    @property
    def is_alive(self) -> bool:
        """Check if the reader thread is still running."""
        return self.running and self.thread is not None and self.thread.is_alive()
