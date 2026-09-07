"""
Camera capture module for the Long Jump Foul Detection System.
Supports webcam, USB cameras, and video files.
"""

import cv2
import os
from typing import Optional, Generator


class Camera:
    """Handles video capture from webcam or video file."""

    def __init__(
        self,
        source=0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_video_file = isinstance(source, str) and os.path.isfile(source)

    def open(self) -> bool:
        """Open the camera or video file."""
        if self.is_video_file:
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            print(f"[Camera] ERROR: Cannot open source: {self.source}")
            return False

        # Read actual properties (camera may not support requested values)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"[Camera] Opened: {self.source}")
        print(f"[Camera] Resolution: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        return True

    def read(self):
        """Read a single frame. Returns (success, frame)."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def frames(self) -> Generator:
        """Generator that yields (frame_number, frame) tuples."""
        frame_num = 0
        while True:
            success, frame = self.read()
            if not success:
                break
            yield frame_num, frame
            frame_num += 1

    def get_total_frames(self) -> int:
        """Get total number of frames (only valid for video files)."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[Camera] Released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()

    def __del__(self):
        self.release()
