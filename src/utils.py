"""
Utility functions for the Long Jump Foul Detection System.
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FoulLine:
    """Represents the foul line as two endpoints in pixel coordinates."""
    x1: int = 0
    y1: int = 400
    x2: int = 1280
    y2: int = 400
    tolerance_px: int = 5
    color: tuple = (0, 0, 255)
    thickness: int = 3

    def as_points(self):
        return np.array([[self.x1, self.y1], [self.x2, self.y2]], dtype=np.int32)


@dataclass
class Detection:
    """A single shoe detection with segmentation mask."""
    bbox: tuple              # (x1, y1, x2, y2)
    confidence: float
    mask: Optional[np.ndarray] = None   # Binary mask (same size as frame)
    class_name: str = "shoe"

    @property
    def bbox_xyxy(self):
        return self.bbox

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def front_bottom(self):
        """The leading bottom corner of the shoe (rightmost-bottom for side view)."""
        x1, y1, x2, y2 = self.bbox
        return (x2, y2)


@dataclass
class FoulEvent:
    """Records a foul event with frame info and replay buffer."""
    frame_number: int
    timestamp: float
    foul_frame: np.ndarray      # The annotated frame where foul was detected
    replay_frames: list = field(default_factory=list)  # Frames before foul


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_foul_line_from_config(config: dict) -> FoulLine:
    """Extract FoulLine object from config dictionary."""
    fl = config.get("foul_line", {})
    color = fl.get("color", [0, 0, 255])
    return FoulLine(
        x1=fl.get("x1", 0),
        y1=fl.get("y1", 400),
        x2=fl.get("x2", 1280),
        y2=fl.get("y2", 400),
        tolerance_px=fl.get("tolerance_px", 5),
        color=tuple(color),
        thickness=fl.get("thickness", 3),
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_foul_line(frame: np.ndarray, foul_line: FoulLine) -> np.ndarray:
    """Draw the foul line on the frame."""
    cv2.line(
        frame,
        (foul_line.x1, foul_line.y1),
        (foul_line.x2, foul_line.y2),
        foul_line.color,
        foul_line.thickness,
    )
    return frame


def draw_detections(
    frame: np.ndarray,
    detections: list,
    foul_line: Optional[FoulLine] = None,
    is_foul: bool = False,
) -> np.ndarray:
    """Draw detection bounding boxes and segmentation masks on the frame."""
    overlay = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = (0, 0, 255) if is_foul else (0, 255, 0)

        # Draw segmentation mask
        if det.mask is not None:
            mask_resized = cv2.resize(det.mask, (frame.shape[1], frame.shape[0]))
            mask_bool = mask_resized.astype(bool)
            overlay[mask_bool] = (
                overlay[mask_bool] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    # Blend mask overlay
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Draw foul line on top
    if foul_line is not None:
        draw_foul_line(frame, foul_line)

    return frame


def draw_foul_alert(frame: np.ndarray, is_foul: bool) -> np.ndarray:
    """Draw a large FOUL / NO FOUL banner on the frame."""
    h, w = frame.shape[:2]

    if is_foul:
        # Red banner
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 180), -1)
        cv2.putText(
            frame, "FOUL", (w // 2 - 80, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4,
        )
    else:
        # Green banner (smaller, less intrusive)
        cv2.putText(
            frame, "NO FOUL", (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2,
        )

    return frame


def draw_frame_counter(frame: np.ndarray, frame_num: int, fps: float) -> np.ndarray:
    """Draw frame counter and FPS info."""
    h, w = frame.shape[:2]
    text = f"Frame: {frame_num} | FPS: {fps:.1f}"
    cv2.putText(
        frame, text, (w - 280, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )
    return frame


# ---------------------------------------------------------------------------
# Frame buffer for replay
# ---------------------------------------------------------------------------

class FrameBuffer:
    """Circular buffer to keep the last N frames for replay."""

    def __init__(self, max_size: int = 60):
        self.buffer = deque(maxlen=max_size)

    def add(self, frame: np.ndarray, frame_num: int):
        self.buffer.append((frame_num, frame.copy()))

    def get_all(self):
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def create_video_writer(
    output_path: str, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter for saving output video."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
