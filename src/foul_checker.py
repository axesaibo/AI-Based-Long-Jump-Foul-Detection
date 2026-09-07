"""
Foul line checking logic for the Long Jump Foul Detection System.
Determines whether a detected shoe has crossed the foul line using
bounding box and/or segmentation mask analysis.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple

from utils import Detection, FoulLine


class FoulChecker:
    """
    Checks whether any detected shoe crosses the foul line.

    Two modes of operation:
    1. Bounding box mode — checks if the shoe bbox overlaps the line (always available)
    2. Segmentation mask mode — checks if actual shoe pixels cross the line (more accurate)
    """

    def __init__(
        self,
        foul_line: FoulLine,
        min_consecutive_frames: int = 2,
        tolerance_px: int = 5,
    ):
        self.foul_line = foul_line
        self.min_consecutive_frames = min_consecutive_frames
        self.tolerance_px = tolerance_px or foul_line.tolerance_px

        # State tracking
        self.consecutive_foul_count = 0
        self.foul_detected = False
        self.foul_frame_number: Optional[int] = None

    def reset(self):
        """Reset state for a new jump attempt."""
        self.consecutive_foul_count = 0
        self.foul_detected = False
        self.foul_frame_number = None

    def check_frame(
        self,
        detections: List[Detection],
        frame_number: int,
    ) -> Tuple[bool, bool]:
        """
        Check if any shoe detection crosses the foul line in this frame.

        Returns:
            (frame_is_foul, confirmed_foul)
            - frame_is_foul: True if the shoe overlaps the line in THIS frame
            - confirmed_foul: True if foul is confirmed after consecutive frame validation
        """
        if not detections:
            self.consecutive_foul_count = 0
            return False, self.foul_detected

        frame_is_foul = False

        for det in detections:
            if self._check_single_detection(det):
                frame_is_foul = True
                break

        # Consecutive frame validation
        if frame_is_foul:
            self.consecutive_foul_count += 1
        else:
            self.consecutive_foul_count = 0

        # Confirm foul only after N consecutive frames
        if (
            self.consecutive_foul_count >= self.min_consecutive_frames
            and not self.foul_detected
        ):
            self.foul_detected = True
            self.foul_frame_number = frame_number - (self.min_consecutive_frames - 1)

        return frame_is_foul, self.foul_detected

    def _check_single_detection(self, det: Detection) -> bool:
        """
        Check if a single detection crosses the foul line.
        Uses segmentation mask if available, otherwise falls back to bounding box.
        """
        if det.mask is not None:
            return self._check_with_mask(det)
        else:
            return self._check_with_bbox(det)

    def _check_with_bbox(self, det: Detection) -> bool:
        """
        Check foul using bounding box.

        The foul line is defined as a line segment from (x1, y1) to (x2, y2).
        We check if any edge of the shoe bounding box crosses this line.

        For a horizontal line (most common setup), this simplifies to checking
        if the shoe's front edge extends past the line's Y coordinate.
        """
        bx1, by1, bx2, by2 = det.bbox
        lx1, ly1 = self.foul_line.x1, self.foul_line.y1
        lx2, ly2 = self.foul_line.x2, self.foul_line.y2

        # Check if the line is approximately horizontal
        if abs(ly2 - ly1) < 20:
            # Horizontal line: check if shoe extends past the line
            line_y = (ly1 + ly2) / 2
            # Shoe bottom must cross the line by at least tolerance pixels
            return by2 >= (line_y + self.tolerance_px)

        # Check if the line is approximately vertical
        elif abs(lx2 - lx1) < 20:
            # Vertical line: check if shoe extends past the line
            line_x = (lx1 + lx2) / 2
            return bx2 >= (line_x + self.tolerance_px)

        # General case: angled line — use line intersection
        else:
            return self._bbox_crosses_line(
                (bx1, by1, bx2, by2),
                (lx1, ly1, lx2, ly2),
            )

    def _check_with_mask(self, det: Detection) -> bool:
        """
        Check foul using the segmentation mask.
        This is more accurate because it checks the actual shoe contour
        rather than the rectangular bounding box.
        """
        mask = det.mask  # Binary mask, same size as frame

        # Create a line mask
        line_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.line(
            line_mask,
            (self.foul_line.x1, self.foul_line.y1),
            (self.foul_line.x2, self.foul_line.y2),
            255,
            thickness=max(self.tolerance_px * 2, 3),
        )

        # Check if any shoe pixels overlap the line region
        overlap = cv2.bitwise_and(mask * 255, line_mask)
        overlap_pixels = np.count_nonzero(overlap)

        return overlap_pixels > 0

    def _bbox_crosses_line(
        self,
        bbox: tuple,
        line: tuple,
    ) -> bool:
        """
        General line-box intersection check.
        Returns True if the bounding box extends past the foul line
        by at least the tolerance amount.
        """
        bx1, by1, bx2, by2 = bbox
        lx1, ly1, lx2, ly2 = line

        # Get the four corners of the bounding box
        corners = [
            (bx1, by1), (bx2, by1),
            (bx2, by2), (bx1, by2),
        ]

        # Compute signed distance of each corner from the line
        # Line equation: (y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1 = 0
        dx = lx2 - lx1
        dy = ly2 - ly1

        for cx, cy in corners:
            # Signed distance (positive = on the "foul" side of the line)
            dist = dy * cx - dx * cy + lx2 * ly1 - ly2 * lx1
            # Normalize
            length = np.sqrt(dx * dx + dy * dy)
            if length > 0:
                dist /= length

            if dist > self.tolerance_px:
                return True

        return False

    def get_shoe_mask_pixels_crossing_line(self, det: Detection) -> int:
        """
        Count the number of shoe mask pixels that are on the foul side of the line.
        Useful for measuring how far the shoe crossed.
        """
        if det.mask is None:
            return 0

        mask = det.mask
        h, w = mask.shape[:2]

        # Create a "foul zone" mask (everything past the line)
        foul_zone = np.zeros((h, w), dtype=np.uint8)
        pts = self._get_foul_zone_polygon(w, h)
        cv2.fillPoly(foul_zone, [pts], 255)

        # Count shoe pixels in the foul zone
        overlap = cv2.bitwise_and(mask * 255, foul_zone)
        return np.count_nonzero(overlap)

    def _get_foul_zone_polygon(self, img_w: int, img_h: int) -> np.ndarray:
        """
        Create a polygon representing the area BEYOND the foul line
        (the sand pit side — the foul zone).

        For a horizontal line at y=Y, the foul zone is everything below Y.
        """
        lx1, ly1 = self.foul_line.x1, self.foul_line.y1
        lx2, ly2 = self.foul_line.x2, self.foul_line.y2

        # Determine which side is the foul zone based on line orientation
        if abs(ly2 - ly1) < 20:
            # Horizontal line: foul zone is below the line
            y = max(ly1, ly2)
            return np.array([
                [0, y], [img_w, y],
                [img_w, img_h], [0, img_h],
            ], dtype=np.int32)
        elif abs(lx2 - lx1) < 20:
            # Vertical line: foul zone is to the right
            x = max(lx1, lx2)
            return np.array([
                [x, 0], [img_w, 0],
                [img_w, img_h], [x, img_h],
            ], dtype=np.int32)
        else:
            # Angled line: use the bottom-right quadrant
            return np.array([
                [lx1, ly1], [lx2, ly2],
                [img_w, img_h], [0, img_h],
            ], dtype=np.int32)
