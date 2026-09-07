"""
Calibration tool for setting the foul line position.

Usage:
    python src/calibration.py                          # Use webcam
    python src/calibration.py --source video.mp4       # Use video file
    python src/calibration.py --config config.yaml     # Specify config path

Instructions:
    1. A window will open showing the camera feed.
    2. Click TWO points on the image to define the foul line.
       - First click: start of the line (e.g., left edge of the board)
       - Second click: end of the line (e.g., right edge of the board)
    3. Press 'c' to confirm and save the line to config.yaml
    4. Press 'r' to reset and pick new points
    5. Press 'q' to quit without saving
"""

import cv2
import yaml
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera import Camera


class FoulLineCalibrator:
    """Interactive tool to set the foul line by clicking on the camera feed."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.points = []
        self.frame = None
        self.display_frame = None
        self.window_name = "Foul Line Calibration — Click 2 points, then press 'c' to save"

    def load_config(self) -> dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_line_to_config(self, x1, y1, x2, y2):
        """Save the foul line coordinates to the config file."""
        config = self.load_config()

        if "foul_line" not in config:
            config["foul_line"] = {}

        config["foul_line"]["x1"] = int(x1)
        config["foul_line"]["y1"] = int(y1)
        config["foul_line"]["x2"] = int(x2)
        config["foul_line"]["y2"] = int(y2)

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n[Calibration] Foul line saved to {self.config_path}:")
        print(f"  Point 1: ({x1}, {y1})")
        print(f"  Point 2: ({x2}, {y2})")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to set line endpoints."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                print(f"[Calibration] Point {len(self.points)} set: ({x}, {y})")

                # Draw the point
                cv2.circle(self.display_frame, (x, y), 8, (0, 255, 0), -1)

                if len(self.points) == 2:
                    # Draw the line
                    cv2.line(
                        self.display_frame,
                        self.points[0],
                        self.points[1],
                        (0, 0, 255),
                        3,
                    )
                    print("[Calibration] Line drawn. Press 'c' to save or 'r' to reset.")

    def draw_overlay(self, frame):
        """Draw current points and instructions on the frame."""
        h, w = frame.shape[:2]

        # Instructions
        instructions = [
            "Click 2 points to define the foul line",
            "Press 'c' to save | 'r' to reset | 'q' to quit",
        ]

        # Semi-transparent overlay bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        for i, text in enumerate(instructions):
            cv2.putText(
                frame, text, (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

        # Draw existing points
        for i, pt in enumerate(self.points):
            cv2.circle(frame, pt, 8, (0, 255, 0), -1)
            cv2.putText(
                frame, f"P{i+1}", (pt[0] + 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        # Draw line if both points set
        if len(self.points) == 2:
            cv2.line(frame, self.points[0], self.points[1], (0, 0, 255), 3)

        return frame

    def run(self, source=0):
        """Run the calibration tool."""
        camera = Camera(source=source)
        if not camera.open():
            print("[Calibration] ERROR: Could not open camera/video source.")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n[Calibration] Starting calibration tool...")
        print("[Calibration] Click two points on the foul line edge.")
        print("[Calibration] Press 'c' to confirm, 'r' to reset, 'q' to quit.\n")

        while True:
            success, frame = camera.read()
            if not success:
                # For video files, loop back to start
                if camera.is_video_file:
                    camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            self.display_frame = frame.copy()
            self.display_frame = self.draw_overlay(self.display_frame)

            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[Calibration] Quit without saving.")
                break

            elif key == ord("r"):
                self.points = []
                print("[Calibration] Reset. Click 2 new points.")

            elif key == ord("c"):
                if len(self.points) == 2:
                    self.save_line_to_config(
                        self.points[0][0], self.points[0][1],
                        self.points[1][0], self.points[1][1],
                    )
                    # Show saved line on frame
                    frame_with_line = frame.copy()
                    cv2.line(
                        frame_with_line,
                        self.points[0],
                        self.points[1],
                        (0, 255, 0),
                        4,
                    )
                    cv2.putText(
                        frame_with_line,
                        "FOUL LINE SAVED!",
                        (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3,
                    )
                    cv2.imshow(self.window_name, frame_with_line)
                    cv2.waitKey(2000)
                    break
                else:
                    print("[Calibration] Need 2 points before saving. Click on the image.")

        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Foul Line Calibration Tool")
    parser.add_argument(
        "--source", default=0,
        help="Camera index (0=webcam) or path to video file",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    # Convert source to int if it's a number
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # Keep as string (video file path)

    calibrator = FoulLineCalibrator(args.config)
    calibrator.run(source=source)


if __name__ == "__main__":
    main()
