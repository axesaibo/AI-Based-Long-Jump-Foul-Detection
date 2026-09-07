"""
Main pipeline for the Long Jump Foul Detection System.
Interactive UI with startup menu, ROI selection, and foul line selection.

Usage:
    python src/main.py                          # Run with interactive menu
    python src/main.py --source video.mp4       # Run on a video file (skip menu)
    python src/main.py --config my_config.yaml  # Use custom config
    python src/main.py --no-display             # Headless mode (no OpenCV window)
"""

import cv2
import argparse
import time
import os
import sys
import glob
import numpy as np
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera import Camera
from detector import ShoeDetector
from foul_checker import FoulChecker
from utils import (
    load_config,
    get_foul_line_from_config,
    draw_detections,
    draw_foul_alert,
    draw_frame_counter,
    draw_foul_line,
    FrameBuffer,
    create_video_writer,
    FoulEvent,
    FoulLine,
)


# ===================================================================
#  HELPERS
# ===================================================================

def _close_window(win_name):
    """Safely destroy an OpenCV window (OpenCV 5.x sometimes throws on this)."""
    try:
        cv2.destroyWindow(win_name)
    except cv2.error:
        pass


# ===================================================================
#  VIDEO FILE HELPERS
# ===================================================================

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def find_local_videos(search_dir=None):
    """
    Scan the project directory (and one level of subfolders) for video files.
    Returns a list of absolute paths.
    """
    if search_dir is None:
        search_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = []
    for entry in os.scandir(search_dir):
        if entry.is_file() and Path(entry.path).suffix.lower() in VIDEO_EXTS:
            found.append(os.path.abspath(entry.path))
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file() and Path(sub.path).suffix.lower() in VIDEO_EXTS:
                    found.append(os.path.abspath(sub.path))
    return sorted(set(found))


def pick_video_dialog():
    """Open a native OS file-chooser dialog and return the selected path."""
    try:
        root = tk.Tk()
        root.withdraw()                        # hide the main window
        root.attributes("-topmost", True)       # bring dialog to front
        path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if path and os.path.isfile(path):
            return path
    except Exception as exc:
        print(f"  [Menu] File dialog error: {exc}")
    return None


# ===================================================================
#  STARTUP MENU
# ===================================================================

def show_startup_menu():
    """
    Display a visual menu to upload a video or use the live camera.
    Auto-detects video files already in the project folder.
    Returns (source, ok).
    """
    local_videos = find_local_videos()

    # --- Build the menu image dynamically based on available videos ----
    n_local = len(local_videos)
    rows = 2 + n_local + 1            # title + videos + camera + browse + footer
    row_h = 50
    top_pad = 80
    menu_h = top_pad + rows * row_h + 60
    menu_w = 820
    menu = np.full((menu_h, menu_w, 3), (35, 35, 35), dtype=np.uint8)
    w = menu_w

    # Title
    cv2.putText(menu, "Long Jump Foul Detection System",
                (w // 2 - 280, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2)

    y = top_pad

    # -- Local video files --
    if n_local > 0:
        cv2.putText(menu, "Videos found in project folder:",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (180, 180, 180), 1)
        y += 10
        for idx, vpath in enumerate(local_videos, start=1):
            y += row_h
            fname = os.path.basename(vpath)
            # Button
            cv2.rectangle(menu, (40, y - 30), (w - 40, y + 10),
                          (55, 55, 55), -1)
            cv2.rectangle(menu, (40, y - 30), (w - 40, y + 10),
                          (0, 200, 0), 2)
            label = f"[{idx}]  {fname}"
            cv2.putText(menu, label,
                        (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 1)
        y += row_h
    else:
        cv2.putText(menu, "(no video files found in project folder)",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 120, 120), 1)
        y += row_h

    # -- Browse for video --
    browse_y = y + row_h
    cv2.rectangle(menu, (40, browse_y - 30), (w - 40, browse_y + 10),
                  (55, 55, 55), -1)
    cv2.rectangle(menu, (40, browse_y - 30), (w - 40, browse_y + 10),
                  (0, 200, 0), 2)
    cv2.putText(menu, "[B]  Browse for a video file...",
                (60, browse_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 1)

    # -- Live camera --
    cam_y = browse_y + row_h
    cv2.rectangle(menu, (40, cam_y - 30), (w - 40, cam_y + 10),
                  (55, 55, 55), -1)
    cv2.rectangle(menu, (40, cam_y - 30), (w - 40, cam_y + 10),
                  (0, 200, 0), 2)
    cv2.putText(menu, "[C]  Live Camera (Webcam)",
                (60, cam_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 1)

    # Footer
    cv2.putText(menu, "Press 'q' to quit",
                (w // 2 - 70, menu_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    win = "Long Jump Foul Detection - Main Menu"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, menu)

    source = None
    while True:
        key = cv2.waitKey(100) & 0xFF

        # Number keys 1-9 for local videos
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < n_local:
                source = local_videos[idx]
                print(f"  [Menu] Selected: {os.path.basename(source)}")
                break

        elif key == ord("b"):
            _close_window(win)
            picked = pick_video_dialog()
            if picked:
                source = picked
                print(f"  [Menu] Selected: {os.path.basename(source)}")
                break
            # If dialog cancelled, re-show menu
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(win, menu)

        elif key == ord("c"):
            source = 0
            print("  [Menu] Using live camera.")
            break

        elif key == ord("q"):
            break

    _close_window(win)
    return source, source is not None




# ===================================================================
#  ROI SELECTION  (4 points → green rectangle)
# ===================================================================

def select_roi(frame):
    """
    Let the user click 4 corners of the takeoff-board area.
    A minimum-area bounding rectangle (green) is fitted automatically.
    Returns (roi_quad, order_corners(roi_quad), warped_w, warped_h) or None.
    """
    h, w = frame.shape[:2]
    points = []
    display = frame.copy()
    win = "Step 1/2 — Select ROI (click 4 corners of the take-off area)"

    default_quad = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
    )

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"  [ROI] Corner {len(points)}: ({x}, {y})")
            cv2.circle(display, (x, y), 7, (0, 255, 0), -1)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("\n" + "=" * 60)
    print("  STEP 1/2 — REGION OF INTEREST (ROI)")
    print("  Click the 4 corners of the take-off board area.")
    print("  A green rectangle will be fitted around your clicks.")
    print("  [c] confirm   [r] reset   [q] skip (full frame)")
    print("=" * 60)

    result = None

    while True:
        overlay = display.copy()

        # Instruction bar
        bar = overlay.copy()
        cv2.rectangle(bar, (0, 0), (w, 55), (0, 0, 0), -1)
        overlay = cv2.addWeighted(bar, 0.7, overlay, 0.3, 0)
        cv2.putText(overlay, "Click 4 corners of the take-off area",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        cv2.putText(overlay,
                    "[c] confirm  [r] reset  [q] skip (full frame)",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (200, 200, 200), 1)

        # Draw clicked corners
        for i, pt in enumerate(points):
            cv2.circle(overlay, pt, 7, (0, 255, 0), -1)
            cv2.putText(overlay, f"C{i + 1}",
                        (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # When 4 corners are clicked, fit and draw the rectangle
        if len(points) == 4:
            pts_arr = np.array(points, dtype=np.float32)
            rect = cv2.minAreaRect(pts_arr)
            box = cv2.boxPoints(rect)
            box_int = np.int32(box)

            # Green rectangle
            cv2.polylines(overlay, [box_int], True, (0, 255, 0), 3)

            # Semi-transparent green fill
            fill = overlay.copy()
            cv2.fillPoly(fill, [box_int], (0, 255, 0))
            overlay = cv2.addWeighted(fill, 0.15, overlay, 0.85, 0)

            # Re-draw points on top
            for i, pt in enumerate(points):
                cv2.circle(overlay, pt, 7, (0, 255, 0), -1)
                cv2.putText(overlay, f"C{i + 1}",
                            (pt[0] + 10, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            cv2.putText(overlay, "Press 'c' to confirm this ROI",
                        (w // 2 - 180, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow(win, overlay)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            print("  [ROI] Skipped — using full frame.")
            ww = int(cv2.norm(default_quad[0] - default_quad[1]))
            wh = int(cv2.norm(default_quad[0] - default_quad[3]))
            result = (default_quad, order_corners(default_quad), ww, wh)
            break

        elif key == ord("r"):
            points.clear()
            display = frame.copy()
            print("  [ROI] Reset. Click 4 new corners.")

        elif key == ord("c"):
            if len(points) == 4:
                pts_arr = np.array(points, dtype=np.float32)
                rect = cv2.minAreaRect(pts_arr)
                box = cv2.boxPoints(rect)

                ww = int(cv2.norm(box[0] - box[1]))
                wh = int(cv2.norm(box[0] - box[3]))
                if ww < 20 or wh < 20:
                    print("  [ROI] Region too small — pick 4 wider corners.")
                    continue

                ordered = order_corners(box)
                print(f"  [ROI] Confirmed  ({ww} x {wh} px)")

                # Flash confirmation
                conf = frame.copy()
                box_i = np.int32(box)
                cv2.polylines(conf, [box_i], True, (0, 255, 0), 4)
                cv2.fillPoly(conf, [box_i], (0, 255, 0))
                conf = cv2.addWeighted(conf, 0.25, frame.copy(), 0.75, 0)
                cv2.polylines(conf, [box_i], True, (0, 255, 0), 4)
                cv2.putText(conf, "ROI SET!",
                            (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
                cv2.imshow(win, conf)
                cv2.waitKey(1500)

                result = (box.astype(np.float32), ordered, ww, wh)
                break
            else:
                print(f"  [ROI] Need 4 corners ({len(points)}/4 clicked).")

    _close_window(win)
    return result


# ===================================================================
#  GEOMETRY HELPERS
# ===================================================================

def order_corners(box):
    """
    Order 4 box corners as: top-left, top-right, bottom-right, bottom-left.
    Works with any quadrilateral (sorts by centroid angles).
    """
    pts = np.float32(box).reshape(-1, 2)
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)

    # arctan2 order: roughly  -π…π  →  bottom-left, top-left, top-right, bottom-right
    # We want: TL, TR, BR, BL
    tl_idx = order[1]
    tr_idx = order[2]
    br_idx = order[3]
    bl_idx = order[0]
    return np.float32([pts[tl_idx], pts[tr_idx], pts[br_idx], pts[bl_idx]])


def compute_warp(src_ordered, dst_w, dst_h):
    """Return (warp_matrix, inverse_warp_matrix)."""
    dst = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    M = cv2.getPerspectiveTransform(src_ordered, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src_ordered)
    return M, M_inv


def warp_point(pt, M):
    """Map a single (x, y) point through a perspective transform."""
    p = np.float32([[pt[0], pt[1]]]).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(p, M)
    r = out[0][0]
    return (float(r[0]), float(r[1]))


def inv_warp_detections(detections, M_inv, frame_hw):
    """Map detections from warped ROI space back to original frame coords."""
    from utils import Detection as Det
    frame_h, frame_w = frame_hw
    mapped = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        tl = warp_point((x1, y1), M_inv)
        br = warp_point((x2, y2), M_inv)
        new_mask = None
        if d.mask is not None:
            new_mask = cv2.warpPerspective(
                d.mask, M_inv, (frame_w, frame_h))
        mapped.append(Det(
            bbox=(tl[0], tl[1], br[0], br[1]),
            confidence=d.confidence,
            mask=new_mask,
            class_name=d.class_name,
        ))
    return mapped


# ===================================================================
#  FOUL LINE SELECTION  (2 points → red line)
# ===================================================================

def select_foul_line(frame, default_fl):
    """
    Let the user click 2 points to define the foul line (drawn in red).
    Returns a FoulLine, or the default if the user quits.
    """
    h, w = frame.shape[:2]
    points = []
    display = frame.copy()
    win = "Step 2/2 — Select Foul Line (click 2 points inside the ROI)"

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            print(f"  [FoulLine] Point {len(points)}: ({x}, {y})")
            cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 0, 255), 3)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("\n" + "=" * 60)
    print("  STEP 2/2 — FOUL LINE")
    print("  Click 2 points to draw the foul line (red).")
    print("  [c] confirm   [r] reset   [q] skip (config default)")
    print("=" * 60)

    foul_line = default_fl

    while True:
        overlay = display.copy()

        # Instruction bar
        bar = overlay.copy()
        cv2.rectangle(bar, (0, 0), (w, 55), (0, 0, 0), -1)
        overlay = cv2.addWeighted(bar, 0.7, display, 0.3, 0)
        cv2.putText(overlay, "Click 2 points to define the foul line",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1)
        cv2.putText(overlay,
                    "[c] confirm  [r] reset  [q] skip (config default)",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (200, 200, 200), 1)

        # Draw points
        for i, pt in enumerate(points):
            cv2.circle(overlay, pt, 7, (0, 0, 255), -1)
            cv2.putText(overlay, f"P{i + 1}",
                        (pt[0] + 10, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Draw red line
        if len(points) == 2:
            cv2.line(overlay, points[0], points[1], (0, 0, 255), 3)
            cv2.putText(overlay, "Press 'c' to confirm",
                        (w // 2 - 130, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(win, overlay)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            print("  [FoulLine] Skipped — using config default.")
            break

        elif key == ord("r"):
            points.clear()
            display = frame.copy()
            print("  [FoulLine] Reset. Click 2 new points.")

        elif key == ord("c"):
            if len(points) == 2:
                foul_line = FoulLine(
                    x1=points[0][0], y1=points[0][1],
                    x2=points[1][0], y2=points[1][1],
                    tolerance_px=default_fl.tolerance_px,
                    color=default_fl.color,
                    thickness=default_fl.thickness,
                )
                print(f"  [FoulLine] Confirmed: "
                      f"({foul_line.x1},{foul_line.y1}) -> "
                      f"({foul_line.x2},{foul_line.y2})")

                # Flash confirmation
                conf = frame.copy()
                cv2.line(conf, points[0], points[1], (0, 0, 255), 4)
                cv2.putText(conf, "FOUL LINE SET!",
                            (w // 2 - 140, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow(win, conf)
                cv2.waitKey(1500)
                break
            else:
                print("  [FoulLine] Need 2 points. Click on the frame.")

    _close_window(win)
    return foul_line


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Long Jump Foul Detection System")
    parser.add_argument(
        "--source", default=None,
        help="Camera index (0=webcam) or path to video file.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without OpenCV display window (headless mode)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save output video",
    )
    return parser.parse_args()


# ===================================================================
#  MAIN PIPELINE
# ===================================================================

def main():
    args = parse_args()

    # -------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------
    config = load_config(args.config)
    model_cfg = config.get("model", {})
    camera_cfg = config.get("camera", {})
    detection_cfg = config.get("detection", {})
    output_cfg = config.get("output", {})

    show_display = output_cfg.get("show_display", True) and not args.no_display
    save_video = output_cfg.get("save_video", True) and not args.no_save

    print("=" * 60)
    print("  Long Jump Foul Detection System")
    print("=" * 60)

    # 1. Load model
    detector = ShoeDetector(
        weights_path=model_cfg.get("weights", "models/shoe_seg.pt"),
        confidence=model_cfg.get("confidence_threshold", 0.5),
        iou=model_cfg.get("iou_threshold", 0.45),
        device=model_cfg.get("device", "auto"),
        imgsz=model_cfg.get("imgsz", 640),
        class_name=detection_cfg.get("class_name", "shoe"),
    )
    if not detector.load():
        print("\nERROR: Model could not be loaded.")
        print("Make sure your trained .pt file is in the models/ folder.")
        return

    # -------------------------------------------------------------------
    # 2. Choose source  (interactive menu or --source flag)
    # -------------------------------------------------------------------
    if args.source is not None:
        source = args.source
        try:
            source = int(source)
        except ValueError:
            pass
    else:
        source, ok = show_startup_menu()
        if not ok:
            print("Exiting.")
            return

    # -------------------------------------------------------------------
    # 3. Read first frame for interactive setup
    # -------------------------------------------------------------------
    cap_preview = cv2.VideoCapture(source)
    if not cap_preview.isOpened():
        print(f"\nERROR: Could not open video source: {source}")
        return
    ret, first_frame = cap_preview.read()
    cap_preview.release()
    if not ret or first_frame is None:
        print("\nERROR: Could not read the first frame.")
        return

    # -------------------------------------------------------------------
    # 4. Select ROI  (4 clicks → green rectangle)
    # -------------------------------------------------------------------
    roi_result = select_roi(first_frame)
    if roi_result is None:
        print("  [ROI] No ROI selected — exiting.")
        return

    roi_quad, roi_ordered, roi_w, roi_h = roi_result
    M_warp, M_inv_warp = compute_warp(roi_ordered, roi_w, roi_h)
    print(f"[ROI] Warp target: {roi_w} x {roi_h}")

    # -------------------------------------------------------------------
    # 5. Select foul line  (2 clicks → red line, inside ROI)
    # -------------------------------------------------------------------
    default_fl = get_foul_line_from_config(config)
    foul_line = select_foul_line(first_frame, default_fl)

    # Map foul line into warped-ROI coordinate space for detection
    fl_roi_p1 = warp_point((foul_line.x1, foul_line.y1), M_warp)
    fl_roi_p2 = warp_point((foul_line.x2, foul_line.y2), M_warp)
    foul_line_roi = FoulLine(
        x1=int(fl_roi_p1[0]), y1=int(fl_roi_p1[1]),
        x2=int(fl_roi_p2[0]), y2=int(fl_roi_p2[1]),
        tolerance_px=foul_line.tolerance_px,
        color=foul_line.color,
        thickness=foul_line.thickness,
    )
    print(f"[FoulLine] (original) "
          f"({foul_line.x1},{foul_line.y1}) -> "
          f"({foul_line.x2},{foul_line.y2})")
    print(f"[FoulLine] Tolerance: {foul_line.tolerance_px}px")

    # -------------------------------------------------------------------
    # 6. Setup foul checker  (operates in ROI-warped space)
    # -------------------------------------------------------------------
    foul_checker = FoulChecker(
        foul_line=foul_line_roi,
        min_consecutive_frames=detection_cfg.get("min_consecutive_frames", 2),
        tolerance_px=foul_line.tolerance_px,
    )

    # -------------------------------------------------------------------
    # 7. Open camera for the main loop
    # -------------------------------------------------------------------
    camera = Camera(
        source=source,
        width=camera_cfg.get("width", 1280),
        height=camera_cfg.get("height", 720),
        fps=camera_cfg.get("fps", 60),
    )
    if not camera.open():
        print(f"\nERROR: Could not open video source: {source}")
        return

    # -------------------------------------------------------------------
    # 8. Setup output
    # -------------------------------------------------------------------
    video_writer = None
    output_dir = output_cfg.get("output_dir", "output")
    if save_video:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir,
                                   f"foul_check_{timestamp}.mp4")
        video_writer = create_video_writer(
            output_path, camera.fps, camera.width, camera.height,
        )
        print(f"[Output] Saving video to: {output_path}")

    # 9. Frame buffer for replay
    replay_size = output_cfg.get("replay_buffer_frames", 60)
    frame_buffer = FrameBuffer(max_size=replay_size)

    # -------------------------------------------------------------------
    # Main detection loop
    # -------------------------------------------------------------------
    print("\n[Main] Starting foul detection...")
    print("[Main] Press 'q' to quit, 'r' to reset for new jump\n")

    fps_timer = time.time()
    fps_count = 0
    current_fps = 0.0
    foul_events = []
    frame_num = 0

    for frame_num, frame in camera.frames():
        # --- Warp ROI → run detection only inside the selected area ---
        roi_frame = cv2.warpPerspective(frame, M_warp, (roi_w, roi_h))
        detections_roi = detector.detect_with_fallback(roi_frame)

        # Map detections back to original frame coordinates
        detections = inv_warp_detections(
            detections_roi, M_inv_warp, frame.shape[:2])

        # --- Foul check (in ROI-warped space) ---
        frame_is_foul, confirmed_foul = foul_checker.check_frame(
            detections_roi, frame_num,
        )

        # --- Visualization ---
        annotated = draw_detections(
            frame.copy(), detections, foul_line, is_foul=frame_is_foul,
        )

        # Draw ROI outline (green)
        roi_pts = np.int32(roi_quad)
        cv2.polylines(annotated, [roi_pts], True, (0, 255, 0), 2)

        # Re-draw foul line on top (red) so it's always visible
        cv2.line(
            annotated,
            (foul_line.x1, foul_line.y1),
            (foul_line.x2, foul_line.y2),
            foul_line.color, foul_line.thickness,
        )

        annotated = draw_foul_alert(annotated, confirmed_foul)
        annotated = draw_frame_counter(annotated, frame_num, current_fps)

        # --- Frame buffer ---
        frame_buffer.add(annotated, frame_num)

        # --- Save foul event ---
        if confirmed_foul and foul_checker.foul_frame_number == frame_num:
            event = FoulEvent(
                frame_number=frame_num,
                timestamp=time.time(),
                foul_frame=annotated.copy(),
                replay_frames=[f.copy() for _, f in frame_buffer.get_all()],
            )
            foul_events.append(event)
            print(f"[FOUL] Detected at frame {frame_num}!")
            if save_video:
                foul_img = os.path.join(
                    output_dir, f"foul_frame_{frame_num}.jpg")
                cv2.imwrite(foul_img, annotated)
                print(f"[FOUL] Screenshot saved: {foul_img}")

        # --- Display ---
        if show_display:
            cv2.imshow("Foul Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                foul_checker.reset()
                frame_buffer.clear()
                print("[Main] Reset for new jump attempt.")

        # --- Save frame to video ---
        if video_writer is not None:
            video_writer.write(annotated)

        # --- FPS calculation ---
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            current_fps = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------
    print(f"\n[Main] Processed {frame_num + 1} frames")
    print(f"[Main] Total foul events: {len(foul_events)}")

    if video_writer is not None:
        video_writer.release()
        print(f"[Output] Video saved to: {output_path}")

    camera.release()
    if show_display:
        cv2.destroyAllWindows()

    # Final verdict
    if foul_events:
        print("\n" + "=" * 60)
        print("  RESULT: FOUL DETECTED")
        print(f"  Foul at frame: {foul_events[0].frame_number}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  RESULT: NO FOUL")
        print("=" * 60)


if __name__ == "__main__":
    main()
