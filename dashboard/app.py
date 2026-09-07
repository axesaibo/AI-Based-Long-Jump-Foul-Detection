"""
Streamlit web dashboard for the Long Jump Foul Detection System.

Usage:
    streamlit run dashboard/app.py
    streamlit run dashboard/app.py --server.port 8501

Features:
    - Upload a video and run foul detection
    - View results with annotated frames
    - Download output video and foul screenshots
"""

import streamlit as st
import cv2
import os
import sys
import tempfile
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detector import ShoeDetector
from foul_checker import FoulChecker
from utils import (
    load_config,
    get_foul_line_from_config,
    draw_detections,
    draw_foul_alert,
    draw_frame_counter,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Long Jump Foul Detection",
    page_icon="🏃",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

config_path = st.sidebar.text_input("Config file", value="config.yaml")
confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.05)
tolerance = st.sidebar.slider("Tolerance (pixels)", 0, 30, 5)
min_frames = st.sidebar.slider("Min consecutive frames", 1, 10, 2)

st.sidebar.divider()
st.sidebar.markdown("### Upload Video")
uploaded_file = st.sidebar.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov", "mkv"],
)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("🏃 Long Jump Foul Detection System")
st.markdown("---")

if uploaded_file is None:
    st.info("👈 Upload a video file from the sidebar to start foul detection.")

    st.markdown("""
    ### How to use:
    1. **Train your model** — Place your trained `.pt` file in `models/shoe_seg.pt`
    2. **Calibrate the foul line** — Run `python src/calibration.py` to set the line
    3. **Upload a video** — Use the sidebar to upload a jump video
    4. **Review results** — The system will detect fouls and show annotated frames

    ### Quick start (command line):
    ```bash
    # 1. Calibrate the foul line
    python src/calibration.py --source your_video.mp4

    # 2. Run detection on a video
    python src/main.py --source your_video.mp4

    # 3. Run with webcam
    python src/main.py --source 0
    ```
    """)
    st.stop()


# ---------------------------------------------------------------------------
# Process uploaded video
# ---------------------------------------------------------------------------

# Save uploaded file to temp
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_file.read())
video_path = tfile.name

# Load config
try:
    config = load_config(config_path)
except FileNotFoundError:
    st.error(f"Config file not found: {config_path}")
    st.stop()

model_cfg = config.get("model", {})
detection_cfg = config.get("detection", {})

# Initialize components
with st.spinner("Loading model..."):
    detector = ShoeDetector(
        weights_path=model_cfg.get("weights", "models/shoe_seg.pt"),
        confidence=confidence,
        iou=model_cfg.get("iou_threshold", 0.45),
        device=model_cfg.get("device", "auto"),
        imgsz=model_cfg.get("imgsz", 640),
        class_name=detection_cfg.get("class_name", "shoe"),
    )
    if not detector.load():
        st.error("Failed to load model. Check that your .pt file exists in models/")
        st.stop()

foul_line = get_foul_line_from_config(config)
foul_checker = FoulChecker(
    foul_line=foul_line,
    min_consecutive_frames=min_frames,
    tolerance_px=tolerance,
)

# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------
st.markdown("### Processing Video...")

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

progress_bar = st.progress(0)
status_text = st.empty()
frame_display = st.empty()

foul_frames = []
results_data = []
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    detections = detector.detect_with_fallback(frame)

    # Check foul
    frame_is_foul, confirmed_foul = foul_checker.check_frame(detections, frame_num)

    # Annotate
    annotated = draw_detections(frame.copy(), detections, foul_line, is_foul=frame_is_foul)
    annotated = draw_foul_alert(annotated, confirmed_foul)
    annotated = draw_frame_counter(annotated, frame_num, video_fps)

    # Store results
    results_data.append({
        "frame": frame_num,
        "detections": len(detections),
        "frame_foul": frame_is_foul,
        "confirmed_foul": confirmed_foul,
    })

    if confirmed_foul and foul_checker.foul_frame_number == frame_num:
        foul_frames.append({
            "frame_number": frame_num,
            "timestamp_sec": frame_num / video_fps,
            "image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        })

    # Update UI every 5 frames (performance)
    if frame_num % 5 == 0:
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_display.image(rgb_frame, channels="RGB", use_container_width=True)
        progress = frame_num / max(total_frames, 1)
        progress_bar.progress(progress)
        status_text.text(
            f"Frame {frame_num}/{total_frames} | "
            f"Detections: {len(detections)} | "
            f"{'FOUL!' if confirmed_foul else 'No foul'}"
        )

    frame_num += 1

cap.release()
progress_bar.progress(1.0)
status_text.text(f"✅ Processed {frame_num} frames")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Results")

col1, col2, col3 = st.columns(3)

total_detections = sum(r["detections"] for r in results_data)
foul_detected = any(r["confirmed_foul"] for r in results_data)

col1.metric("Total Frames", frame_num)
col2.metric("Total Detections", total_detections)
col3.metric("Verdict", "FOUL" if foul_detected else "NO FOUL",
            delta="Foul" if foul_detected else "Clean",
            delta_color="inverse" if foul_detected else "normal")


# ---------------------------------------------------------------------------
# Foul frames
# ---------------------------------------------------------------------------
if foul_frames:
    st.markdown("---")
    st.markdown("### 🚨 Foul Frames")

    for i, ff in enumerate(foul_frames):
        st.markdown(f"**Foul #{i+1}** — Frame {ff['frame_number']} ({ff['timestamp_sec']:.2f}s)")
        st.image(ff["image"], use_container_width=True)
else:
    st.success("No foul detected in this video.")


# ---------------------------------------------------------------------------
# Detection timeline
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Detection Timeline")

try:
    import pandas as pd

    df = pd.DataFrame(results_data)
    # Downsample for chart performance
    if len(df) > 500:
        df = df.iloc[::max(1, len(df) // 500)]

    st.line_chart(df.set_index("frame")["detections"], height=200)
    st.caption("Shoe detections per frame")
except ImportError:
    st.info("Install pandas for timeline charts: pip install pandas")


# Cleanup temp file
try:
    os.unlink(video_path)
except:
    pass
