# Long Jump Foul Detection System

AI-powered system that detects whether an athlete's shoe crosses the foul line during a long jump takeoff. Built with **YOLOv8 segmentation** for pixel-accurate shoe detection and **real-time foul line monitoring** with consecutive-frame validation to eliminate false positives.

## Features

- **Real-time detection** — webcam or video file input at 60+ FPS
- **YOLOv8 segmentation** — pixel-accurate shoe masks (not just bounding boxes)
- **Perspective-warped ROI** — user-defined take-off board region for focused analysis
- **Consecutive-frame foul validation** — configurable threshold to prevent single-frame glitches
- **Modern tkinter GUI** — dark-themed interface with ROI/foul-line drawing, real-time FPS, and foul event tracking
- **OpenCV pipeline** — alternative CLI-based detection with interactive setup
- **Streamlit web dashboard** — browser-based video upload and result viewing
- **Training script** — train your own YOLOv8-seg model on a custom shoe dataset
- **Output recording** — annotated videos and foul-frame screenshots saved automatically

## Project Structure

```
long-jump-foul-detection/
├── config.yaml              # All settings (model, camera, foul line, thresholds)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
├── src/
│   ├── gui.py               # Tkinter GUI — primary entry point
│   ├── main.py              # OpenCV CLI pipeline — alternative entry point
│   ├── camera.py            # Camera/video capture module
│   ├── detector.py          # YOLOv8 shoe segmentation detector
│   ├── foul_checker.py      # Foul line crossing logic (bbox + mask modes)
│   ├── calibration.py       # Interactive foul line calibration tool
│   └── utils.py             # Drawing, config loading, data classes, frame buffer
├── train/
│   └── train.py             # YOLOv8 segmentation training script
├── dashboard/
│   └── app.py               # Streamlit web dashboard
├── models/                  # Place your trained .pt model here
└── output/                  # Output videos and foul screenshots
```

## Quick Start



## 📥 Model Weights Setup (Important)

Due to GitHub's file size limits, the trained YOLOv8 model weights (`.pt` file) are too large to be included directly in the code repository. To make sure the project runs seamlessly, I have uploaded the model to the **Releases** section. 

To set up the model on your local machine, please follow these steps:

1. Go to the [Releases](../../releases) tab on the right side of this GitHub repository.
2. Download the model weight file (e.g., `shoe-seg.pt`) from the latest release assets.
3. Place the downloaded `.pt` file inside the `models/` directory of your cloned repository.

Your folder structure should look exactly like this before you run the code:

```text
AI-Based-Long-Jump-Foul-Detection/
├── models/
│   └── shoe-seg.pt    <-- Place the downloaded file here!


### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Requires:** Python 3.10+, OpenCV 4.8+, Ultralytics YOLOv8

### 2. Train Your Model

Place your labeled shoe/foot images in a `dataset/` folder following YOLO format, then train:

```bash
python train/train.py --epochs 200 --batch 16
```

After training, the best model is automatically copied to `models/shoe_seg.pt`.

You can also train on Google Colab with a free GPU — just copy `train/train.py` and the `dataset/` folder.

### 3. Run the GUI (Recommended)

```bash
python src/gui.py
```

1. Click **Upload Video** or **Live Camera**
2. Click **4 corners** of the take-off board area (green ROI)
3. Click **2 points** for the foul line (red)
4. Click **Start Detection** — AI analysis runs in real-time

### 4. Or Run via CLI

```bash
# Interactive setup (click ROI + foul line, then detection runs)
python src/main.py --source 0

# Using a video file
python src/main.py --source your_video.mp4

# Headless mode (no display window)
python src/main.py --source your_video.mp4 --no-display
```

### 5. Calibrate Foul Line (Optional)

```bash
python src/calibration.py --source your_video.mp4
```

Click two points on the foul line edge, press `c` to save to `config.yaml`.

### 6. Web Dashboard (Optional)

```bash
streamlit run dashboard/app.py
```

Upload videos through the browser and view annotated results.

## Configuration

All settings are in `config.yaml`:

| Setting | Description | Default |
|---------|-------------|---------|
| `model.weights` | Path to trained .pt model file | `models/shoe_seg.pt` |
| `model.confidence_threshold` | Detection confidence (0.0–1.0) | `0.5` |
| `model.iou_threshold` | Non-max suppression IoU | `0.45` |
| `model.device` | Inference device | `auto` |
| `camera.source` | Camera index (0=webcam) or video path | `0` |
| `foul_line.x1, y1, x2, y2` | Foul line pixel coordinates | `0,400 → 1280,400` |
| `foul_line.tolerance_px` | Pixels of tolerance margin | `5` |
| `detection.min_consecutive_frames` | Frames needed to confirm a foul | `2` |
| `output.save_video` | Save annotated output video | `true` |

## Controls

### GUI (`gui.py`)

| Action | Control |
|--------|---------|
| Upload video | **Upload Video** button |
| Start webcam | **Live Camera** button |
| Draw ROI | Click 4 corners on the canvas |
| Draw foul line | Click 2 points on the canvas |
| Start detection | **Start Detection** button |
| Reset | **Reset** button |

### CLI (`main.py`)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset for next jump attempt |

### Calibration (`calibration.py`)

| Key | Action |
|-----|--------|
| Click | Set foul line point (2 clicks needed) |
| `c` | Confirm and save line |
| `r` | Reset points |
| `q` | Quit without saving |

## How It Works

```
Camera Frame → ROI Perspective Warp → YOLOv8 Segmentation → Inverse Warp
                                                              ↓
                                              Foul Line Check (mask/bbox)
                                                              ↓
                                        Consecutive Frame Validation
                                                              ↓
                                                    FOUL / NO FOUL
```

1. **Capture** — Each frame is read from webcam or video file
2. **ROI warp** — The user-defined take-off board region is perspective-warped to a top-down view
3. **Detection** — YOLOv8-seg runs on the warped ROI and produces pixel-accurate shoe segmentation masks
4. **Inverse warp** — Detections are mapped back to original frame coordinates for display
5. **Foul check** — The shoe mask/bbox is compared against the calibrated foul line (supports horizontal, vertical, and angled lines)
6. **Validation** — A foul is only confirmed after N consecutive frames of overlap, eliminating single-frame false positives
7. **Output** — Results are displayed live with overlays; annotated videos and foul-frame screenshots are saved to `output/`

## Dataset Format (for training)

```
dataset/
├── images/
│   ├── train/     ← training images
│   └── val/       ← validation images
└── labels/
    ├── train/     ← YOLO seg labels (.txt)
    └── val/       ← YOLO seg labels (.txt)
```

Each label file: `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>` (normalized 0–1).

Recommended labeling tools: [Roboflow](https://roboflow.com), Label Studio, or CVAT.

## Requirements

- Python 3.10+
- OpenCV 4.8+
- Ultralytics YOLOv8
- A trained shoe segmentation model (.pt file)

## License

This project is open-source and available under the MIT License.
