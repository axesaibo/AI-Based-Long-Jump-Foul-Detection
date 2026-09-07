"""
Training script for YOLOv8 segmentation model on shoe/foot dataset.

Usage:
    python train/train.py                                    # Use default settings
    python train/train.py --epochs 300 --batch 8             # Custom training params
    python train/train.py --data dataset.yaml --model yolov8s-seg.pt  # Custom model/data

Dataset structure (YOLO format):
    dataset/
    ├── images/
    │   ├── train/     ← training images
    │   └── val/       ← validation images
    └── labels/
        ├── train/     ← training labels (.txt, YOLO seg format)
        └── val/       ← validation labels (.txt, YOLO seg format)

Each label file contains segmentation annotations:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    (coordinates are normalized 0–1 relative to image dimensions)

You can label your dataset using:
    - Roboflow (https://roboflow.com) — easiest, exports YOLO-seg format
    - Label Studio — open source
    - CVAT — open source
"""

import argparse
import os
import sys
import yaml
from pathlib import Path


def create_dataset_yaml(data_dir: str, class_names: list) -> str:
    """
    Create a dataset.yaml file for YOLO training.
    Returns the path to the created yaml file.
    """
    data_dir = Path(data_dir).resolve()

    dataset_config = {
        "path": str(data_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = data_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    print(f"[Train] Dataset config saved: {yaml_path}")
    print(f"[Train] Classes: {class_names}")
    print(f"[Train] Train dir: {data_dir / 'images' / 'train'}")
    print(f"[Train] Val dir:   {data_dir / 'images' / 'val'}")

    # Count images
    train_imgs = list((data_dir / "images" / "train").glob("*.*"))
    val_imgs = list((data_dir / "images" / "val").glob("*.*"))
    print(f"[Train] Training images: {len(train_imgs)}")
    print(f"[Train] Validation images: {len(val_imgs)}")

    if len(train_imgs) == 0:
        print("\n[WARNING] No training images found!")
        print("  Place your training images in: dataset/images/train/")
        print("  Place your label files in:     dataset/labels/train/")

    if len(val_imgs) == 0:
        print("\n[WARNING] No validation images found!")
        print("  Place your validation images in: dataset/images/val/")
        print("  Place your label files in:       dataset/labels/val/")

    return str(yaml_path)


def validate_dataset(data_dir: str) -> bool:
    """Check if the dataset directory has the expected structure."""
    data_dir = Path(data_dir)

    required_dirs = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ]

    all_ok = True
    for d in required_dirs:
        path = data_dir / d
        if not path.exists():
            print(f"[Train] Missing directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
            print(f"[Train] Created empty directory: {path}")
            all_ok = False

    return all_ok


def train(args):
    """Run YOLOv8 segmentation training."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Train] ERROR: ultralytics not installed.")
        print("  Run: pip install ultralytics")
        return

    # Validate dataset structure
    print("=" * 60)
    print("  YOLOv8 Segmentation Training")
    print("=" * 60)

    data_dir = Path(args.data_dir).resolve()
    validate_dataset(str(data_dir))

    # Create dataset.yaml
    class_names = [c.strip() for c in args.classes.split(",")]
    dataset_yaml = create_dataset_yaml(str(data_dir), class_names)

    # Load model
    print(f"\n[Train] Loading base model: {args.model}")
    model = YOLO(args.model)

    # Train
    print(f"\n[Train] Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")
    print(f"  Workers: {args.workers}")
    print()

    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Segmentation specific
        mask_ratio=4,
        overlap_mask=True,
    )

    # Export best model
    best_model_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_model_path):
        # Copy to models/ folder
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)

        import shutil
        dest = models_dir / "shoe_seg.pt"
        shutil.copy2(best_model_path, dest)
        print(f"\n[Train] Best model copied to: {dest}")
        print(f"[Train] You can now run: python src/main.py")
    else:
        print(f"\n[Train] Training complete. Check: {args.project}/{args.name}/weights/")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Segmentation Model")

    # Data
    parser.add_argument(
        "--data-dir", default="dataset",
        help="Path to dataset directory (default: dataset/)",
    )
    parser.add_argument(
        "--classes", default="shoe",
        help="Comma-separated class names (default: 'shoe')",
    )

    # Model
    parser.add_argument(
        "--model", default="yolov8n-seg.pt",
        help="Base model (yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or '0' for GPU 0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)

    # Output
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="shoe_seg")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
