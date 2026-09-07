"""
YOLOv8 Segmentation-based shoe/foot detector.
Handles model loading, inference, and mask extraction.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional

from utils import Detection


class ShoeDetector:
    """Detects shoes using a YOLOv8 segmentation model."""

    def __init__(
        self,
        weights_path: str,
        confidence: float = 0.5,
        iou: float = 0.45,
        device: str = "auto",
        imgsz: int = 640,
        class_name: str = "shoe",
    ):
        self.weights_path = weights_path
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.class_name = class_name
        self.model: Optional[YOLO] = None

        # Resolve device
        if device == "auto":
            self.device = None  # Ultralytics auto-selects
        else:
            self.device = device

    def load(self) -> bool:
        """Load the YOLO model from weights file."""
        try:
            self.model = YOLO(self.weights_path)
            print(f"[Detector] Model loaded: {self.weights_path}")

            # Print available classes
            if hasattr(self.model, "names") and self.model.names:
                print(f"[Detector] Classes: {self.model.names}")

            return True
        except Exception as e:
            print(f"[Detector] ERROR: Failed to load model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run segmentation inference on a frame and return shoe detections."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            task="segment",
        )

        detections = []

        for result in results:
            if result.masks is None or result.boxes is None:
                continue

            boxes = result.boxes
            masks = result.masks

            for i in range(len(boxes)):
                # Get class name
                cls_id = int(boxes.cls[i].item())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                # Filter by target class name (if model has multiple classes)
                # Accept if class_name matches or if model only has one class
                if self.class_name.lower() not in cls_name.lower() and len(result.names) > 1:
                    continue

                # Bounding box (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].item())

                # Segmentation mask
                mask = None
                if masks is not None and i < len(masks.data):
                    # masks.data is in the model's input resolution
                    # We need to resize it to the original frame resolution
                    mask_data = masks.data[i].cpu().numpy()
                    mask = cv2.resize(
                        mask_data,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    # Convert to binary mask
                    mask = (mask > 0.5).astype(np.uint8)

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        mask=mask,
                        class_name=cls_name,
                    )
                )

        return detections

    def detect_with_fallback(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect shoes with graceful fallback if segmentation masks are unavailable.
        Works with both YOLOv8-seg and YOLOv8-detect models.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            task="segment",
        )

        detections = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            has_masks = (
                hasattr(result, "masks")
                and result.masks is not None
                and len(result.masks.data) > 0
            )

            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                if self.class_name.lower() not in cls_name.lower() and len(result.names) > 1:
                    continue

                x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy()
                conf = float(result.boxes.conf[i].item())

                mask = None
                if has_masks and i < len(result.masks.data):
                    mask_data = result.masks.data[i].cpu().numpy()
                    mask = cv2.resize(
                        mask_data,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask = (mask > 0.5).astype(np.uint8)

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        mask=mask,
                        class_name=cls_name,
                    )
                )

        return detections
