import logging
from pathlib import Path

import numpy as np
import torch
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class DocLayoutYOLO:
    """DocLayout-YOLO model wrapper class with pathlib support and multi-GPU support"""

    def __init__(self, model_path: str | Path | None = None, use_multi_gpu: bool = True) -> None:
        """
        Initialize the DocLayout-YOLO model

        Args:
            model_path: Local model file path. If not provided,
                       the pre-trained model will be loaded from Hugging Face Hub.
            use_multi_gpu: Whether to use multiple GPUs if available (default: True)
        """
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.use_multi_gpu = use_multi_gpu
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.gpu_count > 1 and use_multi_gpu:
            logger.info("Multi-GPU mode enabled: will use %d GPUs", self.gpu_count)
            # YOLO supports multi-GPU via comma-separated device IDs
            self.device = ",".join(str(i) for i in range(self.gpu_count))
            logger.info("Multi-GPU device string: %s", self.device)
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if use_multi_gpu and self.gpu_count <= 1:
                logger.warning("Multi-GPU requested but only %d GPU available", self.gpu_count)

        self.init_model()

    def init_model(self) -> bool:
        """Initialize the model"""
        try:
            if self.model_path and self.model_path.exists():
                logger.info("Loading local model file: %s", self.model_path)
                self.model = YOLOv10(str(self.model_path))
            else:
                logger.info("Loading pre-trained model from Hugging Face")
                filepath = hf_hub_download(
                    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                    filename="doclayout_yolo_docstructbench_imgsz1024.pt",
                )
                self.model = YOLOv10(filepath)

            logger.info("DocLayout-YOLO model loaded successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize DocLayout-YOLO model: %s", e)
            try:
                from ultralytics import YOLO

                if self.model_path and self.model_path.exists():
                    self.model = YOLO(str(self.model_path))
                else:
                    self.model = YOLO("yolov8n.pt")
                logger.info("Successfully loaded ultralytics YOLO model as alternative")
                return True
            except Exception as e2:
                logger.error("Alternative initialization failed: %s", e2)
                self.model = None
                return False

    def predict(
        self,
        image_input: str | Path | np.ndarray,
        imgsz: int = 1024,
        conf: float = 0.25,
        device: str | None = None,
    ) -> list:
        """
        Perform layout prediction on the image.

        Args:
            image_input: Path to the image file or numpy array
            imgsz: Input image size
            conf: Confidence threshold
            device: Device to use (if None, automatically selected)

        Returns:
            List of prediction results
        """
        if self.model is None:
            logger.error("The model is not initialized")
            return []

        if isinstance(image_input, np.ndarray):
            # Input is numpy array, use directly
            source = image_input
        else:
            # Input is path, validate it exists
            image_path = Path(image_input)
            if not image_path.exists():
                logger.error("Image file does not exist: %s", image_path)
                return []
            source = str(image_path)

        if device is None:
            device = self.device

        try:
            results = self.model.predict(source=source, imgsz=imgsz, conf=conf, device=device)

            # Parse results into regions format
            regions = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, "boxes") and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    class_names = result.names

                    for box, cls_id, confidence in zip(boxes, classes, confs, strict=False):
                        if confidence >= conf:
                            x1, y1, x2, y2 = map(int, box)
                            cls_name = class_names[int(cls_id)]

                            regions.append(
                                {
                                    "type": cls_name,
                                    "coords": [x1, y1, x2 - x1, y2 - y1],
                                    "confidence": float(confidence),
                                }
                            )

            return regions
        except Exception as e:
            logger.error("Prediction failed: %s", e)
            return []
