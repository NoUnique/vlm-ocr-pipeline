"""DocLayout-YOLO detector implementation.

DocLayout-YOLO BBox Format:
- Output: [x, y, width, height]
- Origin: Top-Left (0, 0)
- Coordinate Order: Left-Top + Size
- Example: [100, 50, 200, 150] means rectangle from (100,50) with width=200, height=150
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ...types import BBox, Block, BlockTypeMapper, Detector

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class DocLayoutYOLODetector(Detector):
    """Detector using DocLayout-YOLO model with adaptive batch processing.

    This detector wraps the existing DocLayoutYOLO model and provides
    output in the unified Block format. Supports automatic batch size
    calibration for optimal GPU utilization.

    BBox Format: [x, y, width, height] - Top-Left origin

    Features:
        - Single image detection: detect(image)
        - Batch processing: detect_batch(images)
        - Auto batch size calibration: auto_batch_size=True
        - Manual batch size: batch_size=16
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        auto_batch_size: bool = False,
        batch_size: int | None = None,
        target_memory_fraction: float = 0.85,
    ):
        """Initialize DocLayout-YOLO detector.

        Args:
            model_path: Path to DocLayout-YOLO model weights
            confidence_threshold: Confidence threshold for detections
            auto_batch_size: Auto-calibrate optimal batch size (overrides batch_size)
            batch_size: Manual batch size for batch processing (default: 1)
            target_memory_fraction: Target GPU memory usage for auto-calibration (0.0-1.0)
        """
        from models.doclayout_yolo import DocLayoutYOLO

        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.auto_batch_size = auto_batch_size
        self.target_memory_fraction = target_memory_fraction
        self.model = DocLayoutYOLO(model_path=model_path)

        # Batch size configuration
        if auto_batch_size:
            # Will be calibrated on first batch processing
            self._batch_size = None
            self._calibrated = False
            logger.info("DocLayout-YOLO detector initialized (auto batch size)")
        else:
            self._batch_size = batch_size or 1
            self._calibrated = True
            logger.info("DocLayout-YOLO detector initialized (batch_size=%d)", self._batch_size)

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks in image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected blocks in unified format

        Example:
            >>> detector = DocLayoutYOLODetector()
            >>> blocks = detector.detect(image)
            >>> blocks[0].type
            'plain text'
            >>> blocks[0].bbox.to_xywh()  # (x, y, w, h)
            (100, 50, 200, 150)
            >>> blocks[0].bbox  # BBox object
            BBox(x0=100, y0=50, x1=300, y1=200)
        """
        raw_results = self.model.predict(image, conf=self.confidence_threshold)
        logger.debug("Detected %d blocks with DocLayout-YOLO", len(raw_results))

        return [self._to_block(r) for r in raw_results]

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Block]]:
        """Detect layout blocks in multiple images (batch processing).

        Auto-calibrates batch size on first call if auto_batch_size=True.
        Processes images in batches for optimal GPU utilization.

        Args:
            images: List of input images as numpy arrays (H, W, C)

        Returns:
            List of block lists, one per image

        Example:
            >>> detector = DocLayoutYOLODetector(auto_batch_size=True)
            >>> results = detector.detect_batch([img1, img2, img3])
            >>> len(results)
            3
            >>> len(results[0])  # Number of blocks in first image
            15
        """
        if not images:
            return []

        # Auto-calibrate on first batch if needed
        if self.auto_batch_size and not self._calibrated:
            self._calibrate_batch_size(images[0])

        # Process in batches
        batch_size = self._batch_size or 1
        all_results = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # Process each image in batch individually
            # (DocLayoutYOLO model interface processes one at a time)
            for image in batch:
                blocks = self.detect(image)
                all_results.append(blocks)

        return all_results

    def _calibrate_batch_size(self, sample_image: np.ndarray) -> None:
        """Calibrate optimal batch size using sample image.

        Args:
            sample_image: Sample image for calibration
        """
        from pipeline.optimization import calibrate_batch_size

        logger.info("Calibrating optimal batch size for DocLayout-YOLO...")

        # Inference function for calibration
        def run_inference(batch_size: int):
            # Create batch of sample images
            batch = [sample_image] * batch_size

            # Run detection on batch
            for image in batch:
                _ = self.model.predict(image, conf=self.confidence_threshold)

            return batch_size

        # Run calibration
        try:
            optimal_batch_size = calibrate_batch_size(
                inference_fn=run_inference,
                model_name="doclayout-yolo",
                input_shape=sample_image.shape,
                target_memory_fraction=self.target_memory_fraction,
            )

            self._batch_size = optimal_batch_size
            self._calibrated = True

            logger.info("Calibration complete: batch_size=%d", optimal_batch_size)

        except Exception as e:
            logger.warning("Calibration failed: %s. Using batch_size=1", e)
            self._batch_size = 1
            self._calibrated = True

    def get_batch_size(self) -> int:
        """Get current batch size (after calibration if auto_batch_size=True).

        Returns:
            Current batch size
        """
        return self._batch_size or 1

    def _to_block(self, raw_data: dict) -> Block:
        """Convert DocLayout-YOLO result to unified Block format.

        Args:
            raw_data: {"type": str, "coords": [x, y, w, h], "confidence": float}

        Returns:
            Unified Block dataclass instance with BBox and standardized type
        """
        coords = raw_data["coords"]
        bbox = BBox.from_xywh(*coords[:4])

        # Map detector-specific type to standardized type
        original_type = raw_data["type"]
        standardized_type = BlockTypeMapper.map_type(original_type, "doclayout-yolo")

        return Block(
            type=standardized_type,
            bbox=bbox,
            detection_confidence=float(raw_data["confidence"]),
            source="doclayout-yolo",
        )
