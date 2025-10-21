"""PaddleOCR PP-DocLayoutV2 layout analysis implementation.

PP-DocLayoutV2 is a unified layout analysis model that performs both:
1. Layout detection (identifying document elements)
2. Reading order restoration (sorting elements in correct reading order)

Based on PP-DocLayout_plus-L (RT-DETR-L) cascaded with a lightweight pointer network,
PP-DocLayoutV2 provides high-precision layout localization and reading order recovery.

Model Specifications:
- mAP(0.5): 81.4%
- Model Size: 203.8 MB
- Supports 25 layout element categories

Categories (from PaddleOCR docs):
- doc_title, paragraph_title, text, vertical_text, page_number
- abstract, contents, reference, reference_content, footnote
- header, footer, header_image, footer_image
- algorithm, inline_formula, display_formula, formula_number
- image, table, figure_title (figure/table/chart titles)
- seal, chart, aside_text

BBox Format: [xmin, ymin, xmax, ymax] (xyxy)
Output: Boxes are already sorted in reading order by the pointer network!
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ....types import BBox, Block, Detector

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class PPDocLayoutV2Detector(Detector):
    """Layout analysis detector using PaddleOCR PP-DocLayoutV2.

    PP-DocLayoutV2 combines layout detection with reading order restoration:
    - Detection: RT-DETR-L based PP-DocLayout_plus-L (81.4 mAP)
    - Ordering: Lightweight pointer network (6 Transformer layers)

    Key Features:
    - Detects 25 document element categories
    - Automatically sorts elements in reading order
    - Trained on diverse datasets (papers, magazines, PPTs, contracts, etc.)
    - Supports Chinese, English, Japanese, and vertical text documents

    Note: The output boxes are already sorted in reading order, so no
    additional sorting step is needed when using this detector.

    BBox Format: [xmin, ymin, xmax, ymax] - Top-Left to Bottom-Right corners
    """

    # Type mapping from PP-DocLayoutV2 labels to standardized BlockType
    _TYPE_MAP = {
        # Titles and text
        "doc_title": "title",
        "paragraph_title": "title",
        "text": "text",
        "vertical_text": "text",
        "aside_text": "aside_text",

        # Page elements
        "page_number": "page_number",
        "header": "header",
        "footer": "footer",
        "header_image": "header",
        "footer_image": "footer",

        # Structural elements
        "abstract": "text",
        "contents": "text",
        "reference": "ref_text",
        "reference_content": "ref_text",
        "footnote": "page_footnote",

        # Math and formulas
        "inline_formula": "inline_equation",
        "display_formula": "interline_equation",
        "formula_number": "interline_equation",
        "algorithm": "algorithm",

        # Visual elements
        "image": "image",
        "table": "table",
        "chart": "image",
        "seal": "image",

        # Captions (unified as figure_title in PaddleOCR)
        "figure_title": "image_caption",  # Includes figure/table/chart captions
    }

    def __init__(
        self,
        model_name: str | None = None,
        model_dir: str | Path | None = None,
        threshold: float = 0.5,
        img_size: int | None = None,
        layout_nms: bool = True,
        layout_unclip_ratio: float | None = None,
        layout_merge_bboxes_mode: str | None = None,
        device: str = "cpu",
    ):
        """Initialize PP-DocLayoutV2 layout analysis model.

        Args:
            model_name: Model name (should be "PP-DocLayoutV2")
            model_dir: Path to custom model directory
            threshold: Confidence threshold for detections (default: 0.5)
            img_size: Input image size for model (default: None, uses model default)
            layout_nms: Whether to use NMS (default: True, recommended)
            layout_unclip_ratio: Unclip ratio for layout boxes (default: None)
            layout_merge_bboxes_mode: Mode for merging overlapping boxes (default: None)
            device: Device to run model on ("cpu" or "gpu", default: "cpu")

        Example:
            >>> detector = PPDocLayoutV2Detector()
            >>> blocks = detector.detect(image)
            >>> # Blocks are already sorted in reading order!
            >>> for block in blocks:
            ...     print(f"{block.order}: {block.type}")
        """
        # Lazy import to avoid loading PaddleOCR unless needed
        from paddleocr import LayoutDetection  # noqa: PLC0415  # type: ignore[import-untyped]

        self.model_name = model_name or "PP-DocLayoutV2"
        self.model_dir = Path(model_dir) if model_dir else None
        self.threshold = threshold
        self.img_size = img_size
        self.layout_nms = layout_nms
        self.layout_unclip_ratio = layout_unclip_ratio
        self.layout_merge_bboxes_mode = layout_merge_bboxes_mode
        self.device = device

        # Initialize PaddleOCR LayoutDetection with PP-DocLayoutV2
        init_kwargs = {"model_name": self.model_name}
        if model_dir:
            init_kwargs["model_dir"] = str(model_dir)
        if device:
            init_kwargs["device"] = device

        self.model = LayoutDetection(**init_kwargs)

        logger.info(
            "PP-DocLayoutV2 layout analysis model initialized (threshold: %.2f)",
            self.threshold,
        )

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout elements and restore reading order.

        PP-DocLayoutV2 performs two tasks in one forward pass:
        1. Detects layout elements (text, images, tables, etc.)
        2. Sorts them in correct reading order using pointer network

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected blocks ALREADY SORTED in reading order.
            Each block has its 'order' field set based on the model output.

        Example:
            >>> detector = PPDocLayoutV2Detector()
            >>> blocks = detector.detect(image)
            >>> # Blocks are automatically sorted!
            >>> blocks[0].order  # 0 (first in reading order)
            >>> blocks[1].order  # 1 (second in reading order)
        """
        # Build prediction parameters
        predict_kwargs = {"batch_size": 1, "layout_nms": self.layout_nms}
        if self.threshold is not None:
            predict_kwargs["threshold"] = self.threshold
        if self.img_size is not None:
            predict_kwargs["img_size"] = self.img_size
        if self.layout_unclip_ratio is not None:
            predict_kwargs["layout_unclip_ratio"] = self.layout_unclip_ratio
        if self.layout_merge_bboxes_mode is not None:
            predict_kwargs["layout_merge_bboxes_mode"] = self.layout_merge_bboxes_mode

        # Run detection + ordering
        raw_results = self.model.predict(image, **predict_kwargs)

        # Parse results
        if not raw_results or len(raw_results) == 0:
            logger.warning("PP-DocLayoutV2 returned no results")
            return []

        result_item = raw_results[0]
        boxes = result_item.get("boxes", [])

        logger.debug(
            "PP-DocLayoutV2 detected %d blocks (already sorted in reading order)",
            len(boxes),
        )

        # IMPORTANT: Boxes are already sorted by the pointer network!
        # We just need to assign order indices
        blocks = []
        for order_idx, box_data in enumerate(boxes):
            block = self._to_block(box_data, order_idx)
            blocks.append(block)

        return blocks

    def _to_block(self, box_data: dict, order: int) -> Block:
        """Convert PP-DocLayoutV2 result to unified Block format.

        Args:
            box_data: {
                "cls_id": int,
                "label": str,
                "score": float,
                "coordinate": [xmin, ymin, xmax, ymax]
            }
            order: Reading order index (provided by pointer network)

        Returns:
            Unified Block with bbox, type, and reading order
        """
        # Extract data
        bbox_coords = box_data["coordinate"]  # [xmin, ymin, xmax, ymax]
        label = box_data["label"]
        score = float(box_data["score"])

        # Create BBox from xyxy coordinates
        bbox = BBox.from_xyxy(*bbox_coords[:4])

        # Map label to standardized type
        standardized_type = self._TYPE_MAP.get(label.lower(), label)

        return Block(
            type=standardized_type or label,
            bbox=bbox,
            detection_confidence=score,
            order=order,  # Already sorted by PP-DocLayoutV2!
            source="paddleocr-doclayout-v2",
        )
