"""Block type definitions and Block dataclass.

This module provides:
- BlockType: Standardized block type constants
- BlockTypeMapper: Maps detector-specific types to standardized types
- Block: Document block dataclass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bbox import BBox


class BlockType:
    """Standardized block types based on MinerU 2.5 VLM.

    This provides a canonical set of block types that all detectors
    should map to for consistent processing throughout the pipeline.
    """

    # Content
    TEXT = "text"
    TITLE = "title"

    # Figures
    IMAGE = "image"
    IMAGE_BODY = "image_body"
    IMAGE_CAPTION = "image_caption"
    IMAGE_FOOTNOTE = "image_footnote"

    # Tables
    TABLE = "table"
    TABLE_BODY = "table_body"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"

    # Equations
    EQUATION = "equation"  # Alias for interline_equation
    INTERLINE_EQUATION = "interline_equation"
    INLINE_EQUATION = "inline_equation"

    # Code
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    ALGORITHM = "algorithm"

    # Lists
    LIST = "list"

    # Page Elements
    HEADER = "header"  # Page header (not heading)
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    PAGE_FOOTNOTE = "page_footnote"

    # References
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    ASIDE_TEXT = "aside_text"
    INDEX = "index"

    # Special
    DISCARDED = "discarded"
    ABANDON = "abandon"  # MinerU DocLayoutYOLO legacy

    # Aliases for backward compatibility
    PLAIN_TEXT = "plain text"  # Legacy type
    FIGURE = "figure"  # Alias for image
    ISOLATE_FORMULA = "isolate_formula"  # MinerU DocLayoutYOLO legacy
    FORMULA_CAPTION = "formula_caption"  # MinerU DocLayoutYOLO legacy
    FIGURE_CAPTION = "figure_caption"  # MinerU DocLayoutYOLO legacy
    LIST_ITEM = "list_item"  # Project-specific type


class BlockTypeMapper:
    """Maps detector-specific block types to standardized types."""

    DOCLAYOUT_YOLO_MAP: dict[str, str] = {
        "title": BlockType.TITLE,
        "plain text": BlockType.TEXT,
        "text": BlockType.TEXT,
        "figure": BlockType.IMAGE,
        "image": BlockType.IMAGE,
        "table": BlockType.TABLE,
        "equation": BlockType.INTERLINE_EQUATION,
        "list": BlockType.LIST,
        "list_item": BlockType.LIST,
    }

    MINERU_DOCLAYOUT_YOLO_MAP: dict[str, str] = {
        "title": BlockType.TITLE,
        "plain text": BlockType.TEXT,
        "abandon": BlockType.DISCARDED,
        "figure": BlockType.IMAGE,
        "figure_caption": BlockType.IMAGE_CAPTION,
        "table": BlockType.TABLE,
        "table_caption": BlockType.TABLE_CAPTION,
        "table_footnote": BlockType.TABLE_FOOTNOTE,
        "isolate_formula": BlockType.INTERLINE_EQUATION,
        "formula_caption": BlockType.IMAGE_CAPTION,
    }

    MINERU_VLM_MAP: dict[str, str] = {
        "text": BlockType.TEXT,
        "title": BlockType.TITLE,
        "image": BlockType.IMAGE,
        "image_body": BlockType.IMAGE_BODY,
        "image_caption": BlockType.IMAGE_CAPTION,
        "image_footnote": BlockType.IMAGE_FOOTNOTE,
        "table": BlockType.TABLE,
        "table_body": BlockType.TABLE_BODY,
        "table_caption": BlockType.TABLE_CAPTION,
        "table_footnote": BlockType.TABLE_FOOTNOTE,
        "interline_equation": BlockType.INTERLINE_EQUATION,
        "inline_equation": BlockType.INLINE_EQUATION,
        "code": BlockType.CODE,
        "code_body": BlockType.CODE_BODY,
        "code_caption": BlockType.CODE_CAPTION,
        "algorithm": BlockType.ALGORITHM,
        "list": BlockType.LIST,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
        "page_number": BlockType.PAGE_NUMBER,
        "page_footnote": BlockType.PAGE_FOOTNOTE,
        "ref_text": BlockType.REF_TEXT,
        "phonetic": BlockType.PHONETIC,
        "aside_text": BlockType.ASIDE_TEXT,
        "index": BlockType.INDEX,
        "discarded": BlockType.DISCARDED,
    }

    OLMOCR_VLM_MAP: dict[str, str] = {
        "text": BlockType.TEXT,
    }

    PADDLEOCR_DOCLAYOUT_V2_MAP: dict[str, str] = {
        "doc_title": BlockType.TITLE,
        "paragraph_title": BlockType.TITLE,
        "text": BlockType.TEXT,
        "sidebar_text": BlockType.ASIDE_TEXT,
        "page_number": BlockType.PAGE_NUMBER,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
        "header_image": BlockType.HEADER,
        "footer_image": BlockType.FOOTER,
        "abstract": BlockType.TEXT,
        "contents": BlockType.TEXT,
        "reference": BlockType.REF_TEXT,
        "reference_content": BlockType.REF_TEXT,
        "footnote": BlockType.PAGE_FOOTNOTE,
        "formula": BlockType.INTERLINE_EQUATION,
        "formula_number": BlockType.INTERLINE_EQUATION,
        "algorithm": BlockType.ALGORITHM,
        "image": BlockType.IMAGE,
        "table": BlockType.TABLE,
        "table_title": BlockType.TABLE_CAPTION,
        "chart": BlockType.IMAGE,
        "chart_title": BlockType.IMAGE_CAPTION,
        "seal": BlockType.IMAGE,
    }

    @classmethod
    def map_type(cls, block_type: str, detector_name: str) -> str:
        """Map a detector-specific type to standardized type."""
        mapping_dict: dict[str, str] | None = None

        if detector_name == "doclayout-yolo":
            mapping_dict = cls.DOCLAYOUT_YOLO_MAP
        elif detector_name == "mineru-doclayout-yolo":
            mapping_dict = cls.MINERU_DOCLAYOUT_YOLO_MAP
        elif detector_name == "mineru-vlm":
            mapping_dict = cls.MINERU_VLM_MAP
        elif detector_name == "olmocr-vlm":
            mapping_dict = cls.OLMOCR_VLM_MAP
        elif detector_name == "paddleocr-doclayout-v2":
            mapping_dict = cls.PADDLEOCR_DOCLAYOUT_V2_MAP

        if mapping_dict:
            return mapping_dict.get(block_type.lower(), block_type)

        return block_type


@dataclass
class Block:
    """Document block with bounding box.

    This dataclass provides type safety and better IDE support for
    block data throughout the pipeline.

    Core fields:
    - type: Block type string. Should use standardized types from BlockType class
            (e.g., "text", "title", "image", "table", "code", etc.)
            Detectors may use detector-specific types which should be mapped using
            BlockTypeMapper.map_type() before further processing.
    - bbox: BBox object (required, always present)
    - detection_confidence: Detection confidence (0.0 to 1.0), optional

    Optional fields added by various pipeline stages:
    - order: Reading order rank (Added by sorters)
    - column_index: Column index (Added by multi-column sorters)
    - text: Recognized text (Added by recognizers)
    - corrected_text: VLM-corrected text (Added by text correction)
    - correction_ratio: Block-level correction ratio (0.0 = no change, 1.0 = completely different)
      (Added by text correction)
    - corrected_by: Model name that performed text correction (Added by text correction)
    - image_path: Path to extracted image file for image/figure blocks (Added by image extraction)
    - description: VLM-generated description for image/figure/chart blocks (Added by recognition)
    - source: Which detector/sorter produced this block (internal use only, not serialized)
    - index: Internal index (MinerU VLM)

    Note:
        Block types should ideally use the standardized types defined in BlockType.
        Use BlockTypeMapper.map_type(block.type, detector_name) to normalize
        detector-specific types to standardized types.
    """

    # Core fields (always present)
    type: str  # Ideally from BlockType constants
    bbox: BBox  # Required, no longer optional
    detection_confidence: float | None = None  # Optional, not all detectors provide confidence

    # Optional fields added by pipeline stages
    order: int | None = None  # Reading order rank (renamed from reading_order_rank)
    column_index: int | None = None

    # Text content
    text: str | None = None
    corrected_text: str | None = None
    correction_ratio: float | None = None  # Block-level correction ratio (0.0 = no change, 1.0 = completely different)
    corrected_by: str | None = None  # Model name that performed text correction

    # Image/Figure block fields
    image_path: str | None = None  # Path to extracted image file for image/figure/chart blocks
    description: str | None = None  # VLM-generated description for image/figure/chart blocks

    # Metadata (internal use, not serialized to JSON)
    source: str | None = None  # "doclayout-yolo", "mineru-vlm", etc.
    index: int | None = None  # MinerU VLM internal index

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict.

        Field order:
            order → type → xywh → detection_confidence → column_index → text → corrected_text
            → correction_ratio → corrected_by → image_path → description
        This order prioritizes reading order first, then type, then position, then content.

        Bbox is serialized as xywh list [x, y, width, height] for human readability.
        The 'source' field is excluded from serialization as it's for internal use only.

        Returns:
            Dict with xywh bbox format

        Example:
            >>> block = Block(type="text", bbox=BBox(100, 50, 300, 200), detection_confidence=0.95, order=0)
            >>> block.to_dict()
            {'order': 0, 'type': 'text', 'xywh': [100, 50, 200, 150], 'detection_confidence': 0.95}
        """
        result: dict[str, Any] = {}

        # Add fields in desired order (Python 3.7+ preserves insertion order)
        # 1. Reading order (most important for document reconstruction)
        if self.order is not None:
            result["order"] = self.order

        # 2. Type (what kind of block)
        result["type"] = self.type

        # 3. Position (where it is)
        result["xywh"] = self.bbox.to_xywh_list()

        # 4. Detection confidence (optional)
        if self.detection_confidence is not None:
            result["detection_confidence"] = self.detection_confidence

        # 5. Column index (optional, for multi-column layouts)
        if self.column_index is not None:
            result["column_index"] = self.column_index

        # 6. Text content (extracted)
        if self.text is not None:
            result["text"] = self.text

        # 7. Corrected text (VLM-corrected)
        if self.corrected_text is not None:
            result["corrected_text"] = self.corrected_text

        # 8. Correction ratio (block-level)
        if self.correction_ratio is not None:
            result["correction_ratio"] = self.correction_ratio

        # 9. Corrected by (model name that performed correction)
        if self.corrected_by is not None:
            result["corrected_by"] = self.corrected_by

        # 10. Image/figure block fields
        if self.image_path is not None:
            result["image_path"] = self.image_path

        if self.description is not None:
            result["description"] = self.description

        # Note: 'source' and 'index' are intentionally excluded (internal metadata)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        """Create Block from dict.

        Args:
            data: Dictionary with block data (must have "xywh" field)

        Returns:
            Block object

        Example:
            >>> data = {"type": "text", "xywh": [100, 50, 200, 150], "detection_confidence": 0.95}
            >>> block = Block.from_dict(data)
            >>> block.bbox
            BBox(x0=100, y0=50, x1=300, y1=200)
        """
        data = data.copy()  # Don't modify original

        # Parse xywh field (required)
        if "xywh" not in data:
            raise ValueError("Block dict must have 'xywh' field")

        xywh_value = data.pop("xywh")
        if not isinstance(xywh_value, (list, tuple)):
            raise ValueError(f"Invalid xywh type: {type(xywh_value)}")

        bbox = BBox.from_xywh(*xywh_value[:4])

        # Build Block with remaining fields
        return cls(
            type=data.get("type", "unknown"),
            bbox=bbox,
            detection_confidence=data.get("detection_confidence"),
            order=data.get("order"),
            column_index=data.get("column_index"),
            text=data.get("text"),
            corrected_text=data.get("corrected_text"),
            correction_ratio=data.get("correction_ratio"),
            corrected_by=data.get("corrected_by"),
            image_path=data.get("image_path"),
            description=data.get("description"),
            source=data.get("source"),
            index=data.get("index"),
        )
