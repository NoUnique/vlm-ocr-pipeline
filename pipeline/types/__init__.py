"""Unified type definitions for the VLM OCR pipeline.

This module provides:
- BBox: Integer-based bounding box (internal: xyxy, JSON: xywh)
- Block, BlockType, BlockTypeMapper: Document block types
- Page: Single page processing result
- Document: Multi-page document result
- Detector, Sorter, Recognizer, Renderer: Component interfaces
- StageTimingInfo, PipelineResult: Pipeline result types
"""

from .auxiliary import AuxiliaryInfo, ColumnInfo, ColumnLayout, TextSpan
from .bbox import RGB_IMAGE_NDIM, BBox
from .block import Block, BlockType, BlockTypeMapper
from .document import Document
from .external import PyMuPDFPage, PyMuPDFRect
from .interfaces import Detector, Recognizer, Renderer, Sorter
from .page import Page
from .result import PipelineResult, StageTimingInfo, blocks_to_olmocr_anchor_text

__all__ = [
    # Constants
    "RGB_IMAGE_NDIM",
    # Auxiliary types
    "TextSpan",
    "AuxiliaryInfo",
    "ColumnInfo",
    "ColumnLayout",
    # External library protocols
    "PyMuPDFRect",
    "PyMuPDFPage",
    # Block types
    "BlockType",
    "BlockTypeMapper",
    # Core data models
    "BBox",
    "Block",
    "Page",
    "Document",
    # Component interfaces
    "Detector",
    "Sorter",
    "Recognizer",
    "Renderer",
    # Result types
    "StageTimingInfo",
    "PipelineResult",
    "blocks_to_olmocr_anchor_text",
]
