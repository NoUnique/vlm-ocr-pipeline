"""Pipeline stages for document processing.

This module provides the stage-based architecture for the OCR pipeline.
Each stage performs a specific task and can be composed into pipelines.

Stage inheritance:
    BaseStage → DetectionStage, OrderingStage, RecognitionStage, ...

Usage:
    >>> from pipeline.stages import DetectionStage, BaseStage
    >>> detector = create_detector("doclayout-yolo")
    >>> stage = DetectionStage(detector)
    >>> blocks = stage.process(image)
"""

from __future__ import annotations

from pipeline.stages.base import BaseStage, StageError, StageResult
from pipeline.stages.block_correction_stage import BlockCorrectionStage
from pipeline.stages.detection_stage import DetectionStage
from pipeline.stages.input_stage import InputResult, InputStage
from pipeline.stages.ordering_stage import OrderingStage
from pipeline.stages.output_stage import OutputStage
from pipeline.stages.page_correction_stage import PageCorrectionResult, PageCorrectionStage
from pipeline.stages.recognition_stage import RecognitionStage
from pipeline.stages.rendering_stage import RenderingStage

__all__ = [
    # Base classes
    "BaseStage",
    "StageError",
    "StageResult",
    # Result types
    "InputResult",
    "PageCorrectionResult",
    # Stages
    "InputStage",
    "DetectionStage",
    "OrderingStage",
    "RecognitionStage",
    "BlockCorrectionStage",
    "RenderingStage",
    "PageCorrectionStage",
    "OutputStage",
]
