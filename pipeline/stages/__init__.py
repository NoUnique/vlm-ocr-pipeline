"""Pipeline stages for document processing."""

from __future__ import annotations

from pipeline.stages.block_correction_stage import BlockCorrectionStage
from pipeline.stages.detection_stage import DetectionStage
from pipeline.stages.input_stage import InputStage
from pipeline.stages.ordering_stage import OrderingStage
from pipeline.stages.output_stage import OutputStage
from pipeline.stages.page_correction_stage import PageCorrectionStage
from pipeline.stages.recognition_stage import RecognitionStage
from pipeline.stages.rendering_stage import RenderingStage

__all__ = [
    "InputStage",
    "DetectionStage",
    "OrderingStage",
    "RecognitionStage",
    "BlockCorrectionStage",
    "RenderingStage",
    "PageCorrectionStage",
    "OutputStage",
]
