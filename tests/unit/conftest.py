"""Pytest fixtures specific to unit tests.

Unit tests should be fast and isolated. These fixtures
ensure tests don't require actual models or external services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from pipeline.types import BBox, Block


@pytest.fixture
def unit_test_blocks() -> list[Block]:
    """Create minimal blocks for unit testing.

    Returns:
        List of simple Block objects
    """
    return [
        Block(
            type="text",
            bbox=BBox(0, 0, 100, 50),
            detection_confidence=0.9,
        ),
    ]


@pytest.fixture
def mock_pipeline_config() -> Mock:
    """Create a mock PipelineConfig for testing.

    Returns:
        Mock PipelineConfig object
    """
    config = Mock()
    config.detector = "paddleocr-doclayout-v2"
    config.sorter = "paddleocr-doclayout-v2"
    config.recognizer = "paddleocr-vl"
    config.detector_backend = None
    config.sorter_backend = None
    config.recognizer_backend = "pytorch"
    config.resolved_detector_backend = "paddle"
    config.resolved_sorter_backend = "paddle"
    config.resolved_recognizer_backend = "pytorch"
    config.dpi = 200
    config.detection_dpi = 150
    config.recognition_dpi = 300
    config.use_dual_resolution = False
    config.confidence_threshold = 0.5
    config.use_cache = False
    config.cache_dir = Path(".cache")
    config.output_dir = Path("output")
    config.temp_dir = Path(".tmp")
    config.renderer = "markdown"
    config.gemini_tier = "free"
    config.use_async = False
    config.auto_batch_size = False
    config.batch_size = None
    config.target_memory_fraction = 0.85
    config.detector_model_path = None
    config.sorter_model_path = None
    return config


@pytest.fixture
def small_test_image() -> np.ndarray:
    """Create a small test image (50x50) for fast tests.

    Returns:
        Small numpy array image
    """
    return np.zeros((50, 50, 3), dtype=np.uint8)

