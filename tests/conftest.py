"""Pytest configuration and shared fixtures for VLM-OCR-Pipeline tests.

This module provides:
- Common fixtures for all tests (sample_image, sample_blocks, etc.)
- Mock fixtures for components (mock_detector, mock_recognizer, etc.)
- Test configuration and path setup
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pytest

# Register anyio pytest plugin for async test support
# This enables @pytest.mark.anyio decorator and anyio_backends config option
pytest_plugins = ("anyio",)

# Ensure project root is importable when running tests via uv or python -m pytest
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from pipeline.types import Block


# ==================== Sample Data Fixtures ====================


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image (600x800 RGB).

    Returns:
        Numpy array representing a blank white image
    """
    return np.ones((600, 800, 3), dtype=np.uint8) * 255


@pytest.fixture
def sample_image_small() -> np.ndarray:
    """Create a small sample test image (100x100 RGB).

    Returns:
        Numpy array representing a blank black image
    """
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_blocks() -> list[Block]:
    """Create sample blocks for testing.

    Returns:
        List of Block objects with various types
    """
    from pipeline.types import BBox, Block

    return [
        Block(
            type="title",
            bbox=BBox(100, 50, 400, 100),
            detection_confidence=0.95,
            order=0,
        ),
        Block(
            type="text",
            bbox=BBox(100, 120, 400, 300),
            detection_confidence=0.90,
            order=1,
        ),
        Block(
            type="table",
            bbox=BBox(100, 320, 400, 500),
            detection_confidence=0.85,
            order=2,
        ),
    ]


@pytest.fixture
def sample_blocks_with_text() -> list[Block]:
    """Create sample blocks with text content for testing.

    Returns:
        List of Block objects with text populated
    """
    from pipeline.types import BBox, Block

    return [
        Block(
            type="title",
            bbox=BBox(100, 50, 400, 100),
            detection_confidence=0.95,
            order=0,
            text="Sample Title",
            corrected_text="Sample Title",
        ),
        Block(
            type="text",
            bbox=BBox(100, 120, 400, 300),
            detection_confidence=0.90,
            order=1,
            text="This is sample body text.",
            corrected_text="This is sample body text.",
        ),
    ]


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a path for a sample PDF file.

    Note: This does not create an actual PDF, just a path.
    Use fixtures/sample.pdf for real PDF testing.

    Returns:
        Path to a (non-existent) PDF file
    """
    return tmp_path / "sample.pdf"


# ==================== Mock Component Fixtures ====================


@pytest.fixture
def mock_detector() -> Mock:
    """Create a mock detector.

    Returns:
        Mock object with detect method
    """
    from pipeline.types import BBox, Block

    detector = Mock()
    detector.name = "mock-detector"
    detector.detect.return_value = [
        Block(
            type="text",
            bbox=BBox(100, 100, 200, 200),
            detection_confidence=0.95,
            source="mock-detector",
        )
    ]
    return detector


@pytest.fixture
def mock_detector_empty() -> Mock:
    """Create a mock detector that returns no blocks.

    Returns:
        Mock object with detect method returning empty list
    """
    detector = Mock()
    detector.name = "mock-detector"
    detector.detect.return_value = []
    return detector


@pytest.fixture
def mock_sorter() -> Mock:
    """Create a mock sorter.

    Returns:
        Mock object with sort method
    """
    sorter = Mock()
    sorter.name = "mock-sorter"

    def mock_sort(blocks: list[Any], image: np.ndarray, **kwargs: Any) -> list[Any]:
        # Add order to blocks
        for i, block in enumerate(blocks):
            block.order = i
        return blocks

    sorter.sort.side_effect = mock_sort
    return sorter


@pytest.fixture
def mock_recognizer() -> Mock:
    """Create a mock recognizer.

    Returns:
        Mock object with process_blocks and correct_text methods
    """
    recognizer = Mock()
    recognizer.name = "mock-recognizer"
    recognizer.supports_correction = True

    def mock_process_blocks(image: np.ndarray, blocks: list[Any]) -> list[Any]:
        # Add text to blocks
        for block in blocks:
            block.text = f"Text for {block.type}"
        return list(blocks)

    def mock_correct_text(text: str) -> str:
        return text  # Return unchanged

    recognizer.process_blocks.side_effect = mock_process_blocks
    recognizer.correct_text.side_effect = mock_correct_text
    return recognizer


@pytest.fixture
def mock_recognizer_with_correction() -> Mock:
    """Create a mock recognizer that applies text correction.

    Returns:
        Mock object with text correction behavior
    """
    recognizer = Mock()
    recognizer.name = "mock-recognizer"
    recognizer.supports_correction = True

    def mock_correct_text(text: str) -> dict[str, Any]:
        return {
            "corrected_text": f"[Corrected] {text}",
            "correction_ratio": 0.1,
        }

    recognizer.correct_text.side_effect = mock_correct_text
    return recognizer


# ==================== Configuration Fixtures ====================


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Create a sample pipeline configuration dict.

    Returns:
        Configuration dictionary
    """
    return {
        "detector": "paddleocr-doclayout-v2",
        "recognizer": "paddleocr-vl",
        "dpi": 200,
        "detection_dpi": 150,
        "recognition_dpi": 300,
        "use_dual_resolution": False,
        "confidence_threshold": 0.5,
        "use_cache": False,
        "renderer": "markdown",
    }


@pytest.fixture
def pipeline_config():
    """Create a PipelineConfig for testing.

    Returns:
        Validated PipelineConfig instance
    """
    from pipeline.config import PipelineConfig

    config = PipelineConfig(
        detector="paddleocr-doclayout-v2",
        recognizer="paddleocr-vl",
        use_cache=False,
    )
    config.validate()
    return config


# ==================== Directory Fixtures ====================


@pytest.fixture
def isolated_temp_dir(tmp_path: Path) -> Path:
    """Create an isolated temp directory for each test.

    Returns:
        Path to isolated temporary directory
    """
    test_dir = tmp_path / "test_output"
    test_dir.mkdir()
    return test_dir


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a cache directory for testing.

    Returns:
        Path to cache directory
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create an output directory for testing.

    Returns:
        Path to output directory
    """
    output = tmp_path / "output"
    output.mkdir()
    return output


# ==================== Fixtures Directory ====================


@pytest.fixture
def fixtures_dir() -> Path:
    """Get path to fixtures directory.

    Returns:
        Path to tests/fixtures directory
    """
    return PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture
def sample_fixture_pdf(fixtures_dir: Path) -> Path | None:
    """Get path to sample PDF in fixtures directory.

    Returns:
        Path to sample PDF if it exists, None otherwise
    """
    pdf_path = fixtures_dir / "sample.pdf"
    if pdf_path.exists():
        return pdf_path
    return None


# ==================== Async Configuration ====================


@pytest.fixture
def anyio_backend() -> str:
    """Force anyio to use asyncio backend only (trio not installed)."""
    return "asyncio"


# ==================== Session-level Model Fixtures ====================
# These fixtures load heavy models only once per test session
# to avoid TORCH_LIBRARY registration conflicts


@pytest.fixture(scope="session")
def doclayout_yolo_detector():
    """Create DocLayout-YOLO detector once per test session.

    This prevents TORCH_LIBRARY re-registration errors when running
    multiple tests that use the detector.

    Returns:
        DocLayoutYOLODetector instance or None if unavailable
    """
    try:
        from pipeline.layout.detection.doclayout_yolo import DocLayoutYOLODetector

        return DocLayoutYOLODetector(confidence_threshold=0.5)
    except Exception as e:
        pytest.skip(f"DocLayoutYOLODetector not available: {e}")
        return None


@pytest.fixture(scope="session")
def mineru_doclayout_yolo_detector():
    """Create Mineru DocLayout-YOLO detector once per test session.

    Returns:
        MineruDocLayoutYOLODetector instance or None if unavailable
    """
    try:
        from pipeline.layout.detection.mineru.doclayout_yolo import MinerUDocLayoutYOLODetector

        return MinerUDocLayoutYOLODetector(confidence_threshold=0.5)
    except Exception as e:
        pytest.skip(f"MineruDocLayoutYOLODetector not available: {e}")
        return None


@pytest.fixture(scope="session")
def gpu_config():
    """Get GPU config once per test session.

    This prevents TORCH_LIBRARY re-registration errors.
    """
    from pipeline.gpu_environment import get_gpu_config

    return get_gpu_config()


@pytest.fixture(scope="session")
def openai_recognizer(gpu_config):
    """Create OpenAI recognizer once per test session.

    Depends on gpu_config to ensure torch is loaded first.
    """
    from pipeline.recognition import create_recognizer

    return create_recognizer("openai")


@pytest.fixture(scope="session")
def gemini_recognizer(gpu_config):
    """Create Gemini recognizer once per test session.

    Depends on gpu_config to ensure torch is loaded first.
    """
    from pipeline.recognition import create_recognizer

    return create_recognizer("gemini")


# ==================== Helper Functions ====================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (may require models)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (require full setup)")
    config.addinivalue_line("markers", "slow: Slow tests (skip with -m 'not slow')")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
