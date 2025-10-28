"""Tests for Ray multi-GPU integration.

This module tests Ray-based distributed processing functionality.
Tests are skipped if Ray is not installed (optional dependency).
"""

from __future__ import annotations

import pytest

from pipeline.distributed import is_ray_available

# Skip all tests if Ray is not available
pytestmark = pytest.mark.skipif(not is_ray_available(), reason="Ray not installed")


def test_is_ray_available():
    """Test Ray availability detection."""
    assert is_ray_available() is True


def test_ray_detector_pool_import():
    """Test RayDetectorPool can be imported."""
    from pipeline.distributed import RayDetectorPool

    assert RayDetectorPool is not None


def test_ray_recognizer_pool_import():
    """Test RayRecognizerPool can be imported."""
    from pipeline.distributed import RayRecognizerPool

    assert RayRecognizerPool is not None


def test_pipeline_with_ray_params():
    """Test Pipeline accepts Ray parameters."""
    from pipeline import Pipeline

    # Test that Pipeline accepts Ray parameters without errors
    # (we don't actually initialize Ray to avoid GPU requirements in CI)
    pipeline = Pipeline(
        detector="doclayout-yolo",
        recognizer="gemini-2.5-flash",
        use_ray=False,  # Don't actually use Ray (no GPUs in CI)
        num_gpus=None,
        actors_per_gpu=1,
    )

    assert pipeline.use_ray is False
    assert pipeline.ray_detector_pool is None
    assert pipeline.ray_recognizer_pool is None


def test_graceful_fallback_no_ray():
    """Test graceful fallback when Ray is not used."""
    from pipeline import Pipeline

    # Create pipeline without Ray
    pipeline = Pipeline(
        detector="doclayout-yolo",
        recognizer="gemini-2.5-flash",
        use_ray=False,
    )

    # Verify Ray pools are not initialized
    assert pipeline.use_ray is False
    assert pipeline.ray_detector_pool is None
    assert pipeline.ray_recognizer_pool is None

    # Verify stages work without Ray pools
    assert pipeline.detection_stage is not None
    assert pipeline.recognition_stage is not None


def test_detection_stage_with_ray_pool():
    """Test DetectionStage accepts ray_detector_pool parameter."""
    from pipeline.layout.detection import create_detector
    from pipeline.stages import DetectionStage

    detector = create_detector("doclayout-yolo")
    stage = DetectionStage(detector, ray_detector_pool=None)

    assert stage.detector is not None
    assert stage.ray_detector_pool is None


def test_recognition_stage_with_ray_pool():
    """Test RecognitionStage accepts ray_recognizer_pool parameter."""
    from pipeline.recognition import create_recognizer
    from pipeline.stages import RecognitionStage

    recognizer = create_recognizer("gemini-2.5-flash", backend="gemini")
    stage = RecognitionStage(recognizer, ray_recognizer_pool=None)

    assert stage.recognizer is not None
    assert stage.ray_recognizer_pool is None
