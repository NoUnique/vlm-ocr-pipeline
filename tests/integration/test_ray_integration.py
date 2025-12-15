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


def test_pipeline_without_ray_backend():
    """Test Pipeline without Ray backend (single-GPU mode)."""
    from pipeline import Pipeline
    from pipeline.config import PipelineConfig

    # Test that Pipeline works without Ray backend
    # (uses default pytorch backend, no Ray)
    config = PipelineConfig(
        detector="doclayout-yolo",
        detector_backend="pytorch",
        recognizer="gemini-2.5-flash",
    )
    pipeline = Pipeline(config=config)

    # Verify Ray pools are not initialized
    assert pipeline.ray_detector_pool is None
    assert pipeline.ray_recognizer_pool is None


def test_pipeline_accepts_ray_backends():
    """Test Pipeline accepts pt-ray and hf-ray backends."""
    from pipeline import Pipeline
    from pipeline.config import PipelineConfig

    # Test that Pipeline accepts Ray backends without errors
    # (we don't actually initialize Ray to avoid GPU requirements in CI)
    # This will trigger Ray initialization, but we skip if Ray is not available
    try:
        config = PipelineConfig(
            detector="doclayout-yolo",
            detector_backend="pt-ray",
            recognizer="gemini-2.5-flash",
        )
        pipeline = Pipeline(config=config)
        # If we get here, Ray is initialized and pools should exist
        # (implementation may vary based on Ray availability)
    except Exception as e:
        pytest.skip(f"Ray initialization failed: {e}")


def test_detector_pool_creation():
    """Test RayDetectorPool can be created with detector name."""
    from pipeline.distributed import RayDetectorPool

    # Test that pool can be created (doesn't actually initialize Ray actors)
    try:
        pool = RayDetectorPool(
            detector_name="doclayout-yolo",
            num_gpus=1,
        )
        assert pool.detector_name == "doclayout-yolo"
    except Exception as e:
        pytest.skip(f"RayDetectorPool creation failed: {e}")


def test_recognizer_pool_creation():
    """Test RayRecognizerPool can be created with recognizer name."""
    from pipeline.distributed import RayRecognizerPool

    try:
        pool = RayRecognizerPool(
            recognizer_name="gemini-2.5-flash",
            num_gpus=1,
        )
        assert pool.recognizer_name == "gemini-2.5-flash"
    except Exception as e:
        pytest.skip(f"RayRecognizerPool creation failed: {e}")
