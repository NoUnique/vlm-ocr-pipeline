"""Tests for adaptive batch size calibration.

This module tests the BatchSizeCalibrator and related functions.
Uses mocks to avoid actual GPU/model operations.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.optimization import (
    BatchSizeCalibrator,
    calibrate_batch_size,
    get_optimal_batch_size,
)


class TestBatchSizeCalibrator:
    """Tests for BatchSizeCalibrator class."""

    def test_init_defaults(self):
        """Test calibrator initialization with defaults."""
        calibrator = BatchSizeCalibrator()

        assert calibrator.cache_dir == Path.home() / ".cache" / "vlm-ocr-pipeline"
        assert calibrator.target_memory_fraction == 0.85
        assert calibrator.min_batch_size == 1
        assert calibrator.max_batch_size == 128
        assert calibrator.use_cache is True

    def test_init_custom_params(self):
        """Test calibrator initialization with custom parameters."""
        custom_cache_dir = Path("/tmp/test_cache")
        calibrator = BatchSizeCalibrator(
            cache_dir=custom_cache_dir,
            target_memory_fraction=0.7,
            min_batch_size=2,
            max_batch_size=64,
            use_cache=False,
        )

        assert calibrator.cache_dir == custom_cache_dir
        assert calibrator.target_memory_fraction == 0.7
        assert calibrator.min_batch_size == 2
        assert calibrator.max_batch_size == 64
        assert calibrator.use_cache is False

    def test_generate_cache_key(self):
        """Test cache key generation is deterministic."""
        calibrator = BatchSizeCalibrator()

        key1 = calibrator._generate_cache_key("doclayout-yolo", (1920, 1080, 3), "NVIDIA RTX 4090")
        key2 = calibrator._generate_cache_key("doclayout-yolo", (1920, 1080, 3), "NVIDIA RTX 4090")

        # Same inputs should produce same key
        assert key1 == key2
        assert len(key1) == 16  # MD5 hash truncated to 16 chars

        # Different inputs should produce different keys
        key3 = calibrator._generate_cache_key("doclayout-yolo", (1920, 1080, 3), "NVIDIA RTX 3090")
        assert key1 != key3

    def test_detect_gpu_model_with_cuda(self):
        """Test GPU model detection with CUDA available."""
        import sys
        from unittest.mock import MagicMock

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        sys.modules["torch"] = mock_torch

        try:
            calibrator = BatchSizeCalibrator()
            gpu_model = calibrator._detect_gpu_model()

            assert gpu_model == "NVIDIA RTX 4090"
        finally:
            # Cleanup
            if "torch" in sys.modules:
                del sys.modules["torch"]

    def test_detect_gpu_model_without_cuda(self):
        """Test GPU model detection without CUDA."""
        import sys
        from unittest.mock import MagicMock

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        sys.modules["torch"] = mock_torch

        try:
            calibrator = BatchSizeCalibrator()
            gpu_model = calibrator._detect_gpu_model()

            assert gpu_model == "CPU"
        finally:
            # Cleanup
            if "torch" in sys.modules:
                del sys.modules["torch"]

    def test_detect_gpu_model_exception(self):
        """Test GPU model detection handles exceptions."""
        import sys
        from unittest.mock import MagicMock

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
        sys.modules["torch"] = mock_torch

        try:
            calibrator = BatchSizeCalibrator()
            gpu_model = calibrator._detect_gpu_model()

            assert gpu_model == "Unknown"
        finally:
            # Cleanup
            if "torch" in sys.modules:
                del sys.modules["torch"]

    def test_binary_search_success(self):
        """Test binary search finds optimal batch size."""
        calibrator = BatchSizeCalibrator(min_batch_size=1, max_batch_size=16)

        # Mock inference function that succeeds up to batch_size=8, fails at 9+
        call_count = {"value": 0}

        def mock_inference(batch_size: int):
            call_count["value"] += 1
            if batch_size <= 8:
                return batch_size
            raise RuntimeError("CUDA out of memory")

        with patch.object(calibrator, "_test_batch_size") as mock_test:
            # Mock _test_batch_size to use our mock_inference logic
            mock_test.side_effect = lambda fn, bs: bs <= 8

            optimal = calibrator._binary_search(mock_inference)

            # Should find batch_size=8 as optimal
            assert optimal == 8

    def test_binary_search_all_fail(self):
        """Test binary search when all batch sizes fail."""
        calibrator = BatchSizeCalibrator(min_batch_size=1, max_batch_size=16)

        # Mock inference that always fails
        def mock_inference(batch_size: int):
            raise RuntimeError("CUDA out of memory")

        with patch.object(calibrator, "_test_batch_size") as mock_test:
            mock_test.return_value = False

            optimal = calibrator._binary_search(mock_inference)

            # Should return min_batch_size
            assert optimal == 1

    def test_test_batch_size_success(self):
        """Test _test_batch_size with successful inference."""
        import sys
        from unittest.mock import MagicMock

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = MagicMock()
        sys.modules["torch"] = mock_torch

        try:
            calibrator = BatchSizeCalibrator()

            def mock_inference(batch_size: int):
                return batch_size

            result = calibrator._test_batch_size(mock_inference, 4)

            assert result is True
            assert mock_torch.cuda.empty_cache.call_count >= 2  # Before and after
        finally:
            # Cleanup
            if "torch" in sys.modules:
                del sys.modules["torch"]

    def test_test_batch_size_oom(self):
        """Test _test_batch_size with OOM error."""
        import sys
        from unittest.mock import MagicMock

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = MagicMock()
        sys.modules["torch"] = mock_torch

        try:
            calibrator = BatchSizeCalibrator()

            def mock_inference(batch_size: int):
                raise RuntimeError("CUDA out of memory")

            result = calibrator._test_batch_size(mock_inference, 16)

            assert result is False
        finally:
            # Cleanup
            if "torch" in sys.modules:
                del sys.modules["torch"]

    def test_test_batch_size_unexpected_error(self):
        """Test _test_batch_size re-raises unexpected errors."""
        import sys
        from unittest.mock import MagicMock

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        sys.modules["torch"] = mock_torch

        try:
            calibrator = BatchSizeCalibrator()

            def mock_inference(batch_size: int):
                raise ValueError("Unexpected error")

            with pytest.raises(ValueError, match="Unexpected error"):
                calibrator._test_batch_size(mock_inference, 4)
        finally:
            # Cleanup
            if "torch" in sys.modules:
                del sys.modules["torch"]

    def test_cache_save_and_load(self):
        """Test cache save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            calibrator = BatchSizeCalibrator(cache_dir=cache_dir)

            # Save to cache
            cache_key = "test_key_123"
            calibrator._save_to_cache(cache_key, 16, "doclayout-yolo", "NVIDIA RTX 4090")

            # Verify cache file exists
            assert calibrator.cache_file.exists()

            # Load from cache
            batch_size = calibrator._load_from_cache(cache_key)
            assert batch_size == 16

    def test_cache_load_nonexistent(self):
        """Test loading from nonexistent cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            calibrator = BatchSizeCalibrator(cache_dir=cache_dir)

            # Load from nonexistent cache
            batch_size = calibrator._load_from_cache("nonexistent_key")
            assert batch_size is None

    def test_cache_load_invalid_json(self):
        """Test loading from corrupted cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "batch_size_cache.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Write invalid JSON
            cache_file.write_text("invalid json {")

            calibrator = BatchSizeCalibrator(cache_dir=cache_dir)
            batch_size = calibrator._load_from_cache("any_key")

            assert batch_size is None

    def test_calibrate_with_cache_hit(self):
        """Test calibration uses cached result."""
        import sys
        from unittest.mock import MagicMock

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        sys.modules["torch"] = mock_torch

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_dir = Path(tmpdir)
                calibrator = BatchSizeCalibrator(cache_dir=cache_dir)

                # Pre-populate cache
                cache_key = calibrator._generate_cache_key("doclayout-yolo", (1920, 1080, 3), "NVIDIA RTX 4090")
                calibrator._save_to_cache(cache_key, 16, "doclayout-yolo", "NVIDIA RTX 4090")

                # Mock inference (should not be called)
                def mock_inference(batch_size: int):
                    raise AssertionError("Should not be called - cache hit!")

                # Calibrate (should use cache)
                optimal = calibrator.calibrate(
                    inference_fn=mock_inference,
                    model_name="doclayout-yolo",
                    input_shape=(1920, 1080, 3),
                )

                assert optimal == 16
        finally:
            # Cleanup
            if "torch" in sys.modules:
                del sys.modules["torch"]

    def test_calibrate_with_cache_miss(self):
        """Test calibration runs binary search on cache miss."""
        import sys
        from unittest.mock import MagicMock

        # Mock torch in sys.modules
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        sys.modules["torch"] = mock_torch

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_dir = Path(tmpdir)
                calibrator = BatchSizeCalibrator(
                    cache_dir=cache_dir,
                    min_batch_size=1,
                    max_batch_size=16,
                )

                def mock_inference(batch_size: int):
                    return batch_size

                with patch.object(calibrator, "_binary_search") as mock_search:
                    mock_search.return_value = 8

                    # Calibrate (should run binary search)
                    optimal = calibrator.calibrate(
                        inference_fn=mock_inference,
                        model_name="doclayout-yolo",
                        input_shape=(1920, 1080, 3),
                    )

                    assert optimal == 8
                    mock_search.assert_called_once()

                    # Verify cache was updated
                    cache_key = calibrator._generate_cache_key("doclayout-yolo", (1920, 1080, 3), "NVIDIA RTX 4090")
                    cached = calibrator._load_from_cache(cache_key)
                    assert cached == 8
        finally:
            # Cleanup
            if "torch" in sys.modules:
                del sys.modules["torch"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @patch("pipeline.optimization.batch_size.BatchSizeCalibrator")
    def test_calibrate_batch_size(self, mock_calibrator_class):
        """Test calibrate_batch_size convenience function."""
        mock_calibrator = MagicMock()
        mock_calibrator.calibrate.return_value = 16
        mock_calibrator_class.return_value = mock_calibrator

        def mock_inference(batch_size: int):
            return batch_size

        optimal = calibrate_batch_size(
            inference_fn=mock_inference,
            model_name="doclayout-yolo",
            input_shape=(1920, 1080, 3),
            target_memory_fraction=0.8,
            min_batch_size=2,
            max_batch_size=64,
            use_cache=True,
        )

        assert optimal == 16

        # Verify BatchSizeCalibrator was created with correct params
        mock_calibrator_class.assert_called_once_with(
            target_memory_fraction=0.8,
            min_batch_size=2,
            max_batch_size=64,
            use_cache=True,
        )

        # Verify calibrate was called
        mock_calibrator.calibrate.assert_called_once_with(
            inference_fn=mock_inference,
            model_name="doclayout-yolo",
            input_shape=(1920, 1080, 3),
        )

    def test_get_optimal_batch_size_cached(self):
        """Test get_optimal_batch_size with cached result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "vlm-ocr-pipeline"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "batch_size_cache.json"

            # Pre-populate cache
            calibrator = BatchSizeCalibrator(cache_dir=cache_dir)
            cache_key = calibrator._generate_cache_key("doclayout-yolo", (1920, 1080, 3), "NVIDIA RTX 4090")
            cache_data = {
                cache_key: {
                    "batch_size": 16,
                    "model_name": "doclayout-yolo",
                    "gpu_model": "NVIDIA RTX 4090",
                    "target_memory_fraction": 0.85,
                }
            }
            cache_file.write_text(json.dumps(cache_data))

            # Call get_optimal_batch_size with explicit gpu_model
            # This will create a new calibrator and read from cache
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                optimal = get_optimal_batch_size(
                    model_name="doclayout-yolo",
                    input_shape=(1920, 1080, 3),
                    gpu_model="NVIDIA RTX 4090",
                )

                assert optimal == 16

    def test_get_optimal_batch_size_not_cached(self):
        """Test get_optimal_batch_size with no cached result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Call get_optimal_batch_size (not cached)
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                optimal = get_optimal_batch_size(
                    model_name="doclayout-yolo",
                    input_shape=(1920, 1080, 3),
                    gpu_model="Test GPU",
                )

                assert optimal is None
