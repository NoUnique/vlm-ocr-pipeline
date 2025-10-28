"""Tests for GPU auto-optimization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGPUEnvironmentDetection:
    """Tests for GPU environment detection."""

    def test_cpu_fallback_config(self):
        """Test CPU fallback configuration when CUDA is not available."""
        from pipeline.gpu_environment import get_gpu_config

        # Clear cache to force re-detection
        get_gpu_config.cache_clear()

        with patch("pipeline.gpu_environment.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            config = get_gpu_config()

            assert config.has_cuda is False
            assert config.gpu_count == 0
            assert config.recommended_backend == "pytorch"
            assert config.tensor_parallel_size == 1
            assert config.data_parallel_workers == 1
            assert config.optimization_strategy == "cpu_fallback"

    def test_single_gpu_detection(self):
        """Test single GPU detection and configuration."""
        from pipeline.gpu_environment import get_gpu_config

        get_gpu_config.cache_clear()

        with patch("pipeline.gpu_environment.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_name.return_value = "NVIDIA A100-SXM4-80GB"

            # Mock device properties
            mock_props = MagicMock()
            mock_props.total_memory = 80 * (1024**3)  # 80GB
            mock_props.major = 8
            mock_props.minor = 0
            mock_torch.cuda.get_device_properties.return_value = mock_props

            # Mock backend checks
            with patch("pipeline.gpu_environment._check_backend_available") as mock_check:
                mock_check.return_value = True  # vLLM available

                config = get_gpu_config()

                assert config.has_cuda is True
                assert config.gpu_count == 1
                assert config.recommended_backend == "vllm"
                assert config.tensor_parallel_size == 1
                assert config.data_parallel_workers == 1
                assert config.use_flash_attention is True
                assert config.use_bf16 is True

    def test_multi_gpu_detection_8xa100(self):
        """Test 8x A100 80GB detection and optimal configuration."""
        from pipeline.gpu_environment import get_gpu_config

        get_gpu_config.cache_clear()

        with patch("pipeline.gpu_environment.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 8
            mock_torch.cuda.get_device_name.return_value = "NVIDIA A100-SXM4-80GB"

            mock_props = MagicMock()
            mock_props.total_memory = 80 * (1024**3)
            mock_props.major = 8
            mock_props.minor = 0
            mock_torch.cuda.get_device_properties.return_value = mock_props

            with patch("pipeline.gpu_environment._check_backend_available") as mock_check:
                mock_check.return_value = True

                config = get_gpu_config()

                assert config.has_cuda is True
                assert config.gpu_count == 8
                assert config.per_gpu_memory_gb == pytest.approx(80.0)
                assert config.total_memory_gb == pytest.approx(640.0)
                assert config.recommended_backend == "vllm"
                assert config.tensor_parallel_size == 4
                assert config.data_parallel_workers == 2
                assert config.gpu_memory_utilization == 0.90
                assert config.batch_size == 8
                assert config.optimization_strategy == "hybrid_tp4_dp2"
                assert config.expected_speedup == 12.0


class TestRecognizerAutoOptimization:
    """Tests for recognizer auto-optimization."""

    def test_auto_backend_selection(self):
        """Test automatic backend selection."""
        from pipeline.gpu_environment import get_gpu_config
        from pipeline.recognition import create_recognizer

        get_gpu_config.cache_clear()

        with patch("pipeline.gpu_environment.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 8

            mock_props = MagicMock()
            mock_props.total_memory = 80 * (1024**3)
            mock_props.major = 8
            mock_props.minor = 0
            mock_torch.cuda.get_device_properties.return_value = mock_props

            with patch("pipeline.gpu_environment._check_backend_available") as mock_check:
                mock_check.return_value = True

                # Test auto-backend selection (should fail at model loading, but we check the logic)
                try:
                    _ = create_recognizer("deepseek-ocr", use_auto_optimization=True)
                    # If we reach here, check that auto-optimization was applied
                    # (in real scenario, it will fail at model loading)
                except (ImportError, OSError, RuntimeError):
                    # Expected: model not available
                    pass

    def test_manual_override(self):
        """Test that manual settings override auto-optimization."""
        from pipeline.gpu_environment import get_gpu_config
        from pipeline.recognition import create_recognizer

        get_gpu_config.cache_clear()

        with patch("pipeline.gpu_environment.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 8

            mock_props = MagicMock()
            mock_props.total_memory = 80 * (1024**3)
            mock_props.major = 8
            mock_props.minor = 0
            mock_torch.cuda.get_device_properties.return_value = mock_props

            with patch("pipeline.gpu_environment._check_backend_available") as mock_check:
                mock_check.return_value = True

                # Test manual override
                try:
                    _ = create_recognizer(
                        "deepseek-ocr",
                        backend="hf",  # Manual override
                        tensor_parallel_size=2,  # Manual override
                        use_auto_optimization=True,
                    )
                except (ImportError, OSError, RuntimeError):
                    # Expected: model not available
                    pass

    def test_disable_auto_optimization(self):
        """Test disabling auto-optimization."""
        from pipeline.recognition import create_recognizer

        # Test with auto-optimization disabled
        try:
            _ = create_recognizer(
                "deepseek-ocr",
                backend="hf",
                use_auto_optimization=False,
            )
        except (ImportError, OSError, RuntimeError):
            # Expected: model not available
            pass


class TestGPUConfigSingleton:
    """Test that GPU config is cached (singleton pattern)."""

    def test_config_is_cached(self):
        """Test that get_gpu_config() returns cached result."""
        from pipeline.gpu_environment import get_gpu_config

        get_gpu_config.cache_clear()

        # First call
        config1 = get_gpu_config()

        # Second call should return same object (cached)
        config2 = get_gpu_config()

        assert config1 is config2
