"""Tests for GPU auto-optimization."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestGPUEnvironmentDetection:
    """Tests for GPU environment detection."""

    def test_cpu_fallback_config_no_torch(self):
        """Test CPU fallback configuration when PyTorch is not importable."""
        from pipeline.gpu_environment import get_gpu_config

        # Clear cache to force re-detection
        get_gpu_config.cache_clear()

        # Mock torch ImportError by temporarily removing it from sys.modules
        # and patching builtins.__import__
        _original_modules = sys.modules.copy()  # Keep for reference if restore needed

        def mock_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        original_import = __builtins__["__import__"]  # type: ignore[index]

        try:
            with patch.dict("sys.modules", {"torch": None}):
                with patch("builtins.__import__", side_effect=mock_import):
                    get_gpu_config.cache_clear()
                    config = get_gpu_config()

                    assert config.has_cuda is False
                    assert config.gpu_count == 0
                    assert config.recommended_backend == "pytorch"
        finally:
            get_gpu_config.cache_clear()

    def test_cpu_fallback_config_no_cuda(self):
        """Test CPU fallback configuration when CUDA is not available."""
        from pipeline.gpu_environment import get_gpu_config

        get_gpu_config.cache_clear()

        # Create a mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            get_gpu_config.cache_clear()
            config = get_gpu_config()

            assert config.has_cuda is False
            assert config.gpu_count == 0
            assert config.optimization_strategy == "cpu_fallback"

    def test_actual_gpu_detection(self, gpu_config):
        """Test actual GPU detection (runs only if CUDA available).

        Uses session-scoped fixture to avoid TORCH_LIBRARY conflicts.
        """
        if not gpu_config.has_cuda:
            pytest.skip("CUDA not available")

        assert gpu_config.has_cuda is True
        assert gpu_config.gpu_count >= 1
        assert len(gpu_config.gpu_names) == gpu_config.gpu_count
        assert gpu_config.per_gpu_memory_gb > 0
        # hf = huggingface transformers backend
        assert gpu_config.recommended_backend in ["pytorch", "vllm", "sglang", "hf"]


class TestGPUConfigFunctions:
    """Tests for GPU config helper functions."""

    def test_cpu_fallback_returns_valid_config(self):
        """Test _cpu_fallback_config returns valid GPUConfig."""
        from pipeline.gpu_environment import _cpu_fallback_config

        config = _cpu_fallback_config()

        assert config.has_cuda is False
        assert config.gpu_count == 0
        assert config.gpu_names == []
        assert config.total_memory_gb == 0
        assert config.per_gpu_memory_gb == 0
        assert config.compute_capability == (0, 0)
        assert config.recommended_backend == "pytorch"
        assert config.tensor_parallel_size == 1
        assert config.data_parallel_workers == 1
        assert config.gpu_memory_utilization == 0.0
        assert config.batch_size == 1
        assert config.use_flash_attention is False
        assert config.use_bf16 is False
        assert config.optimization_strategy == "cpu_fallback"
        assert config.expected_speedup == 0.1  # CPU is ~10x slower than GPU

    def test_auto_optimize_single_gpu(self):
        """Test _auto_optimize for single GPU configuration."""
        from pipeline.gpu_environment import _auto_optimize

        config = _auto_optimize(
            gpu_count=1,
            gpu_names=["NVIDIA A100-SXM4-80GB"],
            per_gpu_memory_gb=80.0,
            total_memory_gb=80.0,
            compute_capability=(8, 0),
        )

        assert config.has_cuda is True
        assert config.gpu_count == 1
        assert config.tensor_parallel_size == 1
        assert config.data_parallel_workers == 1
        assert config.use_bf16 is True  # A100 supports BF16

    def test_auto_optimize_multi_gpu(self):
        """Test _auto_optimize for multi-GPU configuration."""
        from pipeline.gpu_environment import _auto_optimize

        config = _auto_optimize(
            gpu_count=4,
            gpu_names=["NVIDIA A100-SXM4-80GB"] * 4,
            per_gpu_memory_gb=80.0,
            total_memory_gb=320.0,
            compute_capability=(8, 0),
        )

        assert config.has_cuda is True
        assert config.gpu_count == 4
        assert config.tensor_parallel_size >= 1
        assert config.use_bf16 is True

    def test_auto_optimize_low_memory_gpu(self):
        """Test _auto_optimize for low memory GPU."""
        from pipeline.gpu_environment import _auto_optimize

        config = _auto_optimize(
            gpu_count=1,
            gpu_names=["NVIDIA GeForce RTX 3080"],
            per_gpu_memory_gb=10.0,
            total_memory_gb=10.0,
            compute_capability=(8, 6),
        )

        assert config.has_cuda is True
        assert config.gpu_count == 1
        # Lower memory should result in smaller batch size
        assert config.batch_size <= 4


class TestGPUConfigSingleton:
    """Test that GPU config is cached (singleton pattern)."""

    def test_config_is_cached(self, gpu_config):
        """Test that get_gpu_config() returns cached result.

        Uses session-scoped fixture to avoid TORCH_LIBRARY conflicts.
        """
        from pipeline.gpu_environment import get_gpu_config

        # Second call should return same object (cached)
        config2 = get_gpu_config()

        # gpu_config fixture already called get_gpu_config(), so this should be cached
        assert gpu_config is config2

    @pytest.mark.skip(reason="cache_clear causes TORCH_LIBRARY re-registration in multi-file test runs")
    def test_cache_clear_works(self, gpu_config):
        """Test that cache_clear resets the cache.

        Note: This test is skipped because cache_clear() causes torch
        to be re-imported, triggering TORCH_LIBRARY conflicts when
        running all tests together. The cache functionality is
        already tested by test_config_is_cached.
        """
        from pipeline.gpu_environment import get_gpu_config

        # Clear cache
        get_gpu_config.cache_clear()

        # Get new config
        config2 = get_gpu_config()

        # Should be equal but not same object (re-computed)
        # Note: they will be the same values since hardware hasn't changed
        assert gpu_config.has_cuda == config2.has_cuda
        assert gpu_config.gpu_count == config2.gpu_count


class TestBackendAvailability:
    """Tests for backend availability checking."""

    def test_check_backend_available_vllm(self):
        """Test _check_backend_available for vLLM."""
        from pipeline.gpu_environment import _check_backend_available

        # This will return True if vllm is installed, False otherwise
        result = _check_backend_available("vllm")
        assert isinstance(result, bool)

    def test_check_backend_available_sglang(self):
        """Test _check_backend_available for SGLang."""
        from pipeline.gpu_environment import _check_backend_available

        result = _check_backend_available("sglang")
        assert isinstance(result, bool)

    def test_check_backend_available_unknown(self):
        """Test _check_backend_available for unknown backend."""
        from pipeline.gpu_environment import _check_backend_available

        result = _check_backend_available("unknown_backend")
        assert result is False
