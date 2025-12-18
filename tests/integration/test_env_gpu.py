"""GPU environment tests.

These tests require CUDA-capable GPU and will be skipped if not available.
Run with: uv run pytest -m gpu
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


class TestGPUAvailability:
    """Tests for GPU availability and configuration."""

    def test_cuda_available(self, has_gpu: bool):
        """Test that CUDA is available."""
        assert has_gpu is True, "CUDA should be available for GPU tests"

    def test_gpu_count(self, gpu_count: int):
        """Test GPU count detection."""
        assert gpu_count >= 1, f"Expected at least 1 GPU, got {gpu_count}"

    def test_gpu_memory(self, gpu_count: int):
        """Test GPU memory detection."""
        import torch

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            assert memory_gb > 0, f"GPU {i} should have memory > 0"
            print(f"GPU {i}: {props.name}, Memory: {memory_gb:.1f} GB")


class TestGPUDetectorPerformance:
    """Tests for GPU-accelerated detector performance."""

    def test_doclayout_yolo_gpu(self, sample_image, doclayout_yolo_detector):
        """Test DocLayout-YOLO detector on GPU."""
        import time

        # Warm up
        _ = doclayout_yolo_detector.detect(sample_image)

        # Time inference
        start = time.perf_counter()
        blocks = doclayout_yolo_detector.detect(sample_image)
        elapsed = time.perf_counter() - start

        print(f"DocLayout-YOLO inference time: {elapsed*1000:.1f}ms, blocks: {len(blocks)}")
        assert elapsed < 5.0, "Inference should complete within 5 seconds"

    def test_paddleocr_doclayout_v2_gpu(self, sample_image):
        """Test PaddleOCR DocLayout-V2 detector on GPU."""
        import time

        from pipeline.layout.detection import create_detector

        detector = create_detector("paddleocr-doclayout-v2")

        # Warm up
        _ = detector.detect(sample_image)

        # Time inference
        start = time.perf_counter()
        blocks = detector.detect(sample_image)
        elapsed = time.perf_counter() - start

        print(f"PaddleOCR DocLayout-V2 inference time: {elapsed*1000:.1f}ms, blocks: {len(blocks)}")
        assert elapsed < 5.0, "Inference should complete within 5 seconds"


class TestGPUMemoryManagement:
    """Tests for GPU memory management."""

    def test_memory_cleanup_after_detection(self, sample_image):
        """Test that GPU memory is properly cleaned up after detection."""
        import gc

        import torch

        from pipeline.layout.detection import create_detector

        initial_memory = torch.cuda.memory_allocated()

        # Run detection
        detector = create_detector("paddleocr-doclayout-v2")
        _ = detector.detect(sample_image)

        # Cleanup
        del detector
        gc.collect()
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory should not grow significantly (allow some variance)
        memory_diff_mb = (final_memory - initial_memory) / (1024**2)
        print(f"Memory difference: {memory_diff_mb:.1f} MB")
        # Allow up to 500MB variance due to CUDA context
        assert memory_diff_mb < 500, f"Memory leak detected: {memory_diff_mb:.1f} MB"


class TestMultiGPU:
    """Tests for multi-GPU configurations."""

    def test_multi_gpu_detection(self, gpu_count: int):
        """Test multi-GPU environment detection."""
        if gpu_count < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")

        import torch

        for i in range(gpu_count):
            device = torch.device(f"cuda:{i}")
            # Simple tensor operation on each GPU
            t = torch.zeros(100, device=device)
            assert t.device.index == i

    def test_gpu_config_multi_gpu(self, gpu_count: int, gpu_config):
        """Test GPU config for multi-GPU setup."""
        if gpu_count < 2:
            pytest.skip("Multi-GPU test requires at least 2 GPUs")

        assert gpu_config.gpu_count == gpu_count
        assert len(gpu_config.gpu_names) == gpu_count
        print(f"Multi-GPU config: {gpu_count} GPUs")
        for i, name in enumerate(gpu_config.gpu_names):
            print(f"  GPU {i}: {name}")
