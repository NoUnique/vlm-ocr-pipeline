"""Ray distributed computing tests.

These tests require Ray to be installed and will be skipped if not available.
Run with: uv run pytest -m ray
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ray


class TestRayAvailability:
    """Tests for Ray availability."""

    def test_ray_installed(self, has_ray: bool):
        """Test that Ray is installed."""
        assert has_ray is True, "Ray should be installed for Ray tests"

    def test_ray_init(self):
        """Test Ray initialization."""
        import ray

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=4)

        assert ray.is_initialized()

        # Get cluster info
        resources = ray.cluster_resources()
        print(f"Ray cluster resources: {resources}")

        ray.shutdown()


class TestRayDetectorPool:
    """Tests for Ray detector pool."""

    def test_create_ray_detector_pool(self, has_gpu: bool, gpu_count: int):
        """Test creating Ray detector pool."""
        import ray

        from pipeline.distributed.ray_detector import RayDetectorPool

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            num_actors = min(gpu_count, 2) if has_gpu else 1

            pool = RayDetectorPool(
                detector_name="paddleocr-doclayout-v2",
                num_actors=num_actors,
            )

            assert pool is not None
            print(f"Created Ray detector pool with {num_actors} actors")

            pool.shutdown()
        finally:
            ray.shutdown()

    def test_ray_detector_pool_detect(self, sample_image, has_gpu: bool, gpu_count: int):
        """Test Ray detector pool detection."""
        import ray

        from pipeline.distributed.ray_detector import RayDetectorPool

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            num_actors = min(gpu_count, 2) if has_gpu else 1

            with RayDetectorPool(
                detector_name="paddleocr-doclayout-v2",
                num_actors=num_actors,
                
            ) as pool:
                # Detect on single image
                blocks = pool.detect(sample_image)

                print(f"Detected {len(blocks)} blocks using Ray pool")
                assert isinstance(blocks, list)
        finally:
            ray.shutdown()

    def test_ray_detector_pool_batch(self, sample_image, has_gpu: bool, gpu_count: int):
        """Test Ray detector pool batch detection."""
        import ray

        from pipeline.distributed.ray_detector import RayDetectorPool

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            num_actors = min(gpu_count, 2) if has_gpu else 1

            with RayDetectorPool(
                detector_name="paddleocr-doclayout-v2",
                num_actors=num_actors,
                
            ) as pool:
                # Batch detect
                images = [sample_image] * 4
                results = pool.detect_batch(images)

                print(f"Batch detected {len(results)} pages")
                assert len(results) == 4
                for blocks in results:
                    assert isinstance(blocks, list)
        finally:
            ray.shutdown()


class TestRayRecognizerPool:
    """Tests for Ray recognizer pool."""

    @pytest.mark.api_gemini
    def test_create_ray_recognizer_pool(self):
        """Test creating Ray recognizer pool with Gemini."""
        import ray

        from pipeline.distributed.ray_recognizer import RayRecognizerPool

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            pool = RayRecognizerPool(
                recognizer_name="gemini-2.5-flash",
                num_actors=2,
            )

            assert pool is not None
            print("Created Ray recognizer pool with 2 actors")

            pool.shutdown()
        finally:
            ray.shutdown()


class TestRayParallelPerformance:
    """Tests for Ray parallel processing performance."""

    def test_ray_vs_sequential_detection(self, sample_image, has_gpu: bool, gpu_count: int):
        """Compare Ray parallel vs sequential detection performance."""
        import time

        import ray

        from pipeline.distributed.ray_detector import RayDetectorPool
        from pipeline.layout.detection import create_detector

        if gpu_count < 2:
            pytest.skip("Performance comparison requires at least 2 GPUs")

        images = [sample_image] * 8

        # Sequential detection
        detector = create_detector("paddleocr-doclayout-v2")
        start = time.perf_counter()
        for img in images:
            _ = detector.detect(img)
        sequential_time = time.perf_counter() - start
        del detector

        # Ray parallel detection
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        try:
            with RayDetectorPool(
                detector_name="paddleocr-doclayout-v2",
                num_actors=gpu_count,
                
            ) as pool:
                start = time.perf_counter()
                _ = pool.detect_batch(images)
                parallel_time = time.perf_counter() - start

            speedup = sequential_time / parallel_time
            print(f"Sequential: {sequential_time:.2f}s, Parallel: {parallel_time:.2f}s")
            print(f"Speedup: {speedup:.2f}x with {gpu_count} GPUs")

            # Expect some speedup with multiple GPUs
            assert speedup > 1.0, "Parallel should be faster than sequential"
        finally:
            ray.shutdown()
