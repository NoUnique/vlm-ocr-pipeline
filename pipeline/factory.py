"""Component factory module for pipeline components.

This module provides:
- ComponentFactory: Unified factory for creating pipeline components
- Lazy loading of heavy dependencies
- Backend mapping and auto-optimization
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .distributed import RayDetectorPool, RayRecognizerPool
    from .gpu_environment import GPUConfig
    from .types import Detector, Recognizer, Sorter

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Unified factory for all pipeline components.

    This factory centralizes the creation of all pipeline components (detectors,
    sorters, recognizers) with proper backend resolution, configuration mapping,
    and auto-optimization.

    Benefits:
    - Single point of component creation
    - Consistent backend mapping across all components
    - Lazy loading of GPU configuration
    - Ray pool initialization support

    Example:
        >>> from pipeline.config import PipelineConfig
        >>> from pipeline.factory import ComponentFactory
        >>>
        >>> config = PipelineConfig(detector="paddleocr-doclayout-v2")
        >>> config.validate()
        >>>
        >>> factory = ComponentFactory(config)
        >>> detector = factory.create_detector()
        >>> sorter = factory.create_sorter()
        >>> recognizer = factory.create_recognizer()
    """

    def __init__(self, config: PipelineConfig):
        """Initialize component factory.

        Args:
            config: Validated pipeline configuration
        """
        self.config = config
        self._gpu_config: GPUConfig | None = None
        self._ray_detector_pool: RayDetectorPool | None = None
        self._ray_recognizer_pool: RayRecognizerPool | None = None

    @property
    def gpu_config(self) -> GPUConfig:
        """Lazy-load GPU configuration.

        Returns:
            GPU configuration with auto-detected settings
        """
        if self._gpu_config is None:
            from .gpu_environment import get_gpu_config

            self._gpu_config = get_gpu_config()
        return self._gpu_config

    def _map_backend(self, stage: str, model_name: str, backend: str | None) -> dict[str, Any]:
        """Map user-friendly backend to actual implementation parameter.

        Args:
            stage: Stage type ("detector", "sorter", "recognizer")
            model_name: Model name to look up
            backend: User-specified backend

        Returns:
            Dictionary with backend parameter name and mapped value

        Example:
            >>> factory._map_backend("recognizer", "paddleocr-vl", "pytorch")
            {"vl_rec_backend": "native"}
        """
        if backend is None:
            return {}

        models_config = self.config.models_config
        stage_config = models_config.get(f"{stage}s", {}).get(model_name, {})
        backend_mapping = stage_config.get("backend_mapping", {})
        backend_param_name = stage_config.get("backend_param_name", "backend")

        if backend_mapping and backend in backend_mapping:
            mapped_value = backend_mapping[backend]
            return {backend_param_name: mapped_value}
        elif backend_param_name:
            return {backend_param_name: backend}
        return {}

    def create_detector(self) -> Detector | None:
        """Create detector instance based on configuration.

        Returns:
            Detector instance or None if detector is "none"

        Example:
            >>> detector = factory.create_detector()
            >>> blocks = detector.detect(image)
        """
        from .layout.detection import create_detector as create_detector_impl

        detector = self.config.detector
        detector_backend = self.config.resolved_detector_backend
        models_config = self.config.models_config

        if detector == "none":
            return None

        # Build detector kwargs based on detector type
        detector_kwargs: dict[str, Any] = {}

        if detector == "doclayout-yolo":
            detector_kwargs = {
                "model_path": self.config.detector_model_path,
                "confidence_threshold": self.config.confidence_threshold,
                "auto_batch_size": self.config.auto_batch_size,
                "batch_size": self.config.batch_size,
                "target_memory_fraction": self.config.target_memory_fraction,
            }
        elif detector == "mineru-vlm":
            detector_config = models_config.get("detectors", {}).get("mineru-vlm", {})
            default_model = detector_config.get("default_model", "opendatalab/MinerU2.5-2509-1.2B")
            final_model = detector if detector.startswith("opendatalab/") else default_model

            backend_kwargs = self._map_backend("detector", "mineru-vlm", detector_backend)
            logger.debug(
                "MinerU VLM detector: detector=%s, final_model=%s, backend_kwargs=%s",
                detector,
                final_model,
                backend_kwargs,
            )

            sorter = self.config.resolved_sorter
            detector_kwargs = {
                "model": final_model,
                **backend_kwargs,
                "detection_only": (sorter != "mineru-vlm"),
            }
        elif detector == "paddleocr-doclayout-v2":
            detector_kwargs = {}
        elif detector == "mineru-doclayout-yolo":
            detector_kwargs = {}

        logger.info("Creating detector: %s with kwargs: %s", detector, detector_kwargs)
        return create_detector_impl(detector, **detector_kwargs)

    def create_sorter(self) -> Sorter | None:
        """Create sorter instance based on configuration.

        Returns:
            Sorter instance or None if no sorter

        Example:
            >>> sorter = factory.create_sorter()
            >>> sorted_blocks = sorter.sort(blocks, image)
        """
        from .layout.ordering import create_sorter as create_sorter_impl

        sorter = self.config.resolved_sorter
        sorter_backend = self.config.resolved_sorter_backend
        models_config = self.config.models_config

        if not sorter:
            return None

        # Build sorter kwargs based on sorter type
        sorter_kwargs: dict[str, Any] = {}

        if sorter == "olmocr-vlm":
            sorter_config = models_config.get("sorters", {}).get("olmocr-vlm", {})
            default_model = sorter_config.get("default_model", "allenai/olmOCR-7B-0825-FP8")
            final_model = sorter if sorter.startswith("allenai/") else default_model

            backend_kwargs = self._map_backend("sorter", "olmocr-vlm", sorter_backend)
            logger.debug(
                "olmOCR VLM sorter: sorter=%s, final_model=%s, backend_kwargs=%s",
                sorter,
                final_model,
                backend_kwargs,
            )

            sorter_kwargs = {
                "model": final_model,
                **backend_kwargs,
                "use_anchoring": True,
            }
        elif sorter == "mineru-vlm":
            sorter_kwargs = {}
        elif sorter == "mineru-layoutreader":
            sorter_kwargs = {}
        else:
            # For pymupdf, mineru-xycut, paddleocr-doclayout-v2, etc.
            sorter_kwargs = {}

        logger.info("Creating sorter: %s with kwargs: %s", sorter, sorter_kwargs)
        return create_sorter_impl(sorter, **sorter_kwargs)

    def create_recognizer(self) -> Recognizer:
        """Create recognizer instance based on configuration.

        Returns:
            Recognizer instance

        Example:
            >>> recognizer = factory.create_recognizer()
            >>> blocks = recognizer.process_blocks(image, blocks)
        """
        from .recognition import create_recognizer

        recognizer = self.config.recognizer
        recognizer_backend = self.config.resolved_recognizer_backend
        _models_config = self.config.models_config  # Reserved for future use

        # Build recognizer kwargs based on backend type
        recognizer_kwargs: dict[str, Any] = {}

        if recognizer_backend in ["pytorch", "vllm", "sglang"] or recognizer.startswith("PaddlePaddle/"):
            # PaddleOCR-VL recognizer
            backend_kwargs = self._map_backend("recognizer", "paddleocr-vl", recognizer_backend)
            logger.debug("PaddleOCR-VL recognizer: recognizer=%s, backend_kwargs=%s", recognizer, backend_kwargs)

            recognizer_kwargs.update({
                "device": None,  # Auto-detect
                **backend_kwargs,
                "use_layout_detection": False,
                "model": recognizer if recognizer.startswith("PaddlePaddle/") else None,
            })
            # Override backend for factory
            recognizer_backend = "paddleocr-vl"
        elif recognizer_backend in ["openai", "gemini"]:
            recognizer_kwargs.update({
                "cache_dir": self.config.cache_dir,
                "use_cache": self.config.use_cache,
                "model": recognizer,
                "gemini_tier": self.config.gemini_tier,
                "use_async": self.config.use_async,
            })
        else:
            # Default to API-based recognizers
            recognizer_kwargs.update({
                "cache_dir": self.config.cache_dir,
                "use_cache": self.config.use_cache,
                "model": recognizer,
                "gemini_tier": self.config.gemini_tier,
                "use_async": self.config.use_async,
            })

        logger.info("Creating recognizer: %s (backend=%s)", recognizer, recognizer_backend)
        return create_recognizer(recognizer, backend=recognizer_backend, **recognizer_kwargs)

    def create_ray_detector_pool(self, detector_kwargs: dict[str, Any] | None = None) -> RayDetectorPool | None:
        """Create Ray detector pool for multi-GPU parallelization.

        Args:
            detector_kwargs: Additional kwargs passed to detector

        Returns:
            Ray detector pool or None if Ray is not available or not needed

        Example:
            >>> pool = factory.create_ray_detector_pool()
            >>> if pool:
            ...     blocks = pool.detect(image)
        """
        if self._ray_detector_pool is not None:
            return self._ray_detector_pool

        detector_backend = self.config.resolved_detector_backend

        # Only create pool for Ray backends
        if detector_backend not in ["pt-ray", "hf-ray"]:
            return None

        from .distributed import RayDetectorPool, is_ray_available

        if not is_ray_available():
            logger.warning(
                "Ray is not available. Falling back to single-GPU mode. Install Ray with: pip install ray"
            )
            return None

        try:
            kwargs = detector_kwargs or {}
            self._ray_detector_pool = RayDetectorPool(
                detector_name=self.config.detector,
                num_actors=None,  # Auto-detect from GPUs
                num_gpus_per_actor=1.0,
                **kwargs,
            )
            logger.info(
                "Ray detector pool initialized: %d actors",
                self._ray_detector_pool.num_actors,
            )
            return self._ray_detector_pool
        except Exception as e:
            logger.warning("Failed to initialize Ray detector pool: %s. Falling back to single-GPU.", e)
            return None

    def create_ray_recognizer_pool(self, recognizer_kwargs: dict[str, Any] | None = None) -> RayRecognizerPool | None:
        """Create Ray recognizer pool for multi-GPU parallelization.

        Args:
            recognizer_kwargs: Additional kwargs passed to recognizer

        Returns:
            Ray recognizer pool or None if Ray is not available or not needed

        Example:
            >>> pool = factory.create_ray_recognizer_pool()
            >>> if pool:
            ...     blocks = pool.recognize_blocks(image, blocks)
        """
        if self._ray_recognizer_pool is not None:
            return self._ray_recognizer_pool

        recognizer_backend = self.config.resolved_recognizer_backend

        # Only create pool for Ray backends
        if recognizer_backend not in ["pt-ray", "hf-ray"]:
            return None

        from .distributed import RayRecognizerPool, is_ray_available

        if not is_ray_available():
            logger.warning(
                "Ray is not available. Falling back to single-GPU mode. Install Ray with: pip install ray"
            )
            return None

        try:
            kwargs = recognizer_kwargs or {}
            self._ray_recognizer_pool = RayRecognizerPool(
                recognizer_name=self.config.recognizer,
                num_actors=None,  # Auto-detect from GPUs
                num_gpus_per_actor=1.0,
                backend=recognizer_backend,
                **kwargs,
            )
            logger.info(
                "Ray recognizer pool initialized: %d actors",
                self._ray_recognizer_pool.num_actors,
            )
            return self._ray_recognizer_pool
        except Exception as e:
            logger.warning("Failed to initialize Ray recognizer pool: %s. Using single-GPU.", e)
            return None

    @property
    def ray_detector_pool(self) -> RayDetectorPool | None:
        """Get cached Ray detector pool."""
        return self._ray_detector_pool

    @property
    def ray_recognizer_pool(self) -> RayRecognizerPool | None:
        """Get cached Ray recognizer pool."""
        return self._ray_recognizer_pool

    def initialize_rate_limiter(self) -> None:
        """Initialize rate limiter for Gemini API.

        This should be called after configuration validation if using Gemini backend.
        """
        if self.config.resolved_recognizer_backend == "gemini":
            from .recognition.api.ratelimit import rate_limiter

            rate_limiter.set_tier_and_model(self.config.gemini_tier, self.config.recognizer)
            logger.info(
                "Rate limiter initialized: tier=%s, model=%s",
                self.config.gemini_tier,
                self.config.recognizer,
            )

    def setup_directories(self) -> None:
        """Create necessary directories for pipeline operation."""
        for directory in [self.config.cache_dir, self.config.output_dir, self.config.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "Directories created: cache=%s, output=%s, temp=%s",
            self.config.cache_dir,
            self.config.output_dir,
            self.config.temp_dir,
        )

