"""Ray-wrapped detector actors for multi-GPU parallelization.

This module provides Ray actor wrappers for layout detectors, enabling
efficient parallel processing across multiple GPUs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from pipeline.types import Block, Detector

logger = logging.getLogger(__name__)


def create_ray_detector_actor(
    detector_name: str,
    num_gpus: float = 1.0,
    **detector_kwargs: Any,
) -> type:
    """Create a Ray actor class for a detector.

    Args:
        detector_name: Name of the detector to wrap
        num_gpus: Number of GPUs to allocate per actor
        **detector_kwargs: Additional arguments for detector initialization

    Returns:
        Ray actor class wrapping the detector

    Example:
        >>> import ray  # type: ignore[import-not-found]
        >>> DetectorActor = create_ray_detector_actor("doclayout-yolo", num_gpus=1.0)
        >>> actor = DetectorActor.remote()
        >>> result = ray.get(actor.detect.remote(image))
    """
    try:
        import ray  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("Ray is required for distributed processing. Install with: pip install ray") from e

    @ray.remote(num_gpus=num_gpus)
    class DetectorActor:
        """Ray actor wrapping a layout detector.

        This actor initializes a detector once and processes multiple batches,
        avoiding repeated model loading overhead.
        """

        def __init__(self) -> None:
            """Initialize detector actor.

            The detector is loaded once when the actor starts and reused
            for all subsequent detect() calls.
            """
            # Lazy import to avoid loading detector before Ray actor starts
            from pipeline.layout.detection import create_detector

            self.detector: Detector = create_detector(detector_name, **detector_kwargs)
            logger.info("DetectorActor initialized with %s", detector_name)

        def detect(self, image: np.ndarray) -> list[Block]:
            """Detect layout blocks in an image.

            Args:
                image: Input image as numpy array

            Returns:
                List of detected blocks
            """
            return self.detector.detect(image)

        def detect_batch(self, images: list[np.ndarray]) -> list[list[Block]]:
            """Detect layout blocks in a batch of images.

            Args:
                images: List of input images

            Returns:
                List of detection results (one per image)
            """
            return [self.detector.detect(image) for image in images]

        def get_info(self) -> dict[str, Any]:
            """Get detector information.

            Returns:
                Dictionary with detector name and configuration
            """
            return {
                "detector_name": detector_name,
                "detector_kwargs": detector_kwargs,
            }

    return DetectorActor


class RayDetectorPool:
    """Pool of Ray detector actors for load-balanced parallel processing.

    This class manages a pool of detector actors across multiple GPUs,
    automatically distributing work and handling failures.

    Example:
        >>> pool = RayDetectorPool(
        ...     detector_name="doclayout-yolo",
        ...     num_actors=4,  # 4 GPUs
        ...     num_gpus_per_actor=1.0,
        ... )
        >>> results = pool.detect_batch(images)
        >>> pool.shutdown()
    """

    def __init__(
        self,
        detector_name: str,
        num_actors: int | None = None,
        num_gpus_per_actor: float = 1.0,
        **detector_kwargs: Any,
    ):
        """Initialize detector actor pool.

        Args:
            detector_name: Name of the detector to use
            num_actors: Number of actors to create (None = auto-detect from GPUs)
            num_gpus_per_actor: GPUs to allocate per actor
            **detector_kwargs: Additional arguments for detector initialization
        """
        try:
            import ray  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("Ray is required for distributed processing. Install with: pip install ray") from e

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Initialized Ray")

        # Auto-detect number of actors from available GPUs
        if num_actors is None:
            try:
                import torch

                if torch.cuda.is_available():
                    num_actors = torch.cuda.device_count()
                    logger.info("Auto-detected %d GPUs, creating %d actors", num_actors, num_actors)
                else:
                    num_actors = 1
                    logger.warning("No GPUs detected, using 1 CPU actor")
            except ImportError:
                num_actors = 1
                logger.warning("PyTorch not available, using 1 CPU actor")

        self.detector_name = detector_name
        self.num_actors = num_actors
        self.num_gpus_per_actor = num_gpus_per_actor
        self.detector_kwargs = detector_kwargs

        # Create Ray actor class
        DetectorActor = create_ray_detector_actor(detector_name, num_gpus_per_actor, **detector_kwargs)

        # Create actor pool
        self.actors = [DetectorActor.remote() for _ in range(num_actors)]
        logger.info("Created %d detector actors", len(self.actors))

        # Track which actor is currently processing
        self.actor_idx = 0

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks in a single image.

        Args:
            image: Input image as numpy array

        Returns:
            List of detected blocks
        """
        import ray  # type: ignore[import-not-found]

        # Round-robin actor selection
        actor = self.actors[self.actor_idx]
        self.actor_idx = (self.actor_idx + 1) % len(self.actors)

        # Submit task and wait for result
        result_ref = actor.detect.remote(image)
        return ray.get(result_ref)

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Block]]:
        """Detect layout blocks in a batch of images (parallel).

        Args:
            images: List of input images

        Returns:
            List of detection results (one per image)
        """
        import ray  # type: ignore[import-not-found]

        # Distribute images across actors
        tasks = []
        for i, image in enumerate(images):
            actor = self.actors[i % len(self.actors)]
            task = actor.detect.remote(image)
            tasks.append(task)

        # Wait for all results
        return ray.get(tasks)

    def shutdown(self) -> None:
        """Shutdown actor pool and free resources."""
        import ray  # type: ignore[import-not-found]

        for actor in self.actors:
            ray.kill(actor)
        logger.info("Shutdown %d detector actors", len(self.actors))

    def __enter__(self) -> RayDetectorPool:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.shutdown()
