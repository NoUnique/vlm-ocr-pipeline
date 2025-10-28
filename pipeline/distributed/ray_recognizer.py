"""Ray-wrapped recognizer actors for multi-GPU parallelization.

This module provides Ray actor wrappers for text recognizers, enabling
efficient parallel text extraction across multiple GPUs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from pipeline.types import Block, Recognizer

logger = logging.getLogger(__name__)


def create_ray_recognizer_actor(
    recognizer_name: str,
    num_gpus: float = 1.0,
    **recognizer_kwargs: Any,
) -> type:
    """Create a Ray actor class for a recognizer.

    Args:
        recognizer_name: Name of the recognizer to wrap
        num_gpus: Number of GPUs to allocate per actor
        **recognizer_kwargs: Additional arguments for recognizer initialization

    Returns:
        Ray actor class wrapping the recognizer

    Example:
        >>> import ray
        >>> RecognizerActor = create_ray_recognizer_actor("paddleocr-vl", num_gpus=1.0)
        >>> actor = RecognizerActor.remote()
        >>> result = ray.get(actor.recognize_blocks.remote(image, blocks))
    """
    try:
        import ray
    except ImportError as e:
        raise ImportError("Ray is required for distributed processing. Install with: pip install ray") from e

    @ray.remote(num_gpus=num_gpus)
    class RecognizerActor:
        """Ray actor wrapping a text recognizer.

        This actor initializes a recognizer once and processes multiple batches,
        avoiding repeated model loading overhead.
        """

        def __init__(self) -> None:
            """Initialize recognizer actor.

            The recognizer is loaded once when the actor starts and reused
            for all subsequent recognize_blocks() calls.
            """
            # Lazy import to avoid loading recognizer before Ray actor starts
            from pipeline.recognition import create_recognizer

            self.recognizer: Recognizer = create_recognizer(recognizer_name, **recognizer_kwargs)
            logger.info("RecognizerActor initialized with %s", recognizer_name)

        def recognize_blocks(self, image: np.ndarray, blocks: list[Block]) -> list[Block]:
            """Extract text from blocks in an image.

            Args:
                image: Input image as numpy array
                blocks: List of blocks to process

            Returns:
                List of blocks with extracted text
            """
            return self.recognizer.process_blocks(image, blocks)

        def recognize_blocks_batch(
            self,
            images: list[np.ndarray],
            blocks_list: list[list[Block]],
        ) -> list[list[Block]]:
            """Extract text from blocks in a batch of images.

            Args:
                images: List of input images
                blocks_list: List of block lists (one per image)

            Returns:
                List of processed block lists (one per image)
            """
            return [
                self.recognizer.process_blocks(img, blocks)
                for img, blocks in zip(images, blocks_list, strict=False)
            ]

        def get_info(self) -> dict[str, Any]:
            """Get recognizer information.

            Returns:
                Dictionary with recognizer name and configuration
            """
            return {
                "recognizer_name": recognizer_name,
                "recognizer_kwargs": recognizer_kwargs,
            }

    return RecognizerActor


class RayRecognizerPool:
    """Pool of Ray recognizer actors for load-balanced parallel processing.

    This class manages a pool of recognizer actors across multiple GPUs,
    automatically distributing work and handling failures.

    Example:
        >>> pool = RayRecognizerPool(
        ...     recognizer_name="paddleocr-vl",
        ...     num_actors=4,  # 4 GPUs
        ...     num_gpus_per_actor=1.0,
        ... )
        >>> results = pool.recognize_blocks_batch(images, blocks_list)
        >>> pool.shutdown()
    """

    def __init__(
        self,
        recognizer_name: str,
        num_actors: int | None = None,
        num_gpus_per_actor: float = 1.0,
        **recognizer_kwargs: Any,
    ):
        """Initialize recognizer actor pool.

        Args:
            recognizer_name: Name of the recognizer to use
            num_actors: Number of actors to create (None = auto-detect from GPUs)
            num_gpus_per_actor: GPUs to allocate per actor
            **recognizer_kwargs: Additional arguments for recognizer initialization
        """
        try:
            import ray
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

        self.recognizer_name = recognizer_name
        self.num_actors = num_actors
        self.num_gpus_per_actor = num_gpus_per_actor
        self.recognizer_kwargs = recognizer_kwargs

        # Create Ray actor class
        RecognizerActor = create_ray_recognizer_actor(recognizer_name, num_gpus_per_actor, **recognizer_kwargs)

        # Create actor pool
        self.actors = [RecognizerActor.remote() for _ in range(num_actors)]
        logger.info("Created %d recognizer actors", len(self.actors))

        # Track which actor is currently processing
        self.actor_idx = 0

    def recognize_blocks(self, image: np.ndarray, blocks: list[Block]) -> list[Block]:
        """Extract text from blocks in a single image.

        Args:
            image: Input image as numpy array
            blocks: List of blocks to process

        Returns:
            List of blocks with extracted text
        """
        import ray

        # Round-robin actor selection
        actor = self.actors[self.actor_idx]
        self.actor_idx = (self.actor_idx + 1) % len(self.actors)

        # Submit task and wait for result
        result_ref = actor.recognize_blocks.remote(image, blocks)
        return ray.get(result_ref)

    def recognize_blocks_batch(
        self,
        images: list[np.ndarray],
        blocks_list: list[list[Block]],
    ) -> list[list[Block]]:
        """Extract text from blocks in a batch of images (parallel).

        Args:
            images: List of input images
            blocks_list: List of block lists (one per image)

        Returns:
            List of processed block lists (one per image)
        """
        import ray

        # Distribute images across actors
        tasks = []
        for i, (image, blocks) in enumerate(zip(images, blocks_list, strict=False)):
            actor = self.actors[i % len(self.actors)]
            task = actor.recognize_blocks.remote(image, blocks)
            tasks.append(task)

        # Wait for all results
        return ray.get(tasks)

    def shutdown(self) -> None:
        """Shutdown actor pool and free resources."""
        import ray

        for actor in self.actors:
            ray.kill(actor)
        logger.info("Shutdown %d recognizer actors", len(self.actors))

    def __enter__(self) -> RayRecognizerPool:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.shutdown()
