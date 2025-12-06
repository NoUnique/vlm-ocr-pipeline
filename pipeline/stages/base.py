"""Base stage class for pipeline stages.

This module defines the abstract base class for all pipeline stages,
providing a consistent interface and common functionality.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["BaseStage", "StageResult", "StageError"]

# Type variables for generic stage input/output
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class StageError(Exception):
    """Exception raised when stage processing fails."""

    def __init__(self, stage_name: str, message: str, cause: Exception | None = None):
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"[{stage_name}] {message}")


@dataclass
class StageResult(Generic[OutputT]):
    """Result from a pipeline stage.

    Attributes:
        data: The output data from the stage
        stage_name: Name of the stage that produced this result
        processing_time_ms: Time taken to process in milliseconds
        metadata: Additional metadata from processing
    """

    data: OutputT
    stage_name: str
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def processing_time_sec(self) -> float:
        """Get processing time in seconds."""
        return self.processing_time_ms / 1000.0


class BaseStage(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all pipeline stages.

    All stage implementations should inherit from this class and implement
    the `_process_impl` method. This base class provides:

    - Consistent interface (process, process_batch)
    - Timing and logging
    - Error handling with stage context
    - Cleanup hooks

    Attributes:
        name: Stage name for logging and identification

    Example:
        >>> class MyStage(BaseStage[np.ndarray, list[Block]]):
        ...     name = "my-stage"
        ...
        ...     def _process_impl(self, input_data, **context):
        ...         # Implementation here
        ...         return result
    """

    # Subclasses should override this
    name: str = "base-stage"

    @abstractmethod
    def _process_impl(self, input_data: InputT, **context: Any) -> OutputT:
        """Internal processing implementation.

        Subclasses must implement this method to perform actual processing.

        Args:
            input_data: Input from previous stage
            **context: Additional context (image, pdf_path, page_num, etc.)

        Returns:
            Processed output for next stage
        """

    def process(self, input_data: InputT, **context: Any) -> OutputT:
        """Process input and produce output.

        This method wraps _process_impl with timing, logging, and error handling.

        Args:
            input_data: Input from previous stage
            **context: Additional context

        Returns:
            Processed output for next stage

        Raises:
            StageError: If processing fails
        """
        start_time = time.perf_counter()

        try:
            result = self._process_impl(input_data, **context)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                "%s completed in %.2fms",
                self.name,
                elapsed_ms,
            )

            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "%s failed after %.2fms: %s",
                self.name,
                elapsed_ms,
                e,
            )
            raise StageError(self.name, str(e), cause=e) from e

    def process_with_result(self, input_data: InputT, **context: Any) -> StageResult[OutputT]:
        """Process and return result with metadata.

        Args:
            input_data: Input from previous stage
            **context: Additional context

        Returns:
            StageResult with data and metadata
        """
        start_time = time.perf_counter()

        try:
            result = self._process_impl(input_data, **context)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return StageResult(
                data=result,
                stage_name=self.name,
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            raise StageError(self.name, str(e), cause=e) from e

    def process_batch(
        self,
        inputs: list[InputT],
        **context: Any,
    ) -> list[OutputT]:
        """Process multiple inputs.

        Default implementation processes inputs sequentially.
        Subclasses may override for parallel/batch processing.

        Args:
            inputs: List of inputs
            **context: Additional context

        Returns:
            List of outputs
        """
        return [self.process(input_data, **context) for input_data in inputs]

    def cleanup(self) -> None:
        """Cleanup resources after processing.

        Override in subclasses to release resources, clear caches, etc.
        """

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name!r})"

