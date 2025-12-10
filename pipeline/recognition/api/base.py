"""Base classes for VLM API clients.

This module provides abstract base classes for synchronous and asynchronous
API clients, ensuring consistent interface across different VLM providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class BaseVLMClient(ABC):
    """Abstract base class for VLM API clients.

    This class defines the common interface for all VLM clients (OpenAI, Gemini, etc.).
    Subclasses must implement the abstract methods for their specific API.

    Common responsibilities:
    - API configuration loading
    - Client initialization
    - Text extraction from images
    - Special content processing (tables, figures)
    - Text correction

    Attributes:
        model: The model name/identifier
        client: The underlying API client instance
    """

    # Subclasses should define these class attributes
    PROVIDER_NAME: str = "base"
    DEFAULT_MODEL: str = "default"

    def __init__(self, model: str | None = None):
        """Initialize base client.

        Args:
            model: Model name to use. If None, uses DEFAULT_MODEL.
        """
        self.model = model or self.DEFAULT_MODEL
        self.client: Any = None

    @abstractmethod
    def _load_api_config(self) -> None:
        """Load API configuration from environment or config files.

        This method should set up any necessary configuration like:
        - API keys
        - Base URLs
        - Rate limit settings
        - Other provider-specific settings
        """
        ...

    @abstractmethod
    def _setup_client(self) -> Any:
        """Set up and return the API client instance.

        Returns:
            The initialized API client, or None if setup fails.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the API client is available and ready.

        Returns:
            True if client is initialized and ready to make requests.
        """
        ...

    @abstractmethod
    def extract_text(
        self,
        block_img: np.ndarray,
        block_info: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Extract text from an image block.

        Args:
            block_img: Image block as numpy array (BGR or RGB)
            block_info: Metadata about the block (type, coordinates, etc.)
            prompt: Prompt for text extraction

        Returns:
            Dictionary containing:
            - type: Block type
            - xywh: Bounding box coordinates
            - text: Extracted text
            - confidence: Confidence score
            - error (optional): Error code if extraction failed
            - error_message (optional): Error description
        """
        ...

    @abstractmethod
    def process_special_block(
        self,
        block_img: np.ndarray,
        block_info: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Process special content like tables or figures.

        Args:
            block_img: Image block as numpy array
            block_info: Metadata about the block
            prompt: Prompt for content analysis

        Returns:
            Dictionary containing:
            - type: Block type
            - xywh: Bounding box coordinates
            - content: Processed content
            - analysis: Content analysis
            - confidence: Confidence score
            - error (optional): Error code if processing failed
            - error_message (optional): Error description
        """
        ...

    @abstractmethod
    def correct_text(
        self,
        text: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Correct OCR text using the VLM.

        Args:
            text: Text to correct
            system_prompt: System instruction prompt
            user_prompt: User prompt with text

        Returns:
            Dictionary containing:
            - corrected_text: The corrected text
            - correction_ratio: How much was changed (0.0-1.0)
            - error (optional): Error code if correction failed
            - error_message (optional): Error description
        """
        ...

    def _text_correction_result(
        self,
        corrected_text: str,
        correction_ratio: float,
        error: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Create standardized text correction result.

        Args:
            corrected_text: The corrected text
            correction_ratio: How much was changed (0.0 = no change, 1.0 = completely different)
            error: Optional error code
            error_message: Optional error description

        Returns:
            Standardized correction result dictionary
        """
        result: dict[str, Any] = {
            "corrected_text": corrected_text,
            "correction_ratio": correction_ratio,
        }
        if error:
            result["error"] = error
        if error_message:
            result["error_message"] = error_message
        return result


class BaseAsyncVLMClient(ABC):
    """Abstract base class for async VLM API clients.

    This class defines the common interface for all async VLM clients.
    Subclasses must implement the abstract methods for their specific API.

    Attributes:
        model: The model name/identifier
        client: The underlying async API client instance
    """

    PROVIDER_NAME: str = "base_async"
    DEFAULT_MODEL: str = "default"

    def __init__(self, model: str | None = None):
        """Initialize base async client.

        Args:
            model: Model name to use. If None, uses DEFAULT_MODEL.
        """
        self.model = model or self.DEFAULT_MODEL
        self.client: Any = None

    @abstractmethod
    def _load_api_config(self) -> None:
        """Load API configuration from environment or config files."""
        ...

    @abstractmethod
    def _setup_client(self) -> Any:
        """Set up and return the async API client instance."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the API client is available and ready."""
        ...

    @abstractmethod
    async def extract_text(
        self,
        block_img: np.ndarray,
        block_info: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Extract text from an image block (async).

        Args:
            block_img: Image block as numpy array
            block_info: Metadata about the block
            prompt: Prompt for text extraction

        Returns:
            Dictionary with extracted text and metadata
        """
        ...

    @abstractmethod
    async def extract_text_batch(
        self,
        batch_data: list[tuple[np.ndarray, dict[str, Any], str]],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Extract text from multiple image regions concurrently.

        Args:
            batch_data: List of (image, block_info, prompt) tuples
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of extraction results
        """
        ...

    @abstractmethod
    async def correct_text(
        self,
        text: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Correct OCR text using the VLM (async).

        Args:
            text: Text to correct
            system_prompt: System instruction prompt
            user_prompt: User prompt with text

        Returns:
            Dictionary with corrected text and metadata
        """
        ...

    def _text_correction_result(
        self,
        corrected_text: str,
        correction_ratio: float,
        error: str | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        """Create standardized text correction result.

        Args:
            corrected_text: The corrected text
            correction_ratio: How much was changed
            error: Optional error code
            error_message: Optional error description

        Returns:
            Standardized correction result dictionary
        """
        result: dict[str, Any] = {
            "corrected_text": corrected_text,
            "correction_ratio": correction_ratio,
        }
        if error:
            result["error"] = error
        if error_message:
            result["error_message"] = error_message
        return result


__all__ = ["BaseVLMClient", "BaseAsyncVLMClient"]

