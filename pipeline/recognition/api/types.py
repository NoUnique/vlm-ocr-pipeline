"""Type definitions for VLM API responses.

This module provides TypedDict definitions for API response types,
enabling better type safety and IDE support throughout the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

__all__ = [
    # Text extraction
    "ExtractionResult",
    "ExtractionError",
    # Text correction
    "CorrectionResult",
    # Special content
    "SpecialContentResult",
]


# ============================================================================
# Text Extraction Types
# ============================================================================


class ExtractionResult(TypedDict, total=False):
    """Result from text extraction API call.

    Required fields:
        type: Block type
        xywh: Bounding box as [x, y, width, height]
        text: Extracted text content
        confidence: Detection confidence (0.0-1.0)

    Optional fields (present on error):
        error: Error code (e.g., "gemini_rate_limit", "rate_limit_daily")
        error_message: Human-readable error description
    """

    type: str
    xywh: list[int]
    text: str
    confidence: float
    error: str
    error_message: str


class ExtractionError(TypedDict):
    """Error response from text extraction.

    All fields are required when an error occurs.
    """

    type: str
    xywh: list[int]
    text: str  # Usually error placeholder like "[RATE_LIMIT_EXCEEDED]"
    confidence: float  # Usually 0.0 on error
    error: str
    error_message: str


# ============================================================================
# Text Correction Types
# ============================================================================


class CorrectionResult(TypedDict, total=False):
    """Result from text correction API call.

    Required fields:
        corrected_text: The corrected text
        correction_ratio: How much was changed (0.0 = no change, 1.0 = completely different)

    Optional fields (present on error):
        error: Error code
        error_message: Human-readable error description
    """

    corrected_text: str
    correction_ratio: float
    error: str
    error_message: str


@dataclass
class TextCorrectionResult:
    """Dataclass version of correction result for type-safe access.

    Use this when you need attribute access instead of dict access.

    Example:
        >>> result = TextCorrectionResult(corrected_text="fixed", correction_ratio=0.1)
        >>> print(result.corrected_text)
        'fixed'
    """

    corrected_text: str
    correction_ratio: float
    error: str | None = None
    error_message: str | None = None

    def to_dict(self) -> CorrectionResult:
        """Convert to TypedDict format."""
        result: CorrectionResult = {
            "corrected_text": self.corrected_text,
            "correction_ratio": self.correction_ratio,
        }
        if self.error is not None:
            result["error"] = self.error
        if self.error_message is not None:
            result["error_message"] = self.error_message
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextCorrectionResult:
        """Create from dict."""
        return cls(
            corrected_text=data.get("corrected_text", ""),
            correction_ratio=data.get("correction_ratio", 0.0),
            error=data.get("error"),
            error_message=data.get("error_message"),
        )


# ============================================================================
# Special Content Types
# ============================================================================


class SpecialContentResult(TypedDict, total=False):
    """Result from special content (table/figure) processing.

    Required fields:
        type: Block type ("table", "figure", etc.)
        xywh: Bounding box as [x, y, width, height]
        content: Main content (markdown table, description, etc.)
        analysis: Analysis or summary
        confidence: Detection confidence (0.0-1.0)

    Optional fields:
        educational_value: Educational context
        related_topics: Related topics list
        exam_relevance: Exam relevance notes
        error: Error code
        error_message: Error description
    """

    type: str
    xywh: list[int]
    content: str
    analysis: str
    confidence: float
    educational_value: str
    related_topics: list[str]
    exam_relevance: str
    error: str
    error_message: str


# ============================================================================
# Helper Functions
# ============================================================================


def create_extraction_error(
    region_info: dict[str, Any],
    error_text: str,
    error_code: str,
    error_message: str,
) -> ExtractionError:
    """Create a standardized extraction error response.

    Args:
        region_info: Original region info with type and xywh
        error_text: Text to show in place of extracted text
        error_code: Machine-readable error code
        error_message: Human-readable error description

    Returns:
        ExtractionError dict

    Example:
        >>> error = create_extraction_error(
        ...     {"type": "text", "xywh": [0, 0, 100, 50]},
        ...     "[RATE_LIMIT_EXCEEDED]",
        ...     "gemini_rate_limit",
        ...     "API rate limit exceeded"
        ... )
    """
    return ExtractionError(
        type=region_info.get("type", "unknown"),
        xywh=region_info.get("xywh", [0, 0, 0, 0]),
        text=error_text,
        confidence=0.0,
        error=error_code,
        error_message=error_message,
    )


def create_correction_error(
    original_text: str,
    error_code: str,
    error_message: str,
) -> CorrectionResult:
    """Create a standardized correction error response.

    Returns the original text unchanged with error info.

    Args:
        original_text: Original text that failed to be corrected
        error_code: Machine-readable error code
        error_message: Human-readable error description

    Returns:
        CorrectionResult dict with error info
    """
    return CorrectionResult(
        corrected_text=original_text,
        correction_ratio=0.0,
        error=error_code,
        error_message=error_message,
    )

