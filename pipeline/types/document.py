"""Document dataclass - multi-page document processing result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .page import Page


@dataclass
class Document:
    """Multi-page document processing result.

    Represents the complete result of processing a PDF or multi-page document.
    This provides type safety for document-level operations.

    Core fields:
    - pdf_name: Document name (without extension)
    - pdf_path: Full path to source PDF
    - num_pages: Total number of pages in document
    - processed_pages: Number of pages actually processed
    - pages: List of Page objects

    Optional metadata fields:
    - output_directory: Directory where results are saved
    - processed_at: ISO timestamp
    - status_summary: Count of pages by status
    """

    # Core fields
    pdf_name: str
    pdf_path: str
    num_pages: int
    processed_pages: int
    pages: list[Page]

    # Optional metadata
    detected_by: str | None = None
    ordered_by: str | None = None
    recognized_by: str | None = None
    corrected_by: str | None = None
    rendered_by: str | None = None
    output_directory: str | None = None
    processed_at: str | None = None
    status_summary: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "pdf_name": self.pdf_name,
            "pdf_path": self.pdf_path,
            "num_pages": self.num_pages,
            "processed_pages": self.processed_pages,
            "pages": [p.to_dict() for p in self.pages],
        }

        if self.detected_by is not None:
            result["detected_by"] = self.detected_by
        if self.ordered_by is not None:
            result["ordered_by"] = self.ordered_by
        if self.recognized_by is not None:
            result["recognized_by"] = self.recognized_by
        if self.corrected_by is not None:
            result["corrected_by"] = self.corrected_by
        if self.rendered_by is not None:
            result["rendered_by"] = self.rendered_by
        if self.output_directory is not None:
            result["output_directory"] = self.output_directory
        if self.processed_at is not None:
            result["processed_at"] = self.processed_at
        if self.status_summary is not None:
            result["status_summary"] = self.status_summary

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create Document from dict."""
        from .page import Page

        return cls(
            pdf_name=data["pdf_name"],
            pdf_path=data["pdf_path"],
            num_pages=data["num_pages"],
            processed_pages=data["processed_pages"],
            pages=[Page.from_dict(p) for p in data.get("pages", [])],
            detected_by=data.get("detected_by"),
            ordered_by=data.get("ordered_by"),
            recognized_by=data.get("recognized_by"),
            corrected_by=data.get("corrected_by"),
            rendered_by=data.get("rendered_by"),
            output_directory=data.get("output_directory"),
            processed_at=data.get("processed_at"),
            status_summary=data.get("status_summary"),
        )
