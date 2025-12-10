"""Page dataclass - single page processing result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .block import Block


@dataclass
class Page:
    """Single page processing result.

    Represents the result of processing one page, containing detected blocks
    and associated metadata. This provides type safety for page-level operations.

    Core fields:
    - page_num: Page number (1-indexed)
    - blocks: List of detected Block objects

    Optional metadata fields:
    - image_path: Path to rendered page image
    - auxiliary_info: Additional info (e.g., text spans for markdown conversion)
    - status: Processing status ("completed", "failed", etc.)
    - processed_at: ISO timestamp
    - page_path: Path to saved JSON output
    """

    # Core fields
    page_num: int
    blocks: list[Block]

    # Optional metadata
    image_path: str | None = None
    auxiliary_info: dict[str, Any] | None = None  # Flexible storage for various metadata
    status: str = "completed"
    processed_at: str | None = None
    page_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict.

        Returns:
            Dict with blocks converted to dicts

        Example:
            >>> page = Page(page_num=1, blocks=[...])
            >>> page.to_dict()
            {'page_num': 1, 'blocks': [...], 'status': 'completed'}
        """
        result: dict[str, Any] = {
            "page_num": self.page_num,
            "blocks": [b.to_dict() for b in self.blocks],
        }

        # Add optional fields (only if not None)
        if self.image_path is not None:
            result["image_path"] = self.image_path
        if self.auxiliary_info is not None:
            result["auxiliary_info"] = self.auxiliary_info
        # Always include status for consistency
        result["status"] = self.status
        if self.processed_at is not None:
            result["processed_at"] = self.processed_at
        if self.page_path is not None:
            result["page_path"] = self.page_path

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Page:
        """Create Page from dict.

        Args:
            data: Dictionary with page data

        Returns:
            Page object

        Example:
            >>> data = {"page_num": 1, "blocks": [...]}
            >>> page = Page.from_dict(data)
        """
        from .block import Block

        return cls(
            page_num=data["page_num"],
            blocks=[Block.from_dict(b) for b in data.get("blocks", [])],
            image_path=data.get("image_path"),
            auxiliary_info=data.get("auxiliary_info"),
            status=data.get("status", "completed"),
            processed_at=data.get("processed_at"),
            page_path=data.get("page_path"),
        )
