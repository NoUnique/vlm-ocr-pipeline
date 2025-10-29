"""Output Stage: Result saving and summary generation."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from pipeline.misc import tz_now
from pipeline.types import Block, ColumnLayout, Document, Page

logger = logging.getLogger(__name__)


class OutputStage:
    """Stage 8: Output - Result saving and summary generation."""

    def __init__(self, temp_dir: Path):
        """Initialize OutputStage.

        Args:
            temp_dir: Temporary directory for intermediate files
        """
        self.temp_dir = temp_dir

    def build_page_result(
        self,
        pdf_path: Path,
        page_num: int,
        page_image: np.ndarray,
        detected_blocks: Sequence[Block],
        processed_blocks: Sequence[Block],
        text: str,
        corrected_text: str,
        correction_ratio: float,
        column_layout: ColumnLayout | None,
        auxiliary_info: dict[str, Any] | None = None,
    ) -> Page:
        """Build Page object from processing results.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number
            page_image: Page image array
            detected_blocks: Detected blocks
            processed_blocks: Blocks with extracted text
            text: Rendered text (markdown or plaintext based on renderer setting)
            corrected_text: VLM-corrected text
            correction_ratio: How much text was changed (0.0 = no change, 1.0 = completely different)
            column_layout: Column layout information
            auxiliary_info: Auxiliary information (e.g., text_spans with font info)

        Returns:
            Page object with all processing results
        """
        page_height, page_width = page_image.shape[0], page_image.shape[1]

        # Build auxiliary_info dict that includes all additional metadata
        full_auxiliary_info = auxiliary_info.copy() if auxiliary_info else {}
        full_auxiliary_info.update(
            {
                "image_path": str(self.temp_dir / f"{pdf_path.stem}_page_{page_num}.jpg"),
                "width": int(page_width),
                "height": int(page_height),
                "text": text,
                "corrected_text": corrected_text,
                "correction_ratio": correction_ratio,
            }
        )

        if column_layout is not None:
            full_auxiliary_info["column_layout"] = column_layout

        page = Page(
            page_num=page_num,
            blocks=list(processed_blocks),  # Use processed_blocks (includes text)
            auxiliary_info=full_auxiliary_info,
            status="completed",
            processed_at=tz_now().isoformat(),
        )

        return page

    def save_page_output(self, page_output_dir: Path, page_num: int, page: Page) -> None:
        """Save page processing output as JSON and Markdown.

        Args:
            page_output_dir: Directory to save page output
            page_num: Page number
            page: Page object to save
        """
        # Create json subdirectory
        json_dir = page_output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON to json/ subdirectory
        page_json_file = json_dir / f"page_{page_num}.json"
        with page_json_file.open("w", encoding="utf-8") as f:
            json.dump(page.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("JSON saved to %s", page_json_file)

        # Save Markdown to main directory (use corrected_text if available, else text)
        page_md_file = page_output_dir / f"page_{page_num}.md"
        auxiliary_info = page.auxiliary_info or {}
        markdown_text = auxiliary_info.get("corrected_text") or auxiliary_info.get("text", "")

        with page_md_file.open("w", encoding="utf-8") as f:
            f.write(markdown_text)
        logger.info("Markdown saved to %s", page_md_file)

    def create_pdf_summary(
        self,
        pdf_path: Path,
        total_pages: int,
        processed_pages: list[Page],
        processing_stopped: bool,
        summary_output_dir: Path,
        detector_name: str,
        sorter_name: str,
        backend: str,
        model: str,
        renderer: str,
    ) -> Document:
        """Create PDF processing summary as Document object.

        Args:
            pdf_path: Path to PDF file
            total_pages: Total number of pages in PDF
            processed_pages: List of Page objects (processed or failed)
            processing_stopped: Whether processing was stopped early
            summary_output_dir: Directory to save summary
            detector_name: Name of detector used
            sorter_name: Name of sorter used
            backend: Backend used (openai, gemini, etc.)
            model: Model name
            renderer: Renderer used

        Returns:
            Document object with full page data
        """
        pages_summary, status_counts = self._build_pages_summary(processed_pages)

        # Check if any pages have errors
        has_errors = any(page.status == "failed" for page in processed_pages)

        # Build stage progress information
        stage_progress = self._build_stage_progress(processed_pages, processing_stopped)

        # Create Document object with full page data
        document = Document(
            pdf_name=pdf_path.stem,
            pdf_path=str(pdf_path),
            num_pages=total_pages,
            processed_pages=len(processed_pages),
            pages=processed_pages,  # Full Page objects
            detected_by=detector_name,
            ordered_by=sorter_name,
            recognized_by=f"{backend}/{model}",
            rendered_by=renderer,
            output_directory=str(summary_output_dir),
            processed_at=tz_now().isoformat(),
            status_summary={k: v for k, v in status_counts.items() if v > 0},
        )

        # Create summary dict for JSON output (subset of Document data)
        summary = {
            "pdf_name": document.pdf_name,
            "pdf_path": document.pdf_path,
            "num_pages": document.num_pages,
            "processed_pages": document.processed_pages,
            "detected_by": document.detected_by,
            "ordered_by": document.ordered_by,
            "recognized_by": document.recognized_by,
            "rendered_by": document.rendered_by,
            "output_directory": document.output_directory,
            "processed_at": document.processed_at,
            "status_summary": document.status_summary,
            "stage_progress": stage_progress,  # Add stage progress information
            "pages": pages_summary,  # Use simplified page summary, not full pages
        }

        summary_filename = self._determine_summary_filename(processing_stopped, has_errors)
        summary_output_file = summary_output_dir / summary_filename
        with summary_output_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", summary_output_file)

        return document

    def _build_pages_summary(self, processed_pages: list[Page]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Build summary of processed pages.

        Args:
            processed_pages: List of Page objects

        Returns:
            Tuple of (pages_summary, status_counts)
        """
        pages_summary: list[dict[str, Any]] = []
        status_counts = {"complete": 0, "partial": 0, "incomplete": 0}

        for page in processed_pages:
            page_no = page.page_num
            status = page.status if page.status != "completed" else "complete"
            if status == "failed":
                status = "partial"
            status_counts[status] += 1
            pages_summary.append({"id": page_no, "status": status})

        return pages_summary, status_counts

    def _build_stage_progress(self, processed_pages: list[Page], processing_stopped: bool) -> dict[str, str]:
        """Build stage progress information.

        Args:
            processed_pages: List of processed pages
            processing_stopped: Whether processing was stopped early

        Returns:
            Dictionary with stage progress status
        """
        if processing_stopped:
            # Determine which stage was incomplete
            return {
                "input": "complete",
                "detection": "complete",
                "ordering": "complete",
                "recognition": "incomplete",
                "correction": "pending",
                "rendering": "pending",
            }

        # All pages processed successfully
        has_blocks = any(len(page.blocks) > 0 for page in processed_pages)
        has_text = any(page.auxiliary_info and page.auxiliary_info.get("text") for page in processed_pages)
        has_corrected_text = any(
            page.auxiliary_info and page.auxiliary_info.get("corrected_text") for page in processed_pages
        )

        stage_progress = {
            "input": "complete",
            "detection": "complete" if has_blocks else "incomplete",
            "ordering": "complete" if has_blocks else "incomplete",
            "recognition": "complete" if has_text else "incomplete",
            "correction": "complete" if has_corrected_text else "incomplete",
            "rendering": "complete" if has_text or has_corrected_text else "incomplete",
        }

        return stage_progress

    def _determine_summary_filename(self, processing_stopped: bool, has_errors: bool) -> str:
        """Determine summary filename based on processing status."""
        if processing_stopped:
            return "summary_incomplete.json"
        if has_errors:
            return "summary_partial.json"
        return "summary.json"

    def save_results(self, result: dict[str, Any], output_path: Path) -> None:
        """Save processing results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
