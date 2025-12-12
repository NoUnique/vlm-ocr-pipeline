"""Output saving utilities for the VLM OCR Pipeline.

This module handles saving intermediate and final results during pipeline processing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.misc import tz_now
from pipeline.types import Block, Page

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class OutputSaver:
    """Handles saving of pipeline processing results.

    This class encapsulates all result-saving logic, including:
    - Intermediate stage results (detection, ordering, recognition, rendering)
    - Final page results
    - Summary files

    Attributes:
        detector_name: Name of the detector used
        sorter_name: Name of the sorter used
        backend: Recognition backend name
        model: Recognition model name
        renderer: Renderer name
    """

    def __init__(
        self,
        detector_name: str,
        sorter_name: str,
        backend: str,
        model: str,
        renderer: str,
    ):
        """Initialize OutputSaver.

        Args:
            detector_name: Name of the detector
            sorter_name: Name of the sorter
            backend: Recognition backend
            model: Recognition model
            renderer: Renderer name
        """
        self.detector_name = detector_name
        self.sorter_name = sorter_name
        self.backend = backend
        self.model = model
        self.renderer = renderer

    def save_intermediate_results(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        page_output_dir: Path,
        detected_blocks: dict[int, list[Block]],
        page_images: dict[int, np.ndarray] | None = None,
        rendered_texts: dict[int, str] | None = None,
        stage: str = "detection",
    ) -> None:
        """Save intermediate results after each stage.

        Args:
            pdf_path: Path to PDF file
            pages_to_process: List of page numbers
            page_output_dir: Output directory
            detected_blocks: Blocks for each page (at current stage)
            page_images: Page images (optional)
            rendered_texts: Rendered markdown texts (optional)
            stage: Current stage name
        """
        json_dir = page_output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)

        # Save each page's current state
        for page_num in pages_to_process:
            self._save_page_state(
                json_dir, page_num, detected_blocks, page_images, rendered_texts
            )

        # Update summary.json with stage progress
        self._save_stage_summary(pdf_path, pages_to_process, page_output_dir, stage)

        logger.info("Saved intermediate results for stage: %s", stage)

    def _save_page_state(
        self,
        json_dir: Path,
        page_num: int,
        detected_blocks: dict[int, list[Block]],
        page_images: dict[int, Any] | None,
        rendered_texts: dict[int, str] | None,
    ) -> None:
        """Save a single page's current state to JSON.

        Args:
            json_dir: Directory for JSON files
            page_num: Page number
            detected_blocks: Detected blocks by page
            page_images: Page images by page
            rendered_texts: Rendered texts by page
        """
        blocks = detected_blocks.get(page_num, [])

        auxiliary_info: dict[str, Any] = {}
        if page_images and page_num in page_images:
            image = page_images[page_num]
            auxiliary_info["width"] = int(image.shape[1])
            auxiliary_info["height"] = int(image.shape[0])

        if rendered_texts and page_num in rendered_texts:
            auxiliary_info["text"] = rendered_texts[page_num]

        page = Page(
            page_num=page_num,
            blocks=blocks,
            auxiliary_info=auxiliary_info if auxiliary_info else None,
            status="in_progress",
            processed_at=tz_now().isoformat(),
        )

        page_json_file = json_dir / f"page_{page_num}.json"
        with page_json_file.open("w", encoding="utf-8") as f:
            json.dump(page.to_dict(), f, ensure_ascii=False, indent=2)

    def _save_stage_summary(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        page_output_dir: Path,
        stage: str,
    ) -> None:
        """Save summary.json with current stage progress.

        Args:
            pdf_path: Path to PDF file
            pages_to_process: List of page numbers
            page_output_dir: Output directory
            stage: Current stage name
        """
        stage_progress = self._build_stage_progress(stage)

        summary = {
            "pdf_name": pdf_path.stem,
            "pdf_path": str(pdf_path),
            "num_pages": len(pages_to_process),
            "processed_pages": len(pages_to_process),
            "detected_by": self.detector_name,
            "ordered_by": self.sorter_name if stage in ["ordering", "recognition", "rendering"] else "pending",
            "recognized_by": f"{self.backend}/{self.model}" if stage in ["recognition", "rendering"] else "pending",
            "rendered_by": self.renderer if stage == "rendering" else "pending",
            "output_directory": str(page_output_dir),
            "processed_at": tz_now().isoformat(),
            "stage_progress": stage_progress,
            "status": f"in_progress ({stage})",
        }

        summary_file = page_output_dir / "summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _build_stage_progress(self, current_stage: str) -> dict[str, str]:
        """Build stage progress dictionary.

        Args:
            current_stage: Name of the current stage

        Returns:
            Dictionary mapping stage names to completion status
        """
        completed_stages = {
            "detection": ["detection", "ordering", "recognition", "rendering"],
            "ordering": ["ordering", "recognition", "rendering"],
            "recognition": ["recognition", "rendering"],
            "rendering": ["rendering"],
        }

        return {
            "input": "complete",
            "detection": "complete" if current_stage in completed_stages.get("detection", []) else "pending",
            "ordering": "complete" if current_stage in completed_stages.get("ordering", []) else "pending",
            "recognition": "complete" if current_stage in completed_stages.get("recognition", []) else "pending",
            "correction": "pending",
            "rendering": "complete" if current_stage == "rendering" else "pending",
        }

    def save_final_results(self, result: dict[str, Any], output_path: Path) -> None:
        """Save final processing results to JSON file.

        Args:
            result: Result dictionary to save
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Saved results to: %s", output_path)

    def build_pages_summary(
        self, processed_pages: list[Page]
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Build summary data from processed pages.

        Args:
            processed_pages: List of processed Page objects

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
            if status == "success":
                status = "complete"
            if status not in status_counts:
                status = "incomplete"
            status_counts[status] += 1
            pages_summary.append({"id": page_no, "status": status})

        return pages_summary, status_counts

    def determine_summary_filename(
        self, processing_stopped: bool, has_errors: bool
    ) -> str:
        """Determine the appropriate summary filename based on processing state.

        Args:
            processing_stopped: Whether processing was stopped early
            has_errors: Whether any errors occurred

        Returns:
            Filename for the summary file
        """
        if processing_stopped:
            return "summary_partial.json"
        elif has_errors:
            return "summary_with_errors.json"
        else:
            return "summary.json"

    def check_for_rate_limit_errors(self, page_result: dict[str, Any]) -> bool:
        """Check if page result contains rate limit errors.

        Args:
            page_result: Page result dictionary

        Returns:
            True if rate limit errors found
        """
        blocks = page_result.get("blocks", [])

        for block in blocks:
            error = block.get("error", "")
            if "rate_limit" in error.lower() or "daily_limit" in error.lower():
                return True

            text = block.get("text", "")
            if "[RATE_LIMIT_EXCEEDED]" in text or "[DAILY_LIMIT_EXCEEDED]" in text:
                return True

        # Check corrected_text as well
        corrected_text = page_result.get("corrected_text", "")
        if corrected_text:
            if "[TEXT_CORRECTION_DAILY_LIMIT_EXCEEDED]" in corrected_text:
                return True
            if "[TEXT_CORRECTION_RATE_LIMIT_EXCEEDED]" in corrected_text:
                return True

        return False

    def check_for_any_errors(self, summary: dict[str, Any]) -> bool:
        """Check if summary contains any errors.

        Args:
            summary: Summary dictionary with pages data

        Returns:
            True if any errors found
        """
        pages = summary.get("pages", [])

        for page in pages:
            # Check page status
            if page.get("status") in ["failed", "partial"]:
                return True

            # Check blocks for errors
            blocks = page.get("blocks", [])
            for block in blocks:
                if block.get("error"):
                    return True

                # Check for error markers in text
                text = block.get("text", "")
                error_markers = [
                    "[RATE_LIMIT_EXCEEDED]",
                    "[DAILY_LIMIT_EXCEEDED]",
                    "[GEMINI_API_ERROR]",
                    "[GEMINI_EXTRACTION_FAILED]",
                    "[TEXT_CORRECTION_FAILED]",
                ]
                for marker in error_markers:
                    if marker in text:
                        return True

        return False

