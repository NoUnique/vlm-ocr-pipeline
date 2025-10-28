"""Staged batch processor for efficient multi-file processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.batch.types import BatchProgress, PageInfo
from pipeline.misc import tz_now

if TYPE_CHECKING:
    from pipeline import Pipeline
    from pipeline.types import Document, Page

logger = logging.getLogger(__name__)


class StagedBatchProcessor:
    """Staged batch processor that processes all files through each stage sequentially.

    Architecture:
        Stage 1: Convert all PDFs to images
        Stage 2: Detect all pages (detector loaded once)
        Stage 3: Sort all pages
        Stage 4: Recognize all pages (recognizer loaded once)
        Stage 5: Save all results

    This maximizes GPU utilization by loading models once per stage.

    Example:
        >>> from pipeline import Pipeline
        >>> from pipeline.batch import StagedBatchProcessor
        >>>
        >>> pipeline = Pipeline()
        >>> processor = StagedBatchProcessor(pipeline)
        >>> results = processor.process_directory("input/", "output/")
    """

    def __init__(self, pipeline: Pipeline):
        """Initialize staged batch processor.

        Args:
            pipeline: Pipeline instance with configured detector/recognizer
        """
        self.pipeline = pipeline
        self.progress: BatchProgress | None = None

    def process_directory(
        self,
        input_dir: Path,
        output_dir: str,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
        specific_pages: list[int] | None = None,
    ) -> dict[str, Any]:
        """Process all PDFs in directory using staged batch processing.

        Args:
            input_dir: Directory containing PDF files
            output_dir: Output directory for results
            max_pages: Maximum pages per PDF to process
            page_range: Page range per PDF to process
            specific_pages: Specific pages per PDF to process

        Returns:
            Dictionary with processing results and summary
        """
        input_dir = Path(input_dir)
        output_base = Path(output_dir)
        model_base_dir = output_base / self.pipeline.model

        if not input_dir.exists() or not input_dir.is_dir():
            return {"error": f"Directory not found: {input_dir}"}

        # Find all PDF files
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            return {"error": f"No PDF files found in directory: {input_dir}"}

        logger.info("Found %d PDF files for staged batch processing", len(pdf_files))

        # Stage 0: Initialize page list
        pages = self._initialize_pages(pdf_files, max_pages, page_range, specific_pages)

        # Initialize progress tracking
        self.progress = BatchProgress(total_pages=len(pages))

        # Stage 1: Convert all PDFs to images
        pages = self._stage_1_conversion(pages)

        # Stage 2: Detect all pages
        pages = self._stage_2_detection(pages)

        # Stage 3: Order all pages
        pages = self._stage_3_ordering(pages)

        # Stage 4: Recognize all pages
        pages = self._stage_4_recognition(pages)

        # Stage 5: Save results
        results = self._stage_5_output(pages, model_base_dir)

        # Generate summary
        summary = self._generate_summary(input_dir, model_base_dir, pages, results)

        logger.info("Staged batch processing complete: %d pages processed", len(pages))

        return summary

    def _initialize_pages(
        self,
        pdf_files: list[Path],
        max_pages: int | None,
        page_range: tuple[int, int] | None,
        specific_pages: list[int] | None,
    ) -> list[PageInfo]:
        """Initialize page list from PDF files.

        Args:
            pdf_files: List of PDF file paths
            max_pages: Maximum pages per PDF
            page_range: Page range per PDF
            specific_pages: Specific pages per PDF

        Returns:
            List of PageInfo objects
        """
        from pipeline.conversion.input import pdf as pdf_converter

        pages: list[PageInfo] = []

        for pdf_file in pdf_files:
            # Get PDF info
            pdf_info = pdf_converter.get_pdf_info(pdf_file)
            total_pages = pdf_info["Pages"]

            # Determine which pages to process
            pages_to_process = pdf_converter.determine_pages_to_process(
                total_pages, max_pages, page_range, specific_pages
            )

            # Create PageInfo for each page
            for page_num in pages_to_process:
                page_info = PageInfo(
                    pdf_path=pdf_file,
                    page_num=page_num,
                    total_pages=total_pages,
                )
                pages.append(page_info)

        logger.info("Initialized %d pages from %d PDF files", len(pages), len(pdf_files))
        return pages

    def _stage_1_conversion(self, pages: list[PageInfo]) -> list[PageInfo]:
        """Stage 1: Convert all PDFs to images.

        Args:
            pages: List of page info objects

        Returns:
            Updated list with images loaded
        """
        logger.info("Stage 1: Converting %d pages to images", len(pages))

        if self.progress:
            self.progress.update(1, "Conversion", 0, 0)

        completed = 0
        failed = 0

        for page_info in pages:
            try:
                # Load PDF page as image
                image = self.pipeline.input_stage.load_pdf_page(
                    page_info.pdf_path,
                    page_info.page_num,
                )
                page_info.image = image

                # Extract auxiliary info
                auxiliary_info = self.pipeline.input_stage.extract_auxiliary_info(
                    page_info.pdf_path,
                    page_info.page_num,
                )
                if auxiliary_info:
                    page_info.auxiliary_info = auxiliary_info

                completed += 1

            except Exception as e:
                logger.error("Failed to convert %s: %s", page_info.page_id, e)
                page_info.mark_failed(f"Conversion error: {e}")
                failed += 1

            # Update progress
            if self.progress:
                self.progress.update(1, "Conversion", completed, failed)

        logger.info("Stage 1 complete: %d/%d pages converted", completed, len(pages))
        return pages

    def _stage_2_detection(self, pages: list[PageInfo]) -> list[PageInfo]:
        """Stage 2: Detect blocks in all pages.

        Args:
            pages: List of page info objects with images loaded

        Returns:
            Updated list with blocks detected
        """
        logger.info("Stage 2: Detecting blocks in %d pages", len(pages))

        if self.progress:
            self.progress.update(2, "Detection", 0, 0)

        # Filter pages with images (exclude failed conversions)
        valid_pages = [p for p in pages if p.image is not None and p.status != "failed"]

        # Use Ray pool if available for batch detection
        if self.pipeline.ray_detector_pool:
            logger.info("Using Ray detector pool for batch detection")
            images = [p.image for p in valid_pages]
            all_blocks = self.pipeline.ray_detector_pool.detect_batch(images)

            for page_info, blocks in zip(valid_pages, all_blocks, strict=False):
                page_info.blocks = blocks

        else:
            # Sequential detection
            for page_info in valid_pages:
                try:
                    blocks = self.pipeline.detection_stage.detect(page_info.image)  # type: ignore
                    page_info.blocks = blocks
                except Exception as e:
                    logger.error("Failed to detect %s: %s", page_info.page_id, e)
                    page_info.mark_failed(f"Detection error: {e}")

        completed = sum(1 for p in valid_pages if p.blocks is not None)
        failed = sum(1 for p in valid_pages if p.status == "failed")

        if self.progress:
            self.progress.update(2, "Detection", completed, failed)

        logger.info("Stage 2 complete: %d/%d pages detected", completed, len(valid_pages))
        return pages

    def _stage_3_ordering(self, pages: list[PageInfo]) -> list[PageInfo]:
        """Stage 3: Order blocks in all pages.

        Args:
            pages: List of page info objects with blocks detected

        Returns:
            Updated list with blocks ordered
        """
        logger.info("Stage 3: Ordering blocks in %d pages", len(pages))

        if self.progress:
            self.progress.update(3, "Ordering", 0, 0)

        # Filter pages with blocks
        valid_pages = [p for p in pages if p.blocks is not None and p.status != "failed"]

        for page_info in valid_pages:
            try:
                sorted_blocks = self.pipeline.ordering_stage.sort(
                    page_info.blocks,  # type: ignore
                    page_info.image,  # type: ignore
                )
                page_info.sorted_blocks = sorted_blocks
            except Exception as e:
                logger.error("Failed to order %s: %s", page_info.page_id, e)
                page_info.mark_failed(f"Ordering error: {e}")

        completed = sum(1 for p in valid_pages if p.sorted_blocks is not None)
        failed = sum(1 for p in valid_pages if p.status == "failed")

        if self.progress:
            self.progress.update(3, "Ordering", completed, failed)

        logger.info("Stage 3 complete: %d/%d pages ordered", completed, len(valid_pages))
        return pages

    def _stage_4_recognition(self, pages: list[PageInfo]) -> list[PageInfo]:
        """Stage 4: Recognize text in all pages.

        Args:
            pages: List of page info objects with ordered blocks

        Returns:
            Updated list with text recognized
        """
        logger.info("Stage 4: Recognizing text in %d pages", len(pages))

        if self.progress:
            self.progress.update(4, "Recognition", 0, 0)

        # Filter pages with sorted blocks
        valid_pages = [p for p in pages if p.sorted_blocks is not None and p.status != "failed"]

        # Use Ray pool if available for batch recognition
        if self.pipeline.ray_recognizer_pool:
            logger.info("Using Ray recognizer pool for batch recognition")
            images = [p.image for p in valid_pages]
            blocks_list = [p.sorted_blocks for p in valid_pages]
            all_recognized = self.pipeline.ray_recognizer_pool.recognize_blocks_batch(
                images,
                blocks_list,  # type: ignore
            )

            for page_info, recognized_blocks in zip(valid_pages, all_recognized, strict=False):
                page_info.recognized_blocks = recognized_blocks

        else:
            # Sequential recognition
            for page_info in valid_pages:
                try:
                    recognized_blocks = self.pipeline.recognition_stage.recognize_blocks(
                        page_info.sorted_blocks,  # type: ignore
                        page_info.image,  # type: ignore
                    )
                    page_info.recognized_blocks = recognized_blocks
                except Exception as e:
                    logger.error("Failed to recognize %s: %s", page_info.page_id, e)
                    page_info.mark_failed(f"Recognition error: {e}")

        completed = sum(1 for p in valid_pages if p.recognized_blocks is not None)
        failed = sum(1 for p in valid_pages if p.status == "failed")

        if self.progress:
            self.progress.update(4, "Recognition", completed, failed)

        logger.info("Stage 4 complete: %d/%d pages recognized", completed, len(valid_pages))
        return pages

    def _stage_5_output(self, pages: list[PageInfo], output_dir: Path) -> dict[str, Document]:
        """Stage 5: Generate output for all pages.

        Args:
            pages: List of page info objects with text recognized
            output_dir: Output directory

        Returns:
            Dictionary mapping PDF path to Document object
        """
        logger.info("Stage 5: Generating output for %d pages", len(pages))

        if self.progress:
            self.progress.update(5, "Output", 0, 0)

        # Group pages by PDF
        pdf_pages: dict[str, list[PageInfo]] = {}
        for page_info in pages:
            pdf_key = str(page_info.pdf_path)
            if pdf_key not in pdf_pages:
                pdf_pages[pdf_key] = []
            pdf_pages[pdf_key].append(page_info)

        results: dict[str, Document] = {}

        # Generate output for each PDF
        for pdf_path_str, pdf_page_list in pdf_pages.items():
            pdf_path = Path(pdf_path_str)
            pdf_output_dir = output_dir / pdf_path.stem
            pdf_output_dir.mkdir(parents=True, exist_ok=True)

            # Convert PageInfo to Page objects
            processed_pages: list[Page] = []
            for page_info in sorted(pdf_page_list, key=lambda p: p.page_num):
                if page_info.recognized_blocks is not None:
                    # Build Page object
                    page = self._build_page_from_page_info(page_info, pdf_output_dir)
                    processed_pages.append(page)

            # Create Document
            document = self._create_document(pdf_path, processed_pages, pdf_output_dir)
            results[pdf_path_str] = document

        completed = sum(len(pdf_pages[pdf]) for pdf in results)
        if self.progress:
            self.progress.update(5, "Output", completed, 0)

        logger.info("Stage 5 complete: %d documents generated", len(results))
        return results

    def _build_page_from_page_info(self, page_info: PageInfo, output_dir: Path) -> Page:
        """Build Page object from PageInfo.

        Args:
            page_info: Page information
            output_dir: Output directory

        Returns:
            Page object
        """
        from pipeline.types import Page

        # Render text
        text = self.pipeline.rendering_stage.render(
            page_info.recognized_blocks,  # type: ignore
            page_info.auxiliary_info,
        )

        # Page correction
        corrected_text, correction_ratio, _ = self.pipeline.page_correction_stage.correct_page(
            text,
            page_info.page_num,
        )

        # Build auxiliary info
        auxiliary_info = page_info.auxiliary_info.copy()
        auxiliary_info.update(
            {
                "text": text,
                "corrected_text": corrected_text,
                "correction_ratio": correction_ratio,
            }
        )

        page = Page(
            page_num=page_info.page_num,
            blocks=page_info.recognized_blocks,  # type: ignore
            auxiliary_info=auxiliary_info,
            status="completed",
            processed_at=tz_now().isoformat(),
        )

        # Save page output
        self.pipeline.output_stage.save_page_output(output_dir, page_info.page_num, page)

        return page

    def _create_document(self, pdf_path: Path, pages: list[Page], output_dir: Path) -> Document:
        """Create Document object from processed pages.

        Args:
            pdf_path: Path to PDF file
            pages: List of processed pages
            output_dir: Output directory

        Returns:
            Document object
        """
        from pipeline.types import Document

        return Document(
            pdf_name=pdf_path.stem,
            pdf_path=str(pdf_path),
            num_pages=max((p.page_num for p in pages), default=0),
            processed_pages=len(pages),
            pages=pages,
            detected_by=self.pipeline.detector_name,
            ordered_by=self.pipeline.sorter_name,
            recognized_by=f"{self.pipeline.backend}/{self.pipeline.model}",
            rendered_by=self.pipeline.renderer,
            output_directory=str(output_dir),
            processed_at=tz_now().isoformat(),
            status_summary={"complete": len(pages)},
        )

    def _generate_summary(
        self,
        input_dir: Path,
        output_dir: Path,
        pages: list[PageInfo],
        results: dict[str, Document],
    ) -> dict[str, Any]:
        """Generate processing summary.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            pages: List of all page info objects
            results: Processing results

        Returns:
            Summary dictionary
        """
        import json

        total_pages = len(pages)
        completed = sum(1 for p in pages if p.status == "completed")
        failed = sum(1 for p in pages if p.status == "failed")

        summary = {
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "total_files": len(results),
            "total_pages": total_pages,
            "completed_pages": completed,
            "failed_pages": failed,
            "success_rate": (completed / total_pages * 100) if total_pages > 0 else 0.0,
            "processing_mode": "staged_batch",
            "results": {str(pdf): doc.to_dict() for pdf, doc in results.items()},
            "processed_at": tz_now().isoformat(),
        }

        # Save summary
        summary_file = output_dir / "batch_summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Batch summary saved to: %s", summary_file)

        return summary
