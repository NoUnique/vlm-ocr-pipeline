"""Unified VLM OCR Pipeline for document processing and text extraction."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

# Load environment variables if not already loaded
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .conversion import DocumentConverter
from .layout.detection import LayoutDetector
from .layout.ordering import ReadingOrderAnalyzer
from .misc import tz_now
from .recognition import TextRecognizer
from .recognition.api.ratelimit import rate_limiter

logger = logging.getLogger(__name__)

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    fitz = None  # type: ignore


class Pipeline:
    """Unified VLM OCR processing pipeline with integrated text correction.
    
    This pipeline orchestrates four main stages:
    1. Document Conversion: Convert PDFs/images to processable format
    2. Layout Detection: Identify regions (text, tables, figures, etc.)
    3. Layout Analysis: Determine reading order of regions
    4. Recognition: Extract and correct text from regions
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        use_cache: bool = True,
        cache_dir: str | Path = ".cache",
        output_dir: str | Path = "output",
        temp_dir: str | Path = ".tmp",
        backend: str = "openai",
        model: str = "gemini-2.5-flash",
        gemini_tier: str = "free",
        enable_multi_column_ordering: bool = False,
    ):
        """Initialize VLM OCR processing pipeline.

        Args:
            model_path: DocLayout-YOLO model path
            confidence_threshold: Detection confidence threshold
            use_cache: Whether to use caching
            cache_dir: Cache directory path
            output_dir: Output directory path
            temp_dir: Temporary files directory path
            backend: Backend API to use ("openai" or "gemini")
            model: Model to use for text processing
            gemini_tier: Gemini API tier for rate limiting (only used with gemini backend)
            enable_multi_column_ordering: Whether to align reading order using multi-column detection
        """
        self.backend = backend.lower()
        self.model = model
        self.gemini_tier = gemini_tier
        self.enable_multi_column_ordering = enable_multi_column_ordering

        if self.enable_multi_column_ordering and fitz is None:
            logger.warning(
                "PyMuPDF is not installed. Disabling multi-column ordering for this run."
            )
            self.enable_multi_column_ordering = False

        # Initialize rate limiter (only for Gemini backend)
        if self.backend == "gemini":
            rate_limiter.set_tier_and_model(gemini_tier, model)

        # Convert paths to Path objects
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)

        # Create directories
        self._setup_directories()

        # Initialize modular components
        self.converter = DocumentConverter(temp_dir=self.temp_dir)
        self.detector = LayoutDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
        self.analyzer = ReadingOrderAnalyzer(
            enable_multi_column=self.enable_multi_column_ordering,
        )
        self.recognizer = TextRecognizer(
            cache_dir=self.cache_dir,
            use_cache=use_cache,
            backend=backend,
            model=model,
            gemini_tier=gemini_tier,
        )

        logger.info("Pipeline initialized: %s (model=%s)", self.backend.upper(), self.model)

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        for directory in [self.cache_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_pdf_output_dir(self, pdf_path: Path) -> Path:
        """Return the output directory for a given PDF as <output>/<model>/<file_stem>."""
        return self.output_dir / self.model / pdf_path.stem

    def process_image(
        self,
        image_path: str | Path,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
        pages: list[int] | None = None,
    ) -> dict[str, Any]:
        """Process single image or PDF.
        
        Args:
            image_path: Path to image or PDF file
            max_pages: Maximum number of pages to process (PDF only)
            page_range: Range of pages to process (PDF only)
            pages: Specific pages to process (PDF only)
            
        Returns:
            Processing results
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if image_path.suffix.lower() == ".pdf":
            return self.process_pdf(image_path, max_pages, page_range, pages)
        else:
            return self.process_single_image(image_path)

    def process_single_image(self, image_path: Path) -> dict[str, Any]:
        """Process a single image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processing results with regions and extracted text
        """
        logger.info("Processing image: %s", image_path)

        # Stage 1: Load image
        image_np = self.converter.load_image(image_path)

        # Stage 2: Detect layout regions
        regions = self.detector.detect(image_np)

        # Stage 3: Reading order analysis (simple top-to-bottom for single images)
        # Note: Multi-column ordering is typically not used for single images

        # Stage 4: Recognition - extract text from regions
        processed_regions = self.recognizer.process_regions(image_np, regions)

        # Correct text for text regions
        for region in processed_regions:
            if region["type"] in ["plain text", "title", "list"] and "text" in region:
                region["corrected_text"] = self.recognizer.correct_text(region["text"])

        result = {
            "image_path": str(image_path),
            "regions": processed_regions,
            "processed_at": tz_now().isoformat(),
        }

        return result

    def process_pdf(
        self,
        pdf_path: Path,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
        pages: list[int] | None = None,
    ) -> dict[str, Any]:
        """Process PDF file with page limiting options.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process
            page_range: Range of pages to process (start, end)
            pages: Specific list of page numbers to process
            
        Returns:
            Summary of PDF processing results
        """
        logger.info("Processing PDF: %s", pdf_path)

        # Get PDF info
        pdf_info = self.converter.get_pdf_info(pdf_path)
        total_pages = pdf_info["Pages"]

        # Determine which pages to process
        pages_to_process = self.converter.determine_pages_to_process(
            total_pages, max_pages, page_range, pages
        )

        logger.info("Processing %d pages: %s", len(pages_to_process), pages_to_process)

        output_dir = self._get_pdf_output_dir(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_pages, processing_stopped = self._process_pdf_pages(
            pdf_path, pages_to_process, total_pages, output_dir
        )

        summary = self._create_pdf_summary(
            pdf_path, total_pages, processed_pages, processing_stopped, output_dir
        )

        logger.info("PDF processing complete: %s -> %s", pdf_path, output_dir)

        return summary

    def _process_pdf_pages(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        total_pages: int,
        page_output_dir: Path,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Process multiple PDF pages."""
        processed_pages: list[dict[str, Any]] = []
        processing_stopped = False

        pymupdf_doc = self.converter.open_pymupdf_document(pdf_path) if self.enable_multi_column_ordering else None
        disable_multi_column = False

        for page_num in pages_to_process:
            logger.info("Processing page %d/%d", page_num, total_pages)
            pymupdf_page = None
            if pymupdf_doc is not None and not disable_multi_column:
                try:
                    pymupdf_page = pymupdf_doc.load_page(page_num - 1)
                except Exception as exc:  # pragma: no cover - PyMuPDF failure path
                    logger.warning(
                        "Failed to load page %d with PyMuPDF for multi-column ordering: %s",
                        page_num,
                        exc,
                    )
                    disable_multi_column = True
                    pymupdf_page = None

            page_result, stop_processing = self._process_pdf_page(
                pdf_path,
                page_num,
                page_output_dir,
                pymupdf_page,
            )

            if page_result is not None:
                processed_pages.append(page_result)

            if stop_processing:
                processing_stopped = True
                break

        if pymupdf_doc is not None:
            pymupdf_doc.close()

        return processed_pages, processing_stopped

    def _process_pdf_page(
        self,
        pdf_path: Path,
        page_num: int,
        page_output_dir: Path,
        pymupdf_page: Any | None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Process a single PDF page through the entire pipeline."""
        page_image = None

        try:
            # Stage 1: Convert PDF page to image
            page_image, temp_image_path = self.converter.render_pdf_page(pdf_path, page_num)

            # Stage 2: Detect layout regions
            regions = self.detector.detect(page_image)

            # Stage 3: Analyze reading order
            sorted_regions, column_layout = self.analyzer.analyze_reading_order(
                regions, page_image, pymupdf_page
            )

            # Stage 4: Recognition - extract text from regions
            processed_regions = self.recognizer.process_regions(page_image, sorted_regions)

            # Check for rate limit errors
            if self._check_for_rate_limit_errors({"regions": processed_regions}):
                logger.warning("Rate limit detected on page %d. Stopping processing.", page_num)
                return None, True

            # Compose page-level text
            raw_text = self.analyzer.compose_page_text(processed_regions)
            
            # Correct text
            corrected_text, correction_confidence, stop_due_to_correction = self._perform_page_correction(
                raw_text, page_num
            )

            if stop_due_to_correction:
                return None, True

            # Build result
            page_result = self._build_page_result(
                pdf_path,
                page_num,
                page_image,
                sorted_regions,
                processed_regions,
                raw_text,
                corrected_text,
                correction_confidence,
                column_layout,
            )

            self._save_page_output(page_output_dir, page_num, page_result)

            return page_result, False

        except Exception as e:
            logger.error("Error processing page %d: %s", page_num, e)
            error_page_result = {
                "page_number": page_num,
                "error": str(e),
                "processed_at": tz_now().isoformat(),
            }
            return error_page_result, False

        finally:
            if page_image is not None:
                del page_image
            gc.collect()

    def _perform_page_correction(
        self, raw_text: str, page_num: int
    ) -> tuple[str, float, bool]:
        """Perform page-level text correction."""
        correction_result = self.recognizer.correct_text(raw_text)

        if isinstance(correction_result, dict):
            corrected_text = correction_result.get("corrected_text", raw_text)
            confidence = float(correction_result.get("confidence", 1.0))
            return corrected_text, confidence, False

        corrected_text = str(correction_result)
        rate_limit_indicators = ["RATE_LIMIT_EXCEEDED", "DAILY_LIMIT_EXCEEDED"]
        if any(indicator in corrected_text for indicator in rate_limit_indicators):
            logger.warning(
                "Rate limit detected during page text correction on page %d. Stopping processing.",
                page_num,
            )
            return corrected_text, 1.0, True

        return corrected_text, 1.0, False

    def _build_page_result(
        self,
        pdf_path: Path,
        page_num: int,
        page_image: Any,
        regions: list[dict[str, Any]],
        processed_regions: list[dict[str, Any]],
        raw_text: str,
        corrected_text: str,
        correction_confidence: float,
        column_layout: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build page result dictionary."""
        page_height, page_width = page_image.shape[0], page_image.shape[1]

        page_result = {
            "image_path": str(self.temp_dir / f"{pdf_path.stem}_page_{page_num}.jpg"),
            "width": int(page_width),
            "height": int(page_height),
            "regions": regions,
            "processed_regions": processed_regions,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "correction_confidence": correction_confidence,
            "processed_at": tz_now().isoformat(),
            "page_number": page_num,
        }

        if column_layout is not None:
            page_result["column_layout"] = column_layout

        return page_result

    def _save_page_output(self, page_output_dir: Path, page_num: int, page_result: dict[str, Any]) -> None:
        """Save page processing output."""
        page_output_file = page_output_dir / f"page_{page_num}.json"
        with page_output_file.open("w", encoding="utf-8") as f:
            json.dump(page_result, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", page_output_file)

    def _create_pdf_summary(
        self,
        pdf_path: Path,
        total_pages: int,
        processed_pages: list[dict[str, Any]],
        processing_stopped: bool,
        summary_output_dir: Path,
    ) -> dict[str, Any]:
        """Create PDF processing summary."""
        pages_summary, status_counts = self._build_pages_summary(processed_pages)

        temp_summary_for_errors = {
            "pages_data": processed_pages,
            "processing_stopped": processing_stopped,
        }
        has_errors = self._check_for_any_errors(temp_summary_for_errors)

        summary = {
            "pdf_name": pdf_path.stem,
            "pdf_path": str(pdf_path),
            "num_pages": total_pages,
            "processed_pages": len(processed_pages),
            "output_directory": str(summary_output_dir),
            "processed_at": tz_now().isoformat(),
            "status_summary": {k: v for k, v in status_counts.items() if v > 0},
            "pages": pages_summary,
        }

        summary_filename = self._determine_summary_filename(processing_stopped, has_errors)
        summary_output_file = summary_output_dir / summary_filename
        with summary_output_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", summary_output_file)

        return summary

    def _build_pages_summary(
        self, processed_pages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Build summary of processed pages."""
        pages_summary: list[dict[str, Any]] = []
        status_counts = {"complete": 0, "partial": 0, "incomplete": 0}

        for page_result in processed_pages:
            page_no = int(page_result.get("page_number", 0))
            status = "partial" if page_result.get("error") else "complete"
            status_counts[status] += 1
            pages_summary.append(
                {"page": page_no, "status": status, "file_suffix": "" if status == "complete" else status}
            )

        return pages_summary, status_counts

    def _determine_summary_filename(self, processing_stopped: bool, has_errors: bool) -> str:
        """Determine summary filename based on processing status."""
        if processing_stopped:
            return "summary_incomplete.json"
        if has_errors:
            return "summary_partial.json"
        return "summary.json"

    def _check_for_rate_limit_errors(self, page_result: dict[str, Any]) -> bool:
        """Check if page result contains rate limit errors."""
        try:
            # Check in regions
            regions = page_result.get("regions", [])
            if isinstance(regions, list):
                for region in regions:
                    if isinstance(region, dict) and region.get("error") in ["gemini_rate_limit", "rate_limit_daily"]:
                        return True

            # Check in corrected_text (can be string or dict)
            corrected_text = page_result.get("corrected_text", "")
            if isinstance(corrected_text, dict) and corrected_text.get("error") in [
                "gemini_rate_limit",
                "rate_limit_daily",
            ]:
                return True
            elif isinstance(corrected_text, str) and any(
                error_indicator in corrected_text
                for error_indicator in [
                    "RATE_LIMIT_EXCEEDED",
                    "TEXT_CORRECTION_RATE_LIMIT_EXCEEDED",
                    "DAILY_LIMIT_EXCEEDED",
                    "TEXT_CORRECTION_DAILY_LIMIT_EXCEEDED",
                ]
            ):
                return True
        except (AttributeError, TypeError) as e:
            logger.debug("Error checking rate limit errors: %s", e)

        return False

    def _check_for_any_errors(self, summary: dict[str, Any]) -> bool:
        """Check if summary contains any processing errors."""
        try:
            pages_data = summary.get("pages_data", [])
            for page_result in pages_data:
                # Check for page-level errors
                if page_result.get("error"):
                    return True

                # Check for region-level errors
                regions = page_result.get("regions", [])
                for region in regions:
                    if isinstance(region, dict) and region.get("error"):
                        return True

                # Check for text correction errors
                corrected_text = page_result.get("corrected_text", "")
                if isinstance(corrected_text, dict) and corrected_text.get("error"):
                    return True
                elif isinstance(corrected_text, str) and any(
                    error_indicator in corrected_text
                    for error_indicator in [
                        "[RATE_LIMIT_EXCEEDED]",
                        "[TEXT_CORRECTION_RATE_LIMIT_EXCEEDED]",
                        "[TEXT_CORRECTION_SERVICE_UNAVAILABLE]",
                        "[TEXT_CORRECTION_FAILED]",
                        "[GEMINI_EXTRACTION_FAILED]",
                        "[VISION_API_FAILED]",
                        "[VISION_API_NOT_INITIALIZED]",
                        "[DAILY_LIMIT_EXCEEDED]",
                        "[TEXT_CORRECTION_DAILY_LIMIT_EXCEEDED]",
                    ]
                ):
                    return True

                # Check for processing_stopped flag
                if summary.get("processing_stopped", False):
                    return True

        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("Error checking for processing errors: %s", e)

        return False

    def process_directory(
        self,
        input_dir: Path,
        output_dir: str,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
        specific_pages: list[int] | None = None,
    ) -> dict[str, Any]:
        """Process all PDFs in a directory."""
        input_dir = Path(input_dir)
        output_base = Path(output_dir)
        model_base_dir = output_base / self.model

        if not input_dir.exists() or not input_dir.is_dir():
            return {"error": f"Directory not found: {input_dir}"}

        # Find all PDF files in directory
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            return {"error": f"No PDF files found in directory: {input_dir}"}

        logger.info("Found %d PDF files to process", len(pdf_files))

        results = {}
        total_files = len(pdf_files)
        processed_files = 0

        for pdf_file in pdf_files:
            logger.info("Processing PDF %d/%d: %s", processed_files + 1, total_files, pdf_file.name)

            try:
                # Process the PDF (outputs will be placed under <output>/<model>/<file>)
                result = self.process_pdf(pdf_file, max_pages=max_pages, page_range=page_range, pages=specific_pages)

                results[str(pdf_file)] = result
                processed_files += 1

                # Check for processing errors that should stop batch processing
                if result.get("processing_stopped", False):
                    logger.warning(
                        "Processing stopped for %s due to rate limits. Continuing with next file.", pdf_file.name
                    )

            except Exception as e:
                logger.error("Error processing %s: %s", pdf_file, e)
                results[str(pdf_file)] = {"error": str(e), "processed_at": tz_now().isoformat()}

        # Ensure model base directory exists
        model_base_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "input_directory": str(input_dir),
            "output_directory": str(model_base_dir),
            "total_files": total_files,
            "processed_files": processed_files,
            "results": results,
            "processed_at": tz_now().isoformat(),
        }

        # Save directory summary under model-specific directory
        summary_file = model_base_dir / "directory_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Directory processing complete. Summary saved to: %s", summary_file)

        return summary

    def _save_results(self, result: dict[str, Any], output_path: Path) -> None:
        """Save processing results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info("Results saved to: %s", output_path)
