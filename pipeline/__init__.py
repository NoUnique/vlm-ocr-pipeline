"""Unified VLM OCR Pipeline for document processing and text extraction."""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

# Load environment variables if not already loaded
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .constants import DEFAULT_CONFIDENCE_THRESHOLD
from .conversion.output.markdown import page_to_markdown
from .conversion.output.plaintext import blocks_to_plaintext
from .layout.detection import create_detector as create_detector_impl
from .layout.ordering import create_sorter as create_sorter_impl, validate_combination
from .misc import tz_now
from .recognition import TextRecognizer
from .recognition.api.ratelimit import rate_limiter
from .types import Block, Detector, Document, Page, Recognizer, Sorter

logger = logging.getLogger(__name__)


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file with error handling.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dict, or empty dict if file not found or invalid
    """
    try:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.debug("Config file not found: %s", config_path)
        return {}
    except Exception as e:
        logger.warning("Failed to load config file %s: %s", config_path, e)
        return {}


try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    fitz = None  # type: ignore


class Pipeline:
    """Unified VLM OCR processing pipeline with integrated text correction.

    This pipeline orchestrates four main stages:
    1. Document Conversion: Convert PDFs/images to processable format
    2. Layout Detection: Identify blocks (text, tables, figures, etc.)
    3. Layout Analysis: Determine reading order of blocks
    4. Recognition: Extract and correct text from blocks
    """

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        use_cache: bool = True,
        cache_dir: str | Path = ".cache",
        output_dir: str | Path = "output",
        temp_dir: str | Path = ".tmp",
        backend: str = "openai",
        model: str = "gemini-2.5-flash",
        gemini_tier: str = "free",
        enable_multi_column_ordering: bool = False,
        # New options for modular detector/sorter
        detector: str = "doclayout-yolo",
        sorter: str | None = None,
        mineru_model: str | None = None,
        mineru_backend: str = "transformers",
        olmocr_model: str | None = None,
        # Renderer option
        renderer: str = "markdown",
    ):
        """Initialize VLM OCR processing pipeline.

        Args:
            model_path: DocLayout-YOLO model path (used if detector="doclayout-yolo")
            confidence_threshold: Detection confidence threshold (None = load from config)
            use_cache: Whether to use caching
            cache_dir: Cache directory path
            output_dir: Output directory path
            temp_dir: Temporary files directory path
            backend: Backend API to use ("openai" or "gemini")
            model: Model to use for text processing
            gemini_tier: Gemini API tier for rate limiting (only used with gemini backend)
            enable_multi_column_ordering: Legacy option - use sorter="pymupdf" instead
            detector: Detector to use ("doclayout-yolo", "mineru-vlm", "none")
            sorter: Sorter to use (None for auto-selection based on legacy options)
            mineru_model: MinerU model path (for mineru-vlm detector/sorter)
            mineru_backend: MinerU backend ("transformers", "vllm-engine", "vllm-async-engine")
            olmocr_model: olmOCR model path (for olmocr-vlm sorter)
            renderer: Output format renderer ("markdown" or "plaintext")
        """
        # Load configuration files
        models_config = _load_yaml_config(Path("settings") / "models.yaml")
        detection_config = _load_yaml_config(Path("settings") / "detection_config.yaml")

        self.backend = backend.lower()
        # Override model name for paddleocr-vl backend
        if self.backend == "paddleocr-vl":
            self.model = "paddleocr-vl"
        else:
            self.model = model
        self.gemini_tier = gemini_tier
        self.enable_multi_column_ordering = enable_multi_column_ordering
        self.renderer = renderer.lower()

        # Validate renderer
        if self.renderer not in ["markdown", "plaintext"]:
            raise ValueError(f"Invalid renderer: {renderer}. Must be 'markdown' or 'plaintext'.")

        # Import REQUIRED_COMBINATIONS for bidirectional auto-selection
        from .layout.ordering import REQUIRED_COMBINATIONS  # noqa: PLC0415

        # Default detector
        detector_default = "doclayout-yolo"
        detector_is_default = detector == detector_default

        # Bidirectional auto-selection for tightly coupled detector/sorter pairs
        # Case 1: Sorter is specified and requires a specific detector
        if sorter is not None and sorter in REQUIRED_COMBINATIONS:
            required_detector = REQUIRED_COMBINATIONS[sorter]
            if not detector_is_default and detector != required_detector:
                # User explicitly specified incompatible detector
                raise ValueError(
                    f"Sorter '{sorter}' requires detector '{required_detector}' (tightly coupled), "
                    f"but detector '{detector}' was specified. "
                    f"Either omit --detector or use --detector {required_detector}."
                )
            if detector_is_default:
                # Auto-select required detector
                detector = required_detector
                logger.info("Auto-selected detector='%s' for '%s' sorter (tightly coupled)", detector, sorter)

        # Case 2: Sorter is not specified, auto-select based on detector or legacy options
        if sorter is None:
            # Special case: tightly coupled detectors require their own sorter
            if detector == "paddleocr-doclayout-v2":
                sorter = "paddleocr-doclayout-v2"
                logger.info("Auto-selected sorter='paddleocr-doclayout-v2' for paddleocr-doclayout-v2 detector")
            elif detector == "mineru-vlm":
                sorter = "mineru-vlm"
                logger.info("Auto-selected sorter='mineru-vlm' for mineru-vlm detector")
            elif enable_multi_column_ordering:
                sorter = "pymupdf"
                logger.info("Legacy option: enable_multi_column_ordering=True → using sorter='pymupdf'")
            else:
                sorter = "mineru-xycut"
                logger.info("Using default sorter='mineru-xycut' (fast and accurate)")

        # Store detector/sorter choices
        self.detector_name = detector
        self.sorter_name = sorter

        # Validate combination
        is_valid, message = validate_combination(detector, sorter)
        if not is_valid:
            raise ValueError(f"Invalid detector/sorter combination: {message}")

        logger.info("Pipeline combination: %s", message)

        # Check PyMuPDF availability for pymupdf sorter
        if sorter == "pymupdf" and fitz is None:
            logger.warning("PyMuPDF is not installed. Falling back to sorter='mineru-xycut'")
            sorter = "mineru-xycut"
            self.sorter_name = "mineru-xycut"

        # Initialize rate limiter (only for Gemini backend)
        if self.backend == "gemini":
            rate_limiter.set_tier_and_model(gemini_tier, model)

        # Convert paths to Path objects
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)

        # Create directories
        self._setup_directories()

        # Initialize components using new factory system
        # Note: Converter is now function-based, no instance needed

        # Resolve confidence_threshold from config if not explicitly provided
        if confidence_threshold is None:
            # Try to get from detection_config, fall back to constant
            detectors_config = detection_config.get("detectors", {})
            detector_specific_config = detectors_config.get(detector, {})
            confidence_threshold = detector_specific_config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
            logger.debug("Using confidence_threshold=%.2f from config for detector=%s", confidence_threshold, detector)

        # Create detector
        detector_kwargs = {}
        if detector == "doclayout-yolo":
            detector_kwargs = {
                "model_path": model_path,
                "confidence_threshold": confidence_threshold,
            }
        elif detector == "mineru-vlm":
            # Get default model from models_config
            mineru_detector_config = models_config.get("detectors", {}).get("mineru-vlm", {})
            default_model = mineru_detector_config.get("default_model", "opendatalab/MinerU2.5-2509-1.2B")
            final_model = mineru_model if mineru_model is not None else default_model
            logger.debug("MinerU VLM detector: mineru_model=%s, final_model=%s", mineru_model, final_model)
            detector_kwargs = {
                "model": final_model,  # MinerU 2.5 VLM (1.2B)
                "backend": mineru_backend,
                "detection_only": (sorter != "mineru-vlm"),  # Full pipeline if using mineru-vlm sorter
            }

        # Create detector
        self.detector: Detector | None = (
            create_detector_impl(detector, **detector_kwargs) if detector != "none" else None
        )

        # Create sorter
        sorter_kwargs = {}
        if sorter == "olmocr-vlm":
            # Get default model from models_config
            olmocr_sorter_config = models_config.get("sorters", {}).get("olmocr-vlm", {})
            default_olmocr_model = olmocr_sorter_config.get("default_model", "allenai/olmOCR-7B-0825-FP8")
            sorter_kwargs = {
                "model": olmocr_model or default_olmocr_model,
                "use_anchoring": True,
            }

        self.sorter: Sorter | None = create_sorter_impl(sorter, **sorter_kwargs) if sorter else None

        # Recognizer (backend-dependent)
        if backend == "paddleocr-vl":
            # Use PaddleOCR-VL recognizer
            from .recognition import PaddleOCRVLRecognizer  # noqa: PLC0415

            self.recognizer: Recognizer = PaddleOCRVLRecognizer(
                device=None,  # Auto-detect
                vl_rec_backend="native",
                use_layout_detection=False,  # We handle layout detection separately
            )
        else:
            # Use traditional VLM-based recognizer
            self.recognizer = TextRecognizer(
                cache_dir=self.cache_dir,
                use_cache=use_cache,
                backend=backend,
                model=model,
                gemini_tier=gemini_tier,
            )

        logger.info(
            "Pipeline initialized: detector=%s, sorter=%s, recognizer=%s (model=%s)",
            detector,
            sorter,
            self.backend.upper(),
            self.model,
        )

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
    ) -> Document | dict[str, Any]:
        """Process single image or PDF.

        Args:
            image_path: Path to image or PDF file
            max_pages: Maximum number of pages to process (PDF only)
            page_range: Range of pages to process (PDF only)
            pages: Specific pages to process (PDF only)

        Returns:
            Document for PDF, dict for single image
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
            Processing results with blocks and extracted text
        """
        logger.info("Processing image: %s", image_path)

        # Stage 1: Load image
        from .conversion.input import image as image_loader

        image_np = image_loader.load_image(image_path)

        # Stage 2: Detect layout blocks (new detector interface)
        blocks: list[Block] = self.detector.detect(image_np) if self.detector else []

        # Stage 3: Sort blocks by reading order (new sorter interface)
        if self.sorter:
            sorted_blocks: list[Block] = self.sorter.sort(blocks, image_np)
        else:
            sorted_blocks = blocks

        # Stage 4: Recognition - extract text from blocks
        if self.sorter_name == "olmocr-vlm":
            processed_blocks: list[Block] = sorted_blocks
        else:
            processed_blocks = self.recognizer.process_blocks(image_np, sorted_blocks)

        # Block-level text correction for text blocks
        for block in processed_blocks:
            if block.type in ["plain text", "title", "list", "text"] and block.text:
                corrected = self.recognizer.correct_text(block.text)
                if isinstance(corrected, dict):
                    block.corrected_text = corrected.get("corrected_text", block.text)
                    block.correction_ratio = corrected.get("correction_ratio", 0.0)
                elif isinstance(corrected, str):
                    block.corrected_text = corrected
                    block.correction_ratio = 0.0  # No ratio info available

        result = {
            "image_path": str(image_path),
            "blocks": [b.to_dict() for b in processed_blocks],
            "processed_at": tz_now().isoformat(),
        }

        return result

    def process_pdf(
        self,
        pdf_path: Path,
        max_pages: int | None = None,
        page_range: tuple[int, int] | None = None,
        pages: list[int] | None = None,
    ) -> Document:
        """Process PDF file with page limiting options.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to process
            page_range: Range of pages to process (start, end)
            pages: Specific list of page numbers to process

        Returns:
            Document object with full processing results
        """
        logger.info("Processing PDF: %s", pdf_path)

        # Import PDF converter functions
        from .conversion.input import pdf as pdf_converter

        # Get PDF info
        pdf_info = pdf_converter.get_pdf_info(pdf_path)
        total_pages = pdf_info["Pages"]

        # Determine which pages to process
        pages_to_process = pdf_converter.determine_pages_to_process(total_pages, max_pages, page_range, pages)

        logger.info("Processing %d pages: %s", len(pages_to_process), pages_to_process)

        output_dir = self._get_pdf_output_dir(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_pages, processing_stopped = self._process_pdf_pages(
            pdf_path, pages_to_process, total_pages, output_dir
        )

        document = self._create_pdf_summary(pdf_path, total_pages, processed_pages, processing_stopped, output_dir)

        logger.info("PDF processing complete: %s -> %s", pdf_path, output_dir)

        return document

    def _process_pdf_pages(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        total_pages: int,
        page_output_dir: Path,
    ) -> tuple[list[Page], bool]:
        """Process multiple PDF pages.

        Args:
            pdf_path: Path to PDF file
            pages_to_process: List of page numbers to process
            total_pages: Total number of pages in PDF
            page_output_dir: Directory to save page outputs

        Returns:
            Tuple of (list of Page objects, processing_stopped flag)
        """
        processed_pages: list[Page] = []
        processing_stopped = False

        from .conversion.input import pdf as pdf_converter

        pymupdf_doc = pdf_converter.open_pymupdf_document(pdf_path) if self.enable_multi_column_ordering else None
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

    def _process_pdf_page(  # noqa: PLR0912
        self,
        pdf_path: Path,
        page_num: int,
        page_output_dir: Path,
        pymupdf_page: Any | None,
    ) -> tuple[Page | None, bool]:
        """Process a single PDF page through the entire pipeline.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process
            page_output_dir: Directory to save page output
            pymupdf_page: PyMuPDF page object (optional)

        Returns:
            Tuple of (Page object or None, should_stop)
        """
        page_image = None

        try:
            # Stage 1: Convert PDF page to image
            from .conversion.input import pdf as pdf_converter

            page_image, temp_image_path = pdf_converter.render_pdf_page(pdf_path, page_num, temp_dir=self.temp_dir)

            # Stage 2: Detect layout blocks (new detector interface)
            blocks: list[Block] = self.detector.detect(page_image) if self.detector else []

            # Stage 3: Sort blocks by reading order (new sorter interface)
            if self.sorter:
                sorted_blocks: list[Block] = self.sorter.sort(blocks, page_image, pymupdf_page=pymupdf_page)
            else:
                # No sorting - keep original order
                sorted_blocks = blocks

            # Extract column layout info if available (for output compatibility)
            column_layout = self._extract_column_layout(sorted_blocks)

            # Stage 4: Recognition - extract text from blocks
            # Note: olmocr-vlm sorter already includes text, skip recognition
            if self.sorter_name == "olmocr-vlm":
                processed_blocks = sorted_blocks
            else:
                processed_blocks = self.recognizer.process_blocks(page_image, sorted_blocks)

            # Check for rate limit errors
            if self._check_for_rate_limit_errors({"blocks": processed_blocks}):
                logger.warning("Rate limit detected on page %d. Stopping processing.", page_num)
                return None, True

            # Block-level text correction for text blocks
            # Skip correction for paddleocr-vl (already does direct VLM extraction, no correction needed)
            if self.backend != "paddleocr-vl":
                for block in processed_blocks:
                    if block.type in ["plain text", "title", "list", "text"] and block.text:
                        corrected = self.recognizer.correct_text(block.text)
                        if isinstance(corrected, dict):
                            block.corrected_text = corrected.get("corrected_text", block.text)
                            block.correction_ratio = corrected.get("correction_ratio", 0.0)
                        elif isinstance(corrected, str):
                            block.corrected_text = corrected
                            block.correction_ratio = 0.0  # No ratio info available

            # Build a temporary Page object for rendering
            temp_page = Page(
                page_num=page_num,
                blocks=list(processed_blocks),
            )

            # Compose page-level text using selected renderer
            if self.renderer == "markdown":
                text = page_to_markdown(temp_page, include_page_header=False)
            else:  # plaintext
                text = blocks_to_plaintext(processed_blocks)

            # Correct the rendered text with VLM
            corrected_text, correction_ratio, stop_due_to_correction = self._perform_page_correction(text, page_num)

            if stop_due_to_correction:
                return None, True

            # Extract auxiliary info (text spans with font info for markdown conversion)
            auxiliary_info = self._extract_auxiliary_info(pdf_path, page_num)

            # Build result
            page_result = self._build_page_result(
                pdf_path,
                page_num,
                page_image,
                sorted_blocks,
                processed_blocks,
                text,
                corrected_text,
                correction_ratio,
                column_layout,
                auxiliary_info,
            )

            self._save_page_output(page_output_dir, page_num, page_result)

            return page_result, False

        except Exception as e:
            logger.error("Error processing page %d: %s", page_num, e, exc_info=True)
            error_page = Page(
                page_num=page_num,
                blocks=[],
                auxiliary_info={"error": str(e)},
                status="failed",
                processed_at=tz_now().isoformat(),
            )
            return error_page, False

        finally:
            if page_image is not None:
                del page_image
            gc.collect()

    def _perform_page_correction(self, raw_text: str, page_num: int) -> tuple[str, float, bool]:
        """Perform page-level text correction.

        Returns:
            tuple: (corrected_text, correction_ratio, should_stop)
                correction_ratio: 0.0 = no change, 1.0 = completely different
        """
        # Skip page correction for PaddleOCR-VL (it already extracts text directly)
        if self.backend == "paddleocr-vl":
            return raw_text, 0.0, False

        correction_result = self.recognizer.correct_text(raw_text)

        if isinstance(correction_result, dict):
            corrected_text = correction_result.get("corrected_text", raw_text)
            correction_ratio = float(correction_result.get("correction_ratio", 0.0))
            return corrected_text, correction_ratio, False

        corrected_text = str(correction_result)
        rate_limit_indicators = ["RATE_LIMIT_EXCEEDED", "DAILY_LIMIT_EXCEEDED"]
        if any(indicator in corrected_text for indicator in rate_limit_indicators):
            logger.warning(
                "Rate limit detected during page text correction on page %d. Stopping processing.",
                page_num,
            )
            return corrected_text, 0.0, True

        return corrected_text, 0.0, False

    def _extract_auxiliary_info(self, pdf_path: Path, page_num: int) -> dict[str, Any] | None:
        """Extract auxiliary information from PDF page.

        This includes text spans with font information for pymupdf4llm-style
        markdown conversion. Uses PyMuPDF terminology.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)

        Returns:
            Dictionary with auxiliary info or None if extraction fails
        """
        try:
            from .conversion.input.pdf import extract_text_spans_from_pdf

            text_spans = extract_text_spans_from_pdf(pdf_path, page_num)
            if text_spans:
                return {
                    "text_spans": text_spans,  # PyMuPDF spans with size, font
                }
            return None
        except Exception as exc:
            logger.warning("Failed to extract auxiliary info from page %d: %s", page_num, exc)
            return None

    def _build_page_result(
        self,
        pdf_path: Path,
        page_num: int,
        page_image: Any,
        detected_blocks: Sequence[Block],
        processed_blocks: Sequence[Block],
        text: str,
        corrected_text: str,
        correction_ratio: float,
        column_layout: dict[str, Any] | None,
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

    def _save_page_output(self, page_output_dir: Path, page_num: int, page: Page) -> None:
        """Save page processing output as JSON.

        Args:
            page_output_dir: Directory to save page output
            page_num: Page number
            page: Page object to save
        """
        page_output_file = page_output_dir / f"page_{page_num}.json"
        with page_output_file.open("w", encoding="utf-8") as f:
            json.dump(page.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", page_output_file)

    def _create_pdf_summary(
        self,
        pdf_path: Path,
        total_pages: int,
        processed_pages: list[Page],
        processing_stopped: bool,
        summary_output_dir: Path,
    ) -> Document:
        """Create PDF processing summary as Document object.

        Args:
            pdf_path: Path to PDF file
            total_pages: Total number of pages in PDF
            processed_pages: List of Page objects (processed or failed)
            processing_stopped: Whether processing was stopped early
            summary_output_dir: Directory to save summary

        Returns:
            Document object with full page data
        """
        pages_summary, status_counts = self._build_pages_summary(processed_pages)

        # Check if any pages have errors
        has_errors = any(page.status == "failed" for page in processed_pages)

        # Create Document object with full page data
        document = Document(
            pdf_name=pdf_path.stem,
            pdf_path=str(pdf_path),
            num_pages=total_pages,
            processed_pages=len(processed_pages),
            pages=processed_pages,  # Full Page objects
            detected_by=self.detector_name,
            ordered_by=self.sorter_name,
            recognized_by=f"{self.backend}/{self.model}",
            rendered_by=self.renderer,
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
            # Check in blocks
            blocks = page_result.get("blocks", [])
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict) and block.get("error") in ["gemini_rate_limit", "rate_limit_daily"]:
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

                # Check for block-level errors
                blocks = page_result.get("blocks", [])
                for block in blocks:
                    if isinstance(block, dict) and block.get("error"):
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

                # Store Document object in results
                results[str(pdf_file)] = result
                processed_files += 1

                # Note: Document objects don't track processing_stopped
                # Continue processing next file regardless

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

    def _extract_column_layout(self, blocks: list[Block]) -> dict[str, Any] | None:
        """Extract column layout information from sorted blocks.

        Args:
            blocks: Sorted blocks (may have column_index)

        Returns:
            Column layout dict or None
        """
        # Check if any blocks have column_index
        has_columns = any(r.column_index is not None for r in blocks)

        if not has_columns:
            return None

        # Extract unique columns
        column_indices = {r.column_index for r in blocks if r.column_index is not None}

        if not column_indices:
            return None

        # Build column layout info (filter out None values)
        columns = []
        for col_idx in sorted(column_indices):
            col_blocks = [r for r in blocks if r.column_index == col_idx]
            if col_blocks:
                # Get bbox if available
                first_block = col_blocks[0]
                if first_block.bbox:
                    bbox = first_block.bbox
                    columns.append(
                        {
                            "index": col_idx,
                            "x0": int(bbox.x0),
                            "x1": int(bbox.x1),
                            "center": bbox.center[0],
                            "width": bbox.width,
                        }
                    )

        if not columns:
            return None

        return {"columns": columns}

    def _save_results(self, result: dict[str, Any], output_path: Path) -> None:
        """Save processing results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info("Results saved to: %s", output_path)
