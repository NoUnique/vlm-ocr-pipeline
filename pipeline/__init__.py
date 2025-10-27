"""Unified VLM OCR Pipeline for document processing and text extraction."""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import numpy as np

# Load environment variables if not already loaded
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .constants import DEFAULT_CONFIDENCE_THRESHOLD
from .layout.detection import create_detector as create_detector_impl
from .layout.ordering import create_sorter as create_sorter_impl, validate_combination
from .misc import tz_now
from .recognition import create_recognizer
from .recognition.api.ratelimit import rate_limiter
from .types import Block, Detector, Document, Page, PyMuPDFPage, Recognizer, Sorter

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
    except yaml.YAMLError as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return {}
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to read config file %s: %s", config_path, e)
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
        confidence_threshold: float | None = None,
        use_cache: bool = True,
        cache_dir: str | Path = ".cache",
        output_dir: str | Path = "output",
        temp_dir: str | Path = ".tmp",
        # Layout Detection Stage
        detector: str = "doclayout-yolo",
        detector_backend: str | None = None,
        detector_model_path: str | Path | None = None,
        # Reading Order Stage
        sorter: str | None = None,
        sorter_backend: str | None = None,
        sorter_model_path: str | Path | None = None,
        # Text Recognition Stage
        recognizer: str = "gemini-2.5-flash",
        recognizer_backend: str | None = None,
        # API-specific options
        gemini_tier: str = "free",
        # Output options
        renderer: str = "markdown",
        # Performance options
        use_async: bool = False,
    ):
        """Initialize VLM OCR processing pipeline.

        Args:
            confidence_threshold: Detection confidence threshold (None = load from config)
            use_cache: Whether to use caching
            cache_dir: Cache directory path
            output_dir: Output directory path
            temp_dir: Temporary files directory path
            detector: Detector model name or alias (e.g., "doclayout-yolo", "mineru-vlm")
            detector_backend: Inference backend for detector (None = auto-select)
            detector_model_path: Custom detector model path (overrides model name resolution)
            sorter: Sorter model name or alias (None = auto-select)
            sorter_backend: Inference backend for sorter (None = auto-select)
            sorter_model_path: Custom sorter model path (overrides model name resolution)
            recognizer: Recognizer model name (e.g., "gemini-2.5-flash", "gpt-4o", "paddleocr-vl")
            recognizer_backend: Inference backend for recognizer (None = auto-select)
            gemini_tier: Gemini API tier for rate limiting (only for gemini-* recognizers)
            renderer: Output format renderer ("markdown" or "plaintext")
            use_async: Enable async API clients for concurrent block processing (improves performance)
        """
        # Load configuration files
        models_config = _load_yaml_config(Path("settings") / "models.yaml")
        detection_config = _load_yaml_config(Path("settings") / "detection_config.yaml")

        # Import backend validator
        from .backend_validator import (  # noqa: PLC0415
            resolve_detector_backend,
            resolve_recognizer_backend,
            resolve_sorter_backend,
        )

        # Resolve and validate backends
        detector_backend, detector_error = resolve_detector_backend(detector, detector_backend)
        if detector_error:
            logger.warning("Detector backend validation: %s", detector_error)

        self.recognizer_model = recognizer  # Store recognizer model name
        recognizer_backend, recognizer_error = resolve_recognizer_backend(recognizer, recognizer_backend)
        if recognizer_error:
            raise ValueError(f"Recognizer backend validation failed: {recognizer_error}")

        # Ensure recognizer_backend is not None after validation
        if recognizer_backend is None:
            raise ValueError(f"No backend available for recognizer: {recognizer}")

        # Store resolved values
        self.detector_backend = detector_backend
        self.recognizer_backend = recognizer_backend
        self.gemini_tier = gemini_tier
        self.renderer = renderer.lower()

        # Backward compatibility: store backend and model attributes
        self.backend = recognizer_backend  # For backward compatibility
        self.model = recognizer  # For backward compatibility

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

        # Case 2: Sorter is not specified, auto-select based on detector
        if sorter is None:
            # Special case: tightly coupled detectors require their own sorter
            if detector == "paddleocr-doclayout-v2":
                sorter = "paddleocr-doclayout-v2"
                logger.info("Auto-selected sorter='paddleocr-doclayout-v2' for paddleocr-doclayout-v2 detector")
            elif detector == "mineru-vlm":
                sorter = "mineru-vlm"
                logger.info("Auto-selected sorter='mineru-vlm' for mineru-vlm detector")
            else:
                sorter = "mineru-xycut"
                logger.info("Using default sorter='mineru-xycut' (fast and accurate)")

        # Resolve and validate sorter backend
        sorter_backend, sorter_error = resolve_sorter_backend(sorter, sorter_backend)
        if sorter_error:
            logger.warning("Sorter backend validation: %s", sorter_error)

        # Store detector/sorter choices
        self.detector_name = detector
        self.sorter_name = sorter
        self.sorter_backend = sorter_backend

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

        # Initialize rate limiter (only for Gemini recognizer)
        if self.recognizer_backend == "gemini":
            rate_limiter.set_tier_and_model(gemini_tier, recognizer)

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

        # Helper function to map backend using backend_mapping from config
        def map_backend(stage: str, model_name: str, backend: str | None) -> dict[str, Any]:
            """Map user-friendly backend to actual implementation parameter."""
            if backend is None:
                return {}

            stage_config = models_config.get(f"{stage}s", {}).get(model_name, {})
            backend_mapping = stage_config.get("backend_mapping", {})
            backend_param_name = stage_config.get("backend_param_name", "backend")

            if backend_mapping and backend in backend_mapping:
                mapped_value = backend_mapping[backend]
                return {backend_param_name: mapped_value}
            elif backend_param_name:
                return {backend_param_name: backend}
            return {}

        # Create detector
        detector_kwargs = {}
        if detector == "doclayout-yolo":
            detector_kwargs = {
                "model_path": detector_model_path,
                "confidence_threshold": confidence_threshold,
            }
        elif detector == "mineru-vlm":
            # Use detector as model name (e.g., "opendatalab/PDF-Extract-Kit-1.0")
            # Or get default model from models_config
            detector_config = models_config.get("detectors", {}).get("mineru-vlm", {})
            default_model = detector_config.get("default_model", "opendatalab/MinerU2.5-2509-1.2B")
            final_model = detector if detector.startswith("opendatalab/") else default_model

            # Map backend (hf -> transformers, vllm -> vllm-engine)
            backend_kwargs = map_backend("detector", "mineru-vlm", detector_backend)
            logger.debug(
                "MinerU VLM detector: detector=%s, final_model=%s, backend_kwargs=%s",
                detector,
                final_model,
                backend_kwargs,
            )

            detector_kwargs = {
                "model": final_model,
                **backend_kwargs,
                "detection_only": (sorter != "mineru-vlm"),  # Full pipeline if using mineru-vlm sorter
            }
        elif detector == "paddleocr-doclayout-v2":
            detector_kwargs = {}  # No backend parameter
        elif detector == "mineru-doclayout-yolo":
            detector_kwargs = {}  # No backend parameter

        # Create detector
        self.detector: Detector | None = (
            create_detector_impl(detector, **detector_kwargs) if detector != "none" else None
        )

        # Create sorter
        sorter_kwargs = {}
        if sorter == "olmocr-vlm":
            # Use sorter as model name if it looks like a HF path
            sorter_config = models_config.get("sorters", {}).get("olmocr-vlm", {})
            default_model = sorter_config.get("default_model", "allenai/olmOCR-7B-0825-FP8")
            final_model = sorter if sorter.startswith("allenai/") else default_model

            # Map backend (hf -> use_vllm=False, vllm -> use_vllm=True)
            backend_kwargs = map_backend("sorter", "olmocr-vlm", sorter_backend)
            logger.debug(
                "olmOCR VLM sorter: sorter=%s, final_model=%s, backend_kwargs=%s", sorter, final_model, backend_kwargs
            )

            sorter_kwargs = {
                "model": final_model,
                **backend_kwargs,
                "use_anchoring": True,
            }
        elif sorter == "mineru-vlm":
            sorter_kwargs = {}  # Uses detector's backend, no separate parameters
        elif sorter == "mineru-layoutreader":
            sorter_kwargs = {}  # No backend parameter, uses device
        else:
            # For pymupdf, mineru-xycut, paddleocr-doclayout-v2, etc. (no backend needed)
            sorter_kwargs = {}

        self.sorter: Sorter | None = create_sorter_impl(sorter, **sorter_kwargs) if sorter else None

        # Recognizer (using factory pattern)
        recognizer_kwargs: dict[str, Any] = {}
        if recognizer_backend in ["pytorch", "vllm", "sglang"] or recognizer.startswith("PaddlePaddle/"):
            # PaddleOCR-VL recognizer
            # Map backend (pytorch -> native, vllm -> vllm-server, sglang -> sglang-server)
            backend_kwargs = map_backend("recognizer", "paddleocr-vl", recognizer_backend)
            logger.debug("PaddleOCR-VL recognizer: recognizer=%s, backend_kwargs=%s", recognizer, backend_kwargs)

            recognizer_kwargs.update(
                {
                    "device": None,  # Auto-detect
                    **backend_kwargs,
                    "use_layout_detection": False,  # We handle layout detection separately
                    "model": recognizer if recognizer.startswith("PaddlePaddle/") else None,
                }
            )
            recognizer_backend = "paddleocr-vl"  # Override for factory
        elif recognizer_backend in ["openai", "gemini"]:
            recognizer_kwargs.update(
                {
                    "cache_dir": self.cache_dir,
                    "use_cache": use_cache,
                    "model": recognizer,
                    "gemini_tier": gemini_tier,
                    "use_async": use_async,
                }
            )
        else:
            # Default to API-based recognizers
            recognizer_kwargs.update(
                {
                    "cache_dir": self.cache_dir,
                    "use_cache": use_cache,
                    "model": recognizer,
                    "gemini_tier": gemini_tier,
                    "use_async": use_async,
                }
            )

        self.recognizer: Recognizer = create_recognizer(recognizer_backend, **recognizer_kwargs)

        logger.info(
            "Pipeline initialized: detector=%s (backend=%s), sorter=%s (backend=%s), recognizer=%s (backend=%s)",
            detector,
            detector_backend or "auto",
            sorter,
            sorter_backend or "auto",
            recognizer,
            recognizer_backend,
        )

        # Initialize 8 pipeline stages
        from pipeline.stages import (
            BlockCorrectionStage,
            DetectionStage,
            InputStage,
            OrderingStage,
            OutputStage,
            PageCorrectionStage,
            RecognitionStage,
            RenderingStage,
        )

        self.input_stage = InputStage(temp_dir=self.temp_dir)
        self.detection_stage = DetectionStage(self.detector)  # type: ignore[arg-type]
        self.ordering_stage = OrderingStage(self.sorter)  # type: ignore[arg-type]
        self.recognition_stage = RecognitionStage(self.recognizer)
        self.block_correction_stage = BlockCorrectionStage(enable=False)  # Placeholder for future
        self.rendering_stage = RenderingStage(renderer=self.renderer)
        self.page_correction_stage = PageCorrectionStage(recognizer=self.recognizer, backend=self.backend, enable=True)  # type: ignore[arg-type]
        self.output_stage = OutputStage(temp_dir=self.temp_dir)

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

        # Open PyMuPDF document only if using pymupdf sorter for multi-column ordering
        pymupdf_doc = pdf_converter.open_pymupdf_document(pdf_path) if self.sorter_name == "pymupdf" else None
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
        pymupdf_page: PyMuPDFPage | None,
    ) -> tuple[Page | None, bool]:
        """Process a single PDF page through 8-stage pipeline.

        8-Stage Pipeline:
            1. Input: Load PDF page as image + extract auxiliary info
            2. Detection: Detect layout blocks
            3. BlockOrdering: Sort blocks by reading order
            4. Recognition: Extract text from blocks
            5. BlockCorrection: Block-level text correction (optional, disabled by default)
            6. Rendering: Convert blocks to Markdown/plaintext
            7. PageCorrection: Page-level text correction
            8. Output: Build Page object and save

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process
            page_output_dir: Directory to save page output
            pymupdf_page: PyMuPDF page object (optional, for pymupdf sorter)

        Returns:
            Tuple of (Page object or None, should_stop)
        """
        page_image = None

        try:
            # Stage 1: Input - Load PDF page
            page_image = self.input_stage.load_pdf_page(pdf_path, page_num)
            auxiliary_info = self.input_stage.extract_auxiliary_info(pdf_path, page_num)

            # Stage 2: Detection - Detect layout blocks
            blocks: list[Block] = self.detection_stage.detect(page_image)

            # Stage 3: BlockOrdering - Sort blocks by reading order
            sorted_blocks: list[Block] = self.ordering_stage.sort(blocks, page_image, pymupdf_page=pymupdf_page)

            # Extract column layout info (for backward compatibility)
            column_layout = self.detection_stage.extract_column_layout(sorted_blocks)

            # Stage 4: Recognition - Extract text from blocks
            # Note: olmocr-vlm sorter already includes text, skip recognition
            if self.sorter_name == "olmocr-vlm":
                processed_blocks = sorted_blocks
            else:
                processed_blocks = self.recognition_stage.recognize_blocks(sorted_blocks, page_image)

            # Check for rate limit errors after recognition
            if self._check_for_rate_limit_errors({"blocks": processed_blocks}):
                logger.warning("Rate limit detected on page %d. Stopping processing.", page_num)
                return None, True

            # Stage 5: BlockCorrection - Block-level correction (optional, currently disabled)
            processed_blocks = self.block_correction_stage.correct_blocks(processed_blocks)

            # Stage 6: Rendering - Convert blocks to Markdown/plaintext
            text = self.rendering_stage.render(processed_blocks, auxiliary_info)

            # Stage 7: PageCorrection - Page-level text correction
            corrected_text, correction_ratio, stop_due_to_correction = self.page_correction_stage.correct_page(
                text, page_num
            )

            if stop_due_to_correction:
                return None, True

            # Stage 8: Output - Build Page object and save
            page_result = self.output_stage.build_page_result(
                pdf_path=pdf_path,
                page_num=page_num,
                page_image=page_image,
                detected_blocks=sorted_blocks,
                processed_blocks=processed_blocks,
                text=text,
                corrected_text=corrected_text,
                correction_ratio=correction_ratio,
                column_layout=column_layout,
                auxiliary_info=auxiliary_info,
            )

            self.output_stage.save_page_output(page_output_dir, page_num, page_result)

            return page_result, False

        except Exception as e:
            # Fallback for unexpected errors (allowed per ERROR_HANDLING.md section 3.3)
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
            # Fallback for unexpected errors - auxiliary info is optional (allowed per ERROR_HANDLING.md section 3.3)
            logger.warning("Failed to extract auxiliary info from page %d: %s", page_num, exc)
            return None

    def _build_page_result(
        self,
        pdf_path: Path,
        page_num: int,
        page_image: np.ndarray,
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
                # Fallback for unexpected errors in batch processing - continue with next file
                # (allowed per ERROR_HANDLING.md section 3.3)
                logger.error("Error processing %s: %s", pdf_file, e, exc_info=True)
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
