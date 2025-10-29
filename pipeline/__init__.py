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
from .misc import tz_now
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
        detector: str = "paddleocr-doclayout-v2",
        detector_backend: str | None = None,
        detector_model_path: str | Path | None = None,
        # Batch processing options
        auto_batch_size: bool = False,
        batch_size: int | None = None,
        target_memory_fraction: float = 0.85,
        # Reading Order Stage
        sorter: str | None = None,
        sorter_backend: str | None = None,
        sorter_model_path: str | Path | None = None,
        # Text Recognition Stage
        recognizer: str = "paddleocr-vl",
        recognizer_backend: str | None = None,
        # API-specific options
        gemini_tier: str = "free",
        # Output options
        renderer: str = "markdown",
        # Performance options
        use_async: bool = False,
        # DPI options
        dpi: int | None = None,
        detection_dpi: int | None = None,
        recognition_dpi: int | None = None,
        use_dual_resolution: bool = False,
    ):
        """Initialize VLM OCR processing pipeline.

        Args:
            confidence_threshold: Detection confidence threshold (None = load from config)
            use_cache: Whether to use caching
            cache_dir: Cache directory path
            output_dir: Output directory path
            temp_dir: Temporary files directory path
            detector: Detector model name or alias (default: "paddleocr-doclayout-v2")
            detector_backend: Inference backend for detector (None = auto-select)
            detector_model_path: Custom detector model path (overrides model name resolution)
            auto_batch_size: Auto-calibrate optimal batch size for detector (recommended for multi-image)
            batch_size: Manual batch size for detector (ignored if auto_batch_size=True)
            target_memory_fraction: Target GPU memory fraction for auto-calibration (0.0-1.0)
            sorter: Sorter model name or alias (None = auto-select)
            sorter_backend: Inference backend for sorter (None = auto-select)
            sorter_model_path: Custom sorter model path (overrides model name resolution)
            recognizer: Recognizer model name (default: "paddleocr-vl")
            recognizer_backend: Inference backend for recognizer (None = auto-select)
            gemini_tier: Gemini API tier for rate limiting (only for gemini-* recognizers)
            renderer: Output format renderer ("markdown" or "plaintext")
            use_async: Enable async API clients for concurrent block processing (improves performance)
            dpi: DPI for PDF-to-image conversion (overrides default_dpi from config)
            detection_dpi: DPI for detection stage (overrides detection_dpi from config)
            recognition_dpi: DPI for recognition stage (overrides recognition_dpi from config)
            use_dual_resolution: Use different DPIs for detection and recognition stages
        """
        # Load configuration files
        models_config = _load_yaml_config(Path("settings") / "models.yaml")
        detection_config = _load_yaml_config(Path("settings") / "detection_config.yaml")
        pipeline_config = _load_yaml_config(Path("settings") / "config.yaml")

        # Load and merge DPI configuration
        conversion_config = pipeline_config.get("conversion", {})
        # CLI args override config values
        self.dpi = dpi if dpi is not None else conversion_config.get("default_dpi", 200)
        self.detection_dpi = detection_dpi if detection_dpi is not None else conversion_config.get("detection_dpi", 150)
        self.recognition_dpi = (
            recognition_dpi if recognition_dpi is not None else conversion_config.get("recognition_dpi", 300)
        )
        self.use_dual_resolution = use_dual_resolution or conversion_config.get("use_dual_resolution", False)

        logger.info(
            "DPI configuration: default=%d, detection=%d, recognition=%d, dual_resolution=%s",
            self.dpi,
            self.detection_dpi,
            self.recognition_dpi,
            self.use_dual_resolution,
        )

        # Lazy imports for factory functions (avoid loading heavy dependencies at module import)
        from .backend_validator import (  # noqa: PLC0415
            resolve_detector_backend,
            resolve_recognizer_backend,
            resolve_sorter_backend,
        )
        from .layout.detection import create_detector as create_detector_impl  # noqa: PLC0415
        from .layout.ordering import (  # noqa: PLC0415
            REQUIRED_COMBINATIONS,
            create_sorter as create_sorter_impl,
            validate_combination,
        )
        from .recognition import create_recognizer  # noqa: PLC0415
        from .recognition.api.ratelimit import rate_limiter  # noqa: PLC0415

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
                "auto_batch_size": auto_batch_size,
                "batch_size": batch_size,
                "target_memory_fraction": target_memory_fraction,
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

        self.recognizer: Recognizer = create_recognizer(
            self.recognizer_model, backend=recognizer_backend, **recognizer_kwargs
        )

        # Initialize Ray pools for multi-GPU parallelization (backend-based)
        self.ray_detector_pool = None
        self.ray_recognizer_pool = None

        # Initialize Ray detector pool if backend is pt-ray or hf-ray
        if detector_backend in ["pt-ray", "hf-ray"]:
            from pipeline.distributed import RayDetectorPool, is_ray_available  # noqa: PLC0415

            if not is_ray_available():
                logger.warning(
                    "Ray is not available. Falling back to single-GPU mode. Install Ray with: pip install ray"
                )
            else:
                try:
                    self.ray_detector_pool = RayDetectorPool(
                        detector_name=detector,
                        num_actors=None,  # Auto-detect from GPUs
                        num_gpus_per_actor=1.0,
                        **detector_kwargs,
                    )
                    logger.info(
                        "Ray detector pool initialized: %d actors",
                        self.ray_detector_pool.num_actors,
                    )
                except Exception as e:
                    logger.warning("Failed to initialize Ray detector pool: %s. Falling back to single-GPU.", e)

        # Initialize Ray recognizer pool if backend is pt-ray or hf-ray
        if recognizer_backend in ["pt-ray", "hf-ray"]:
            from pipeline.distributed import RayRecognizerPool, is_ray_available  # noqa: PLC0415

            if not is_ray_available():
                logger.warning(
                    "Ray is not available. Falling back to single-GPU mode. Install Ray with: pip install ray"
                )
            else:
                try:
                    self.ray_recognizer_pool = RayRecognizerPool(
                        recognizer_name=recognizer,
                        num_actors=None,  # Auto-detect from GPUs
                        num_gpus_per_actor=1.0,
                        backend=recognizer_backend,
                        **recognizer_kwargs,
                    )
                    logger.info(
                        "Ray recognizer pool initialized: %d actors",
                        self.ray_recognizer_pool.num_actors,
                    )
                except Exception as e:
                    logger.warning("Failed to initialize Ray recognizer pool: %s. Using single-GPU.", e)

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

        self.input_stage = InputStage(
            temp_dir=self.temp_dir,
            dpi=self.dpi,
            detection_dpi=self.detection_dpi,
            recognition_dpi=self.recognition_dpi,
            use_dual_resolution=self.use_dual_resolution,
        )
        self.detection_stage = DetectionStage(
            self.detector,  # type: ignore[arg-type]
            ray_detector_pool=self.ray_detector_pool,
        )
        self.ordering_stage = OrderingStage(self.sorter)  # type: ignore[arg-type]
        self.recognition_stage = RecognitionStage(
            self.recognizer,
            ray_recognizer_pool=self.ray_recognizer_pool,
        )
        self.block_correction_stage = BlockCorrectionStage(enable=False)  # Placeholder for future
        self.rendering_stage = RenderingStage(renderer=self.renderer)
        self.page_correction_stage = PageCorrectionStage(recognizer=self.recognizer, backend=self.backend, enable=True)  # type: ignore[arg-type]
        self.output_stage = OutputStage(temp_dir=self.temp_dir)

    def _setup_directories(self) -> None:
        """Create necessary directories."""
        for directory in [self.cache_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_pdf_output_dir(self, pdf_path: Path) -> Path:
        """Return the output directory for a given PDF as <output>/<file_stem>."""
        return self.output_dir / pdf_path.stem

    def _scale_blocks(self, blocks: list[Block], scale_factor: float) -> list[Block]:
        """Scale block bounding boxes by a factor.

        Used for dual-resolution mode where detection and recognition use different DPIs.

        Args:
            blocks: List of blocks with bounding boxes
            scale_factor: Scale factor to apply to bbox coordinates

        Returns:
            List of blocks with scaled bounding boxes
        """
        from .types import BBox

        scaled_blocks = []
        for block in blocks:
            if block.bbox:
                # Scale bbox coordinates
                scaled_bbox = BBox(
                    x0=int(block.bbox.x0 * scale_factor),
                    y0=int(block.bbox.y0 * scale_factor),
                    x1=int(block.bbox.x1 * scale_factor),
                    y1=int(block.bbox.y1 * scale_factor),
                )
                # Create new block with scaled bbox
                scaled_block = Block(
                    type=block.type,
                    bbox=scaled_bbox,
                    text=block.text,
                    detection_confidence=block.detection_confidence,
                    order=block.order,
                    column_index=block.column_index,
                    corrected_text=block.corrected_text,
                    correction_ratio=block.correction_ratio,
                    source=block.source,
                )
                scaled_blocks.append(scaled_block)
            else:
                # Block without bbox, keep as is
                scaled_blocks.append(block)
        return scaled_blocks

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
        """Process PDF pages using staged batch processing.

        This method processes all pages through each stage sequentially:
        1. Load all page images
        2. Detect all pages (batch) → unload detector
        3. Sort all pages → unload sorter
        4. Recognize all pages (batch) → unload recognizer
        5. Render and save all pages

        Args:
            pdf_path: Path to PDF file
            pages_to_process: List of page numbers to process
            total_pages: Total number of pages in PDF
            page_output_dir: Directory to save page outputs

        Returns:
            Tuple of (list of Page objects, processing_stopped flag)
        """
        import numpy as np

        processing_stopped = False
        processed_pages: list[Page] = []

        from .conversion.input import pdf as pdf_converter

        # Stage 1: Load all page images and auxiliary info
        logger.info("[Stage 1/7] Loading %d page images...", len(pages_to_process))
        page_images: dict[int, np.ndarray] = {}
        recognition_images: dict[int, np.ndarray] = {}
        auxiliary_infos: dict[int, dict[str, Any]] = {}
        pymupdf_pages: dict[int, PyMuPDFPage | None] = {}

        # Open PyMuPDF document if needed
        pymupdf_doc = None
        if self.sorter_name == "pymupdf":
            pymupdf_doc = pdf_converter.open_pymupdf_document(pdf_path)

        for page_num in pages_to_process:
            # Load images
            if self.use_dual_resolution:
                page_images[page_num] = self.input_stage.load_pdf_page(pdf_path, page_num, dpi=self.detection_dpi)
                recognition_images[page_num] = self.input_stage.load_pdf_page(
                    pdf_path, page_num, dpi=self.recognition_dpi
                )
            else:
                page_img = self.input_stage.load_pdf_page(pdf_path, page_num)
                page_images[page_num] = page_img
                recognition_images[page_num] = page_img

            # Extract auxiliary info
            aux_info = self.input_stage.extract_auxiliary_info(pdf_path, page_num)
            auxiliary_infos[page_num] = aux_info if aux_info is not None else {}

            # Load PyMuPDF page if needed
            if pymupdf_doc is not None:
                try:
                    pymupdf_pages[page_num] = pymupdf_doc.load_page(page_num - 1)
                except Exception as e:
                    logger.warning("Failed to load PyMuPDF page %d: %s", page_num, e)
                    pymupdf_pages[page_num] = None

        logger.info("[Stage 1/7] Loaded %d pages", len(page_images))

        # Stage 2: Detection - batch process all pages
        logger.info("[Stage 2/7] Detecting layout blocks for %d pages...", len(page_images))
        detected_blocks: dict[int, list[Block]] = {}
        for page_num in pages_to_process:
            detected_blocks[page_num] = self.detection_stage.detect(page_images[page_num])

        logger.info("[Stage 2/7] Detection complete")

        # Save intermediate results after detection
        self._save_intermediate_results(
            pdf_path, pages_to_process, page_output_dir, detected_blocks=detected_blocks, stage="detection"
        )

        # Stage 3: Ordering - sort all pages
        logger.info("[Stage 3/7] Sorting blocks for %d pages...", len(detected_blocks))
        sorted_blocks: dict[int, list[Block]] = {}
        for page_num in pages_to_process:
            sorted_blocks[page_num] = self.ordering_stage.sort(
                detected_blocks[page_num],
                page_images[page_num],
                pymupdf_page=pymupdf_pages.get(page_num),
            )

        # Close PyMuPDF document
        if pymupdf_doc is not None:
            pymupdf_doc.close()

        logger.info("[Stage 3/7] Ordering complete")

        # Save intermediate results after ordering
        self._save_intermediate_results(
            pdf_path, pages_to_process, page_output_dir, detected_blocks=sorted_blocks, stage="ordering"
        )

        # Scale blocks if using dual resolution
        if self.use_dual_resolution and self.detection_dpi != self.recognition_dpi:
            scale_factor = self.recognition_dpi / self.detection_dpi
            logger.info("[Stage 3.5/7] Scaling blocks by factor %.2f", scale_factor)
            for page_num in pages_to_process:
                sorted_blocks[page_num] = self._scale_blocks(sorted_blocks[page_num], scale_factor)

        # Stage 4: Recognition - batch process all pages
        logger.info("[Stage 4/7] Recognizing text for %d pages...", len(sorted_blocks))
        recognized_blocks: dict[int, list[Block]] = {}
        for page_num in pages_to_process:
            if self.sorter_name == "olmocr-vlm":
                # olmocr-vlm already includes text
                recognized_blocks[page_num] = sorted_blocks[page_num]
            else:
                recognized_blocks[page_num] = self.recognition_stage.recognize_blocks(
                    sorted_blocks[page_num], recognition_images[page_num]
                )

        logger.info("[Stage 4/7] Recognition complete")

        # Save intermediate results after recognition
        self._save_intermediate_results(
            pdf_path,
            pages_to_process,
            page_output_dir,
            detected_blocks=recognized_blocks,
            page_images=page_images,
            stage="recognition",
        )

        # Stage 5: Block Correction (currently disabled)
        logger.info("[Stage 5/7] Block correction (skipped)")
        corrected_blocks: dict[int, list[Block]] = {}
        for page_num in pages_to_process:
            corrected_blocks[page_num] = self.block_correction_stage.correct_blocks(recognized_blocks[page_num])

        # Stage 6: Rendering - render all pages
        logger.info("[Stage 6/7] Rendering %d pages...", len(corrected_blocks))
        rendered_texts: dict[int, str] = {}
        for page_num in pages_to_process:
            rendered_texts[page_num] = self.rendering_stage.render(
                corrected_blocks[page_num], auxiliary_infos[page_num]
            )

        logger.info("[Stage 6/7] Rendering complete")

        # Save intermediate results after rendering
        self._save_intermediate_results(
            pdf_path,
            pages_to_process,
            page_output_dir,
            detected_blocks=corrected_blocks,
            page_images=page_images,
            rendered_texts=rendered_texts,
            stage="rendering",
        )

        # Stage 7: Page Correction & Output
        logger.info("[Stage 7/7] Correcting and saving %d pages...", len(rendered_texts))
        for page_num in pages_to_process:
            # Page correction
            corrected_text, correction_ratio, stop_due_to_correction = self.page_correction_stage.correct_page(
                rendered_texts[page_num], page_num
            )

            if stop_due_to_correction:
                processing_stopped = True
                break

            # Build Page result
            column_layout = self.detection_stage.extract_column_layout(sorted_blocks[page_num])
            page_result = self.output_stage.build_page_result(
                pdf_path=pdf_path,
                page_num=page_num,
                page_image=page_images[page_num],
                detected_blocks=sorted_blocks[page_num],
                processed_blocks=corrected_blocks[page_num],
                text=rendered_texts[page_num],
                corrected_text=corrected_text,
                correction_ratio=correction_ratio,
                column_layout=column_layout,
                auxiliary_info=auxiliary_infos[page_num],
            )

            # Save output
            self.output_stage.save_page_output(page_output_dir, page_num, page_result)
            processed_pages.append(page_result)

        logger.info("[Stage 7/7] Processing complete")

        # Cleanup
        del page_images, recognition_images, detected_blocks, sorted_blocks, recognized_blocks, corrected_blocks
        gc.collect()

        return processed_pages, processing_stopped

    def _save_intermediate_results(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        page_output_dir: Path,
        detected_blocks: dict[int, list[Block]],
        page_images: dict[int, Any] | None = None,
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
        import numpy as np

        # Create json directory
        json_dir = page_output_dir / "json"
        json_dir.mkdir(parents=True, exist_ok=True)

        # Save each page's current state
        for page_num in pages_to_process:
            blocks = detected_blocks.get(page_num, [])

            # Build auxiliary_info based on available data
            auxiliary_info: dict[str, Any] = {}
            if page_images and page_num in page_images:
                img = page_images[page_num]
                auxiliary_info["width"] = int(img.shape[1])
                auxiliary_info["height"] = int(img.shape[0])

            if rendered_texts and page_num in rendered_texts:
                auxiliary_info["text"] = rendered_texts[page_num]

            # Create Page object with current state
            page = Page(
                page_num=page_num,
                blocks=blocks,
                auxiliary_info=auxiliary_info if auxiliary_info else None,
                status="in_progress",
                processed_at=tz_now().isoformat(),
            )

            # Save to JSON
            page_json_file = json_dir / f"page_{page_num}.json"
            with page_json_file.open("w", encoding="utf-8") as f:
                json.dump(page.to_dict(), f, ensure_ascii=False, indent=2)

        # Update summary.json with stage progress
        stage_progress = {
            "input": "complete",
            "detection": "complete" if stage in ["detection", "ordering", "recognition", "rendering"] else "pending",
            "ordering": "complete" if stage in ["ordering", "recognition", "rendering"] else "pending",
            "recognition": "complete" if stage in ["recognition", "rendering"] else "pending",
            "correction": "pending",
            "rendering": "complete" if stage == "rendering" else "pending",
        }

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

        logger.info("Saved intermediate results for stage: %s", stage)

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
        # Delegate to OutputStage for consistent summary creation
        return self.output_stage.create_pdf_summary(
            pdf_path=pdf_path,
            total_pages=total_pages,
            processed_pages=processed_pages,
            processing_stopped=processing_stopped,
            summary_output_dir=summary_output_dir,
            detector_name=self.detector_name,
            sorter_name=self.sorter_name,
            backend=self.backend,
            model=self.model,
            renderer=self.renderer,
        )

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
        """Process all PDFs in a directory using staged batch processing.

        When input is a directory, automatically uses staged batch processing
        to maximize GPU utilization by processing all files through each stage
        sequentially (convert all → detect all → sort all → recognize all).

        This is more efficient than sequential per-file processing because:
        - Models are loaded once per stage (not per file)
        - Better GPU utilization
        - Optimal for Ray multi-GPU parallelization

        Args:
            input_dir: Directory containing PDF files
            output_dir: Output directory
            max_pages: Maximum pages per PDF to process
            page_range: Page range per PDF to process
            specific_pages: Specific pages per PDF to process

        Returns:
            Processing summary dictionary
        """
        # Use staged batch processor for directory processing
        from pipeline.batch import StagedBatchProcessor  # noqa: PLC0415

        logger.info("Using staged batch processing for directory input")

        processor = StagedBatchProcessor(self)
        summary = processor.process_directory(
            input_dir,
            output_dir,
            max_pages=max_pages,
            page_range=page_range,
            specific_pages=specific_pages,
        )

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
