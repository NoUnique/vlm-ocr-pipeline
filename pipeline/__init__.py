"""Unified VLM OCR Pipeline for document processing and text extraction."""

from __future__ import annotations

import gc
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from pipeline.stages import (
        BlockCorrectionStage,
        DetectionStage,
        OrderingStage,
        PageCorrectionStage,
        RecognitionStage,
        RenderingStage,
    )

# Load environment variables if not already loaded
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .config import PipelineConfig
from .factory import ComponentFactory
from .misc import tz_now
from .types import Block, ColumnLayout, Detector, Document, Page, PyMuPDFPage, Recognizer, Sorter

logger = logging.getLogger(__name__)


class Pipeline:
    """Unified VLM OCR processing pipeline with integrated text correction.

    This pipeline orchestrates four main stages:
    1. Document Conversion: Convert PDFs/images to processable format
    2. Layout Detection: Identify blocks (text, tables, figures, etc.)
    3. Layout Analysis: Determine reading order of blocks
    4. Recognition: Extract and correct text from blocks

    Example:
        >>> # New recommended way (using PipelineConfig)
        >>> from pipeline import Pipeline
        >>> from pipeline.config import PipelineConfig
        >>>
        >>> config = PipelineConfig(
        ...     detector="paddleocr-doclayout-v2",
        ...     recognizer="gemini-2.5-flash",
        ... )
        >>> pipeline = Pipeline(config=config)
        >>> document = pipeline.process_pdf(Path("document.pdf"))

        >>> # Legacy way (still supported with deprecation warning)
        >>> pipeline = Pipeline(
        ...     detector="paddleocr-doclayout-v2",
        ...     recognizer="gemini-2.5-flash",
        ... )
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        # Legacy parameters (for backward compatibility)
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
            config: Pipeline configuration object (recommended).
                   If provided, other parameters are ignored.

            # Legacy parameters (deprecated, use PipelineConfig instead):
            confidence_threshold: Detection confidence threshold (None = load from config)
            use_cache: Whether to use caching
            cache_dir: Cache directory path
            output_dir: Output directory path
            temp_dir: Temporary files directory path
            detector: Detector model name or alias (default: "paddleocr-doclayout-v2")
            detector_backend: Inference backend for detector (None = auto-select)
            detector_model_path: Custom detector model path
            auto_batch_size: Auto-calibrate optimal batch size for detector
            batch_size: Manual batch size for detector
            target_memory_fraction: Target GPU memory fraction for auto-calibration (0.0-1.0)
            sorter: Sorter model name or alias (None = auto-select)
            sorter_backend: Inference backend for sorter (None = auto-select)
            sorter_model_path: Custom sorter model path
            recognizer: Recognizer model name (default: "paddleocr-vl")
            recognizer_backend: Inference backend for recognizer (None = auto-select)
            gemini_tier: Gemini API tier for rate limiting
            renderer: Output format renderer ("markdown" or "plaintext")
            use_async: Enable async API clients for concurrent block processing
            dpi: DPI for PDF-to-image conversion
            detection_dpi: DPI for detection stage
            recognition_dpi: DPI for recognition stage
            use_dual_resolution: Use different DPIs for detection and recognition stages
        """
        # Handle configuration
        if config is not None:
            # Use provided config
            self.config = config
        else:
            # Create config from kwargs (legacy mode)
            # Check if any non-default values were passed
            has_custom_args = self._has_custom_arguments(
                confidence_threshold=confidence_threshold,
                use_cache=use_cache,
                cache_dir=cache_dir,
                output_dir=output_dir,
                temp_dir=temp_dir,
                detector=detector,
                detector_backend=detector_backend,
                detector_model_path=detector_model_path,
                auto_batch_size=auto_batch_size,
                batch_size=batch_size,
                target_memory_fraction=target_memory_fraction,
                sorter=sorter,
                sorter_backend=sorter_backend,
                sorter_model_path=sorter_model_path,
                recognizer=recognizer,
                recognizer_backend=recognizer_backend,
                gemini_tier=gemini_tier,
                renderer=renderer,
                use_async=use_async,
                dpi=dpi,
                detection_dpi=detection_dpi,
                recognition_dpi=recognition_dpi,
                use_dual_resolution=use_dual_resolution,
            )

            if has_custom_args:
                warnings.warn(
                    "Passing kwargs directly to Pipeline() is deprecated. "
                    "Use PipelineConfig instead:\n"
                    "  config = PipelineConfig(...)\n"
                    "  pipeline = Pipeline(config=config)",
                    DeprecationWarning,
                    stacklevel=2,
                )

            self.config = PipelineConfig(
                confidence_threshold=confidence_threshold,
                use_cache=use_cache,
                cache_dir=Path(cache_dir),
                output_dir=Path(output_dir),
                temp_dir=Path(temp_dir),
                detector=detector,
                detector_backend=detector_backend,
                detector_model_path=detector_model_path,
                auto_batch_size=auto_batch_size,
                batch_size=batch_size,
                target_memory_fraction=target_memory_fraction,
                sorter=sorter,
                sorter_backend=sorter_backend,
                sorter_model_path=sorter_model_path,
                recognizer=recognizer,
                recognizer_backend=recognizer_backend,
                gemini_tier=gemini_tier,
                renderer=renderer,
                use_async=use_async,
                dpi=dpi,
                detection_dpi=detection_dpi,
                recognition_dpi=recognition_dpi,
                use_dual_resolution=use_dual_resolution,
            )

        # Validate configuration
        self.config.validate()

        # Create component factory
        self.factory = ComponentFactory(self.config)

        # Setup directories
        self.factory.setup_directories()

        # Initialize rate limiter if needed
        self.factory.initialize_rate_limiter()

        # Create components using factory
        self.detector: Detector | None = self.factory.create_detector()
        self.sorter: Sorter | None = self.factory.create_sorter()
        self.recognizer: Recognizer = self.factory.create_recognizer()

        # Initialize Ray pools (lazy - only created when backend requires)
        self.ray_detector_pool = self.factory.create_ray_detector_pool(
            self._get_detector_kwargs() if self.config.detector == "doclayout-yolo" else None
        )
        self.ray_recognizer_pool = self.factory.create_ray_recognizer_pool()

        # Log pipeline configuration
        logger.info(
            "Pipeline initialized: detector=%s (backend=%s), sorter=%s (backend=%s), recognizer=%s (backend=%s)",
            self.config.detector,
            self.config.resolved_detector_backend or "auto",
            self.config.resolved_sorter,
            self.config.resolved_sorter_backend or "auto",
            self.config.recognizer,
            self.config.resolved_recognizer_backend,
        )

        # Initialize lightweight stages (InputStage and OutputStage)
        from pipeline.stages import InputStage, OutputStage

        self.input_stage = InputStage(
            temp_dir=self.config.temp_dir,
            dpi=self.config.dpi or 200,
            detection_dpi=self.config.detection_dpi or 150,
            recognition_dpi=self.config.recognition_dpi or 300,
            use_dual_resolution=self.config.use_dual_resolution,
        )
        self.output_stage = OutputStage(temp_dir=self.config.temp_dir)

        # Heavy stages will be created on-demand (lazy loading)
        self.detection_stage: DetectionStage | None = None
        self.ordering_stage: OrderingStage | None = None
        self.recognition_stage: RecognitionStage | None = None
        self.block_correction_stage: BlockCorrectionStage | None = None
        self.rendering_stage: RenderingStage | None = None
        self.page_correction_stage: PageCorrectionStage | None = None

    def _has_custom_arguments(self, **kwargs: Any) -> bool:
        """Check if any non-default arguments were passed."""
        defaults = {
            "confidence_threshold": None,
            "use_cache": True,
            "cache_dir": ".cache",
            "output_dir": "output",
            "temp_dir": ".tmp",
            "detector": "paddleocr-doclayout-v2",
            "detector_backend": None,
            "detector_model_path": None,
            "auto_batch_size": False,
            "batch_size": None,
            "target_memory_fraction": 0.85,
            "sorter": None,
            "sorter_backend": None,
            "sorter_model_path": None,
            "recognizer": "paddleocr-vl",
            "recognizer_backend": None,
            "gemini_tier": "free",
            "renderer": "markdown",
            "use_async": False,
            "dpi": None,
            "detection_dpi": None,
            "recognition_dpi": None,
            "use_dual_resolution": False,
        }
        for key, value in kwargs.items():
            if key in defaults and value != defaults[key]:
                return True
        return False

    def _get_detector_kwargs(self) -> dict[str, Any]:
        """Get detector kwargs for Ray pool initialization."""
        return {
            "model_path": self.config.detector_model_path,
            "confidence_threshold": self.config.confidence_threshold,
            "auto_batch_size": self.config.auto_batch_size,
            "batch_size": self.config.batch_size,
            "target_memory_fraction": self.config.target_memory_fraction,
        }

    # ==================== Backward Compatibility Properties ====================

    @property
    def detector_name(self) -> str:
        """Get detector name (backward compatibility)."""
        return self.config.detector

    @property
    def sorter_name(self) -> str:
        """Get sorter name (backward compatibility)."""
        return self.config.resolved_sorter

    @property
    def detector_backend(self) -> str | None:
        """Get detector backend (backward compatibility)."""
        return self.config.resolved_detector_backend

    @property
    def sorter_backend(self) -> str | None:
        """Get sorter backend (backward compatibility)."""
        return self.config.resolved_sorter_backend

    @property
    def recognizer_backend(self) -> str | None:
        """Get recognizer backend (backward compatibility)."""
        return self.config.resolved_recognizer_backend

    @property
    def recognizer_model(self) -> str:
        """Get recognizer model name (backward compatibility)."""
        return self.config.recognizer

    @property
    def backend(self) -> str:
        """Get backend (backward compatibility)."""
        return self.config.resolved_recognizer_backend

    @property
    def model(self) -> str:
        """Get model name (backward compatibility)."""
        return self.config.recognizer

    @property
    def gemini_tier(self) -> str:
        """Get Gemini tier (backward compatibility)."""
        return self.config.gemini_tier

    @property
    def renderer(self) -> str:
        """Get renderer (backward compatibility)."""
        return self.config.renderer

    @property
    def dpi(self) -> int:
        """Get DPI (backward compatibility)."""
        return self.config.dpi or 200

    @property
    def detection_dpi(self) -> int:
        """Get detection DPI (backward compatibility)."""
        return self.config.detection_dpi or 150

    @property
    def recognition_dpi(self) -> int:
        """Get recognition DPI (backward compatibility)."""
        return self.config.recognition_dpi or 300

    @property
    def use_dual_resolution(self) -> bool:
        """Get dual resolution flag (backward compatibility)."""
        return self.config.use_dual_resolution

    @property
    def cache_dir(self) -> Path:
        """Get cache directory (backward compatibility)."""
        return self.config.cache_dir

    @property
    def output_dir(self) -> Path:
        """Get output directory (backward compatibility)."""
        return self.config.output_dir

    @property
    def temp_dir(self) -> Path:
        """Get temp directory (backward compatibility)."""
        return self.config.temp_dir

    # ==================== Stage Factory Methods ====================

    def _create_detection_stage(self) -> DetectionStage:
        """Create detection stage on-demand."""
        from pipeline.stages import DetectionStage

        logger.info("Loading detection stage (%s)...", self.detector_name)
        return DetectionStage(
            self.detector,  # type: ignore[arg-type]
            ray_detector_pool=self.ray_detector_pool,
        )

    def _create_ordering_stage(self) -> OrderingStage:
        """Create ordering stage on-demand."""
        from pipeline.stages import OrderingStage

        logger.info("Loading ordering stage (%s)...", self.sorter_name)
        return OrderingStage(self.sorter)  # type: ignore[arg-type]

    def _create_recognition_stage(self) -> RecognitionStage:
        """Create recognition stage on-demand."""
        from pipeline.stages import RecognitionStage

        logger.info("Loading recognition stage (%s/%s)...", self.backend, self.model)
        return RecognitionStage(
            self.recognizer,
            ray_recognizer_pool=self.ray_recognizer_pool,
        )

    def _create_block_correction_stage(self) -> BlockCorrectionStage:
        """Create block correction stage on-demand."""
        from pipeline.stages import BlockCorrectionStage

        return BlockCorrectionStage(enable=False)

    def _create_rendering_stage(self) -> RenderingStage:
        """Create rendering stage on-demand."""
        from pipeline.stages import RenderingStage

        return RenderingStage(renderer=self.renderer)

    def _create_page_correction_stage(self) -> PageCorrectionStage:
        """Create page correction stage on-demand."""
        from pipeline.stages import PageCorrectionStage

        logger.info("Loading page correction stage...")
        return PageCorrectionStage(recognizer=self.recognizer, backend=self.backend, enable=True)  # type: ignore[arg-type]

    def _cleanup_stage(self, stage_name: str) -> None:
        """Cleanup and unload a stage to free memory."""
        stage_attr = f"{stage_name}_stage"
        if hasattr(self, stage_attr):
            stage = getattr(self, stage_attr)
            if stage is not None:
                del stage
                setattr(self, stage_attr, None)
                gc.collect()
                logger.info("Unloaded %s stage", stage_name)

    # ==================== Processing Methods ====================

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

    def _load_page_images(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, dict[str, Any]]]:
        """Stage 1: Load all page images and auxiliary info.

        Returns:
            Tuple of (page_images, recognition_images, auxiliary_infos)
        """
        logger.info("[Stage 1/7] Loading %d page images...", len(pages_to_process))

        page_images: dict[int, np.ndarray] = {}
        recognition_images: dict[int, np.ndarray] = {}
        auxiliary_infos: dict[int, dict[str, Any]] = {}

        for page_num in pages_to_process:
            if self.use_dual_resolution:
                page_images[page_num] = self.input_stage.load_pdf_page(pdf_path, page_num, dpi=self.detection_dpi)
                recognition_images[page_num] = self.input_stage.load_pdf_page(
                    pdf_path, page_num, dpi=self.recognition_dpi
                )
            else:
                page_img = self.input_stage.load_pdf_page(pdf_path, page_num)
                page_images[page_num] = page_img
                recognition_images[page_num] = page_img

            aux_info = self.input_stage.extract_auxiliary_info(pdf_path, page_num)
            auxiliary_infos[page_num] = aux_info if aux_info is not None else {}

        logger.info("[Stage 1/7] Loaded %d pages", len(page_images))
        return page_images, recognition_images, auxiliary_infos

    def _run_detection_stage(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        page_images: dict[int, np.ndarray],
        page_output_dir: Path,
    ) -> dict[int, list[Block]]:
        """Stage 2: Detect layout blocks for all pages."""
        logger.info("[Stage 2/7] Detecting layout blocks for %d pages...", len(page_images))

        detection_stage = self._create_detection_stage()
        detected_blocks: dict[int, list[Block]] = {}

        for page_num in pages_to_process:
            detected_blocks[page_num] = detection_stage.detect(page_images[page_num])

        logger.info("[Stage 2/7] Detection complete")
        del detection_stage
        gc.collect()

        self._save_intermediate_results(
            pdf_path, pages_to_process, page_output_dir, detected_blocks=detected_blocks, stage="detection"
        )
        return detected_blocks

    def _run_ordering_stage(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        detected_blocks: dict[int, list[Block]],
        page_images: dict[int, np.ndarray],
        page_output_dir: Path,
    ) -> dict[int, list[Block]]:
        """Stage 3: Sort blocks for all pages."""
        from .conversion.input import pdf as pdf_converter

        logger.info("[Stage 3/7] Sorting blocks for %d pages...", len(detected_blocks))

        # Open PyMuPDF document if needed
        pymupdf_doc = None
        pymupdf_pages: dict[int, PyMuPDFPage | None] = {}
        if self.sorter_name == "pymupdf":
            pymupdf_doc = pdf_converter.open_pymupdf_document(pdf_path)
            for page_num in pages_to_process:
                try:
                    pymupdf_pages[page_num] = pymupdf_doc.load_page(page_num - 1)
                except Exception as e:
                    logger.warning("Failed to load PyMuPDF page %d: %s", page_num, e)
                    pymupdf_pages[page_num] = None

        ordering_stage = self._create_ordering_stage()
        sorted_blocks: dict[int, list[Block]] = {}

        for page_num in pages_to_process:
            sorted_blocks[page_num] = ordering_stage.sort(
                detected_blocks[page_num],
                page_images[page_num],
                pymupdf_page=pymupdf_pages.get(page_num),
            )

        if pymupdf_doc is not None:
            pymupdf_doc.close()

        logger.info("[Stage 3/7] Ordering complete")
        del ordering_stage
        gc.collect()

        self._save_intermediate_results(
            pdf_path, pages_to_process, page_output_dir, detected_blocks=sorted_blocks, stage="ordering"
        )

        # Scale blocks if using dual resolution
        if self.use_dual_resolution and self.detection_dpi != self.recognition_dpi:
            scale_factor = self.recognition_dpi / self.detection_dpi
            logger.info("[Stage 3.5/7] Scaling blocks by factor %.2f", scale_factor)
            for page_num in pages_to_process:
                sorted_blocks[page_num] = self._scale_blocks(sorted_blocks[page_num], scale_factor)

        return sorted_blocks

    def _run_recognition_stage(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        sorted_blocks: dict[int, list[Block]],
        recognition_images: dict[int, np.ndarray],
        page_images: dict[int, np.ndarray],
        page_output_dir: Path,
    ) -> dict[int, list[Block]]:
        """Stage 4: Recognize text for all pages."""
        logger.info("[Stage 4/7] Recognizing text for %d pages...", len(sorted_blocks))

        recognition_stage = self._create_recognition_stage()
        recognized_blocks: dict[int, list[Block]] = {}

        for page_num in pages_to_process:
            if self.sorter_name == "olmocr-vlm":
                recognized_blocks[page_num] = sorted_blocks[page_num]
            else:
                recognized_blocks[page_num] = recognition_stage.recognize_blocks(
                    sorted_blocks[page_num], recognition_images[page_num]
                )

        logger.info("[Stage 4/7] Recognition complete")
        del recognition_stage
        gc.collect()

        self._save_intermediate_results(
            pdf_path, pages_to_process, page_output_dir,
            detected_blocks=recognized_blocks, page_images=page_images, stage="recognition",
        )
        return recognized_blocks

    def _run_correction_and_rendering_stages(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        recognized_blocks: dict[int, list[Block]],
        auxiliary_infos: dict[int, dict[str, Any]],
        page_images: dict[int, np.ndarray],
        page_output_dir: Path,
    ) -> tuple[dict[int, list[Block]], dict[int, str]]:
        """Stages 5-6: Block correction and rendering."""
        # Stage 5: Block Correction
        logger.info("[Stage 5/7] Block correction (skipped)")
        block_correction_stage = self._create_block_correction_stage()
        corrected_blocks: dict[int, list[Block]] = {}
        for page_num in pages_to_process:
            corrected_blocks[page_num] = block_correction_stage.correct_blocks(recognized_blocks[page_num])
        del block_correction_stage
        gc.collect()

        # Stage 6: Rendering
        logger.info("[Stage 6/7] Rendering %d pages...", len(corrected_blocks))
        rendering_stage = self._create_rendering_stage()
        rendered_texts: dict[int, str] = {}
        for page_num in pages_to_process:
            rendered_texts[page_num] = rendering_stage.render(corrected_blocks[page_num], auxiliary_infos[page_num])

        logger.info("[Stage 6/7] Rendering complete")
        del rendering_stage
        gc.collect()

        self._save_intermediate_results(
            pdf_path, pages_to_process, page_output_dir,
            detected_blocks=corrected_blocks, page_images=page_images,
            rendered_texts=rendered_texts, stage="rendering",
        )
        return corrected_blocks, rendered_texts

    def _run_output_stage(
        self,
        pdf_path: Path,
        pages_to_process: list[int],
        sorted_blocks: dict[int, list[Block]],
        corrected_blocks: dict[int, list[Block]],
        rendered_texts: dict[int, str],
        page_images: dict[int, np.ndarray],
        auxiliary_infos: dict[int, dict[str, Any]],
        page_output_dir: Path,
    ) -> tuple[list[Page], bool]:
        """Stage 7: Page correction and output."""
        logger.info("[Stage 7/7] Correcting and saving %d pages...", len(rendered_texts))

        processed_pages: list[Page] = []
        processing_stopped = False

        page_correction_stage = self._create_page_correction_stage()

        for page_num in pages_to_process:
            corrected_text, correction_ratio, stop_due_to_correction = page_correction_stage.correct_page(
                rendered_texts[page_num], page_num
            )

            if stop_due_to_correction:
                processing_stopped = True
                break

            column_layout = self._extract_column_layout(sorted_blocks[page_num])
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

            self.output_stage.save_page_output(page_output_dir, page_num, page_result)
            processed_pages.append(page_result)

        logger.info("[Stage 7/7] Processing complete")
        del page_correction_stage
        gc.collect()

        return processed_pages, processing_stopped

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
        5. Correct and render all pages
        6. Save outputs

        Args:
            pdf_path: Path to PDF file
            pages_to_process: List of page numbers to process
            total_pages: Total number of pages in PDF
            page_output_dir: Directory to save page outputs

        Returns:
            Tuple of (list of Page objects, processing_stopped flag)
        """
        # Stage 1: Load images
        page_images, recognition_images, auxiliary_infos = self._load_page_images(pdf_path, pages_to_process)

        # Stage 2: Detection
        detected_blocks = self._run_detection_stage(pdf_path, pages_to_process, page_images, page_output_dir)

        # Stage 3: Ordering
        sorted_blocks = self._run_ordering_stage(
            pdf_path, pages_to_process, detected_blocks, page_images, page_output_dir
        )

        # Stage 4: Recognition
        recognized_blocks = self._run_recognition_stage(
            pdf_path, pages_to_process, sorted_blocks, recognition_images, page_images, page_output_dir
        )

        # Stages 5-6: Correction and Rendering
        corrected_blocks, rendered_texts = self._run_correction_and_rendering_stages(
            pdf_path, pages_to_process, recognized_blocks, auxiliary_infos, page_images, page_output_dir
        )

        # Stage 7: Output
        processed_pages, processing_stopped = self._run_output_stage(
            pdf_path, pages_to_process, sorted_blocks, corrected_blocks,
            rendered_texts, page_images, auxiliary_infos, page_output_dir
        )

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

    def _extract_column_layout(self, blocks: list[Block]) -> ColumnLayout | None:
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
