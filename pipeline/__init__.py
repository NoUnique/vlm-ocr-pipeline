"""Unified VLM OCR Pipeline for document processing and text extraction."""

from __future__ import annotations

import gc
import logging
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
from .io import InputLoader, OutputSaver
from .misc import tz_now
from .stages import DetectionStage
from .types import Block, Detector, Document, Page, PyMuPDFPage, Recognizer, Sorter

logger = logging.getLogger(__name__)


class Pipeline:
    """Unified VLM OCR processing pipeline with integrated text correction.

    This pipeline orchestrates eight main stages:
    1. Input: Load documents and extract auxiliary information (text spans, font metadata)
    2. Detection: Detect layout blocks using selected detector (DocLayout-YOLO, PaddleOCR, MinerU)
    3. Ordering: Analyze reading order using selected sorter (PyMuPDF, LayoutReader, XY-Cut, VLM)
    4. Recognition: Extract text from blocks using VLM (OpenAI, Gemini) or local model (PaddleOCR-VL)
    5. Block Correction: Block-level text correction (optional, disabled by default)
    6. Rendering: Convert processed blocks to Markdown or plaintext
    7. Page Correction: Page-level VLM correction for improved quality (optional, disabled by default)
    8. Output: Save results as JSON/Markdown and generate document summaries

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
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize VLM OCR processing pipeline.

        Args:
            config: Pipeline configuration object. If None, uses default configuration.

        Example:
            >>> from pipeline import Pipeline
            >>> from pipeline.config import PipelineConfig
            >>>
            >>> # Recommended: Use PipelineConfig
            >>> config = PipelineConfig(
            ...     detector="paddleocr-doclayout-v2",
            ...     recognizer="gemini-2.5-flash",
            ... )
            >>> pipeline = Pipeline(config=config)
            >>>
            >>> # Or use default configuration
            >>> pipeline = Pipeline()
        """
        # Use provided config or create default
        self.config = config if config is not None else PipelineConfig()

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

        # Initialize output saver (delegates saving logic)
        self._output_saver: OutputSaver | None = None

        # Initialize PDF processor (delegates PDF loading logic)
        self._input_loader: InputLoader | None = None

    @property
    def output_saver(self) -> OutputSaver:
        """Get or create OutputSaver instance (lazy initialization)."""
        if self._output_saver is None:
            self._output_saver = OutputSaver(
                detector_name=self.config.detector,
                sorter_name=self.config.resolved_sorter,
                backend=self.config.resolved_recognizer_backend,
                model=self.config.recognizer,
                renderer=self.config.renderer,
            )
        return self._output_saver

    @property
    def input_loader(self) -> InputLoader:
        """Get or create InputLoader instance (lazy initialization)."""
        if self._input_loader is None:
            self._input_loader = InputLoader(
                input_stage=self.input_stage,
                use_dual_resolution=self.config.use_dual_resolution,
                detection_dpi=self.config.detection_dpi or 150,
                recognition_dpi=self.config.recognition_dpi or 300,
            )
        return self._input_loader

    def _get_detector_kwargs(self) -> dict[str, Any]:
        """Get detector kwargs for Ray pool initialization."""
        return {
            "model_path": self.config.detector_model_path,
            "confidence_threshold": self.config.confidence_threshold,
            "auto_batch_size": self.config.auto_batch_size,
            "batch_size": self.config.batch_size,
            "target_memory_fraction": self.config.target_memory_fraction,
        }

    # ==================== Stage Factory Methods ====================

    def _create_detection_stage(self) -> DetectionStage:
        """Create detection stage on-demand."""
        from pipeline.stages import DetectionStage

        logger.info("Loading detection stage (%s)...", self.config.detector)
        return DetectionStage(
            self.detector,  # type: ignore[arg-type]
            ray_detector_pool=self.ray_detector_pool,
        )

    def _create_ordering_stage(self) -> OrderingStage:
        """Create ordering stage on-demand."""
        from pipeline.stages import OrderingStage

        logger.info("Loading ordering stage (%s)...", self.config.resolved_sorter)
        return OrderingStage(self.sorter)  # type: ignore[arg-type]

    def _create_recognition_stage(self) -> RecognitionStage:
        """Create recognition stage on-demand."""
        from pipeline.stages import RecognitionStage

        logger.info(
            "Loading recognition stage (%s/%s)...",
            self.config.resolved_recognizer_backend,
            self.config.recognizer,
        )
        return RecognitionStage(
            self.recognizer,
            ray_recognizer_pool=self.ray_recognizer_pool,
        )

    def _create_block_correction_stage(self) -> BlockCorrectionStage:
        """Create block correction stage on-demand."""
        from pipeline.stages import BlockCorrectionStage

        return BlockCorrectionStage(enable=self.config.enable_block_correction)

    def _create_rendering_stage(self) -> RenderingStage:
        """Create rendering stage on-demand."""
        from pipeline.stages import RenderingStage

        return RenderingStage(
            renderer=self.config.renderer,
            image_render_mode=self.config.image_render_mode,
        )

    def _create_page_correction_stage(self) -> PageCorrectionStage:
        """Create page correction stage on-demand."""
        from pipeline.stages import PageCorrectionStage

        return PageCorrectionStage(
            recognizer=self.recognizer,  # type: ignore[arg-type]
            backend=self.config.resolved_recognizer_backend,
            enable=self.config.enable_page_correction,
        )

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
        return self.config.output_dir / pdf_path.stem

    def _scale_blocks(self, blocks: list[Block], scale_factor: float) -> list[Block]:
        """Scale block bounding boxes by a factor.

        Delegates to InputLoader for actual scaling logic.

        Args:
            blocks: List of blocks with bounding boxes
            scale_factor: Scale factor to apply to bbox coordinates

        Returns:
            List of blocks with scaled bounding boxes
        """
        return self.input_loader.scale_blocks_for_recognition(blocks, scale_factor)

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
        from .io.input import image as image_loader

        image_np = image_loader.load_image(image_path)

        # Stage 2: Detect layout blocks (new detector interface)
        blocks: list[Block] = self.detector.detect(image_np) if self.detector else []

        # Stage 3: Sort blocks by reading order (new sorter interface)
        if self.sorter:
            sorted_blocks: list[Block] = self.sorter.sort(blocks, image_np)
        else:
            sorted_blocks = blocks

        # Stage 4: Recognition - extract text from blocks
        if self.config.resolved_sorter == "olmocr-vlm":
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
        from .io.input import pdf as pdf_converter

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
            detected_blocks[page_num] = detection_stage.process(page_images[page_num])

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
        from .io.input import pdf as pdf_converter

        logger.info("[Stage 3/7] Sorting blocks for %d pages...", len(detected_blocks))

        # Open PyMuPDF document if needed
        pymupdf_doc = None
        pymupdf_pages: dict[int, PyMuPDFPage | None] = {}
        if self.config.resolved_sorter == "pymupdf":
            pymupdf_doc = pdf_converter.open_pymupdf_document(pdf_path)
            if pymupdf_doc is not None:
                for page_num in pages_to_process:
                    try:
                        pymupdf_pages[page_num] = pymupdf_doc.load_page(page_num - 1)
                    except Exception as e:
                        logger.warning("Failed to load PyMuPDF page %d: %s", page_num, e)
                        pymupdf_pages[page_num] = None

        ordering_stage = self._create_ordering_stage()
        sorted_blocks: dict[int, list[Block]] = {}

        for page_num in pages_to_process:
            sorted_blocks[page_num] = ordering_stage.process(
                detected_blocks[page_num],
                image=page_images[page_num],
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
        detection_dpi = self.config.detection_dpi or 150
        recognition_dpi = self.config.recognition_dpi or 300
        if self.config.use_dual_resolution and detection_dpi != recognition_dpi:
            scale_factor = recognition_dpi / detection_dpi
            logger.info("[Stage 3.5/7] Scaling blocks by factor %.2f", scale_factor)
            for page_num in pages_to_process:
                sorted_blocks[page_num] = self._scale_blocks(sorted_blocks[page_num], scale_factor)

        return sorted_blocks

    def _run_image_extraction(
        self,
        pages_to_process: list[int],
        blocks: dict[int, list[Block]],
        page_images: dict[int, np.ndarray],
        page_output_dir: Path,
    ) -> None:
        """Extract and save image blocks to separate files.

        Args:
            pages_to_process: List of page numbers
            blocks: Blocks for each page (will be modified in-place to add image_path)
            page_images: Page images for extraction
            page_output_dir: Output directory
        """
        from .image_extractor import extract_images_from_blocks  # noqa: PLC0415

        logger.info("[Stage 3.5/7] Extracting image blocks for %d pages...", len(pages_to_process))

        for page_num in pages_to_process:
            extract_images_from_blocks(
                page_image=page_images[page_num],
                blocks=blocks[page_num],
                output_dir=page_output_dir,
                page_num=page_num,
            )

        logger.info("[Stage 3.5/7] Image extraction complete")

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
            if self.config.resolved_sorter == "olmocr-vlm":
                recognized_blocks[page_num] = sorted_blocks[page_num]
            else:
                recognized_blocks[page_num] = recognition_stage.process(
                    sorted_blocks[page_num], image=recognition_images[page_num]
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
        corrected_blocks: dict[int, list[Block]] = {}
        if self.config.enable_block_correction:
            logger.info("[Stage 5/7] Block correction for %d pages...", len(recognized_blocks))
            block_correction_stage = self._create_block_correction_stage()
            for page_num in pages_to_process:
                corrected_blocks[page_num] = block_correction_stage.process(recognized_blocks[page_num])
            del block_correction_stage
            gc.collect()
        else:
            logger.info("[Stage 5/7] Block correction (skipped)")
            # Copy text to corrected_text directly
            for page_num in pages_to_process:
                for block in recognized_blocks[page_num]:
                    if block.text is not None:
                        block.corrected_text = block.text
                corrected_blocks[page_num] = recognized_blocks[page_num]

        # Stage 6: Rendering (can be skipped with skip_rendering option)
        rendered_texts: dict[int, str] = {}
        if self.config.skip_rendering:
            logger.info("[Stage 6/7] Rendering (skipped - JSON only mode)")
            for page_num in pages_to_process:
                rendered_texts[page_num] = ""
            self._save_intermediate_results(
                pdf_path, pages_to_process, page_output_dir,
                detected_blocks=corrected_blocks, page_images=page_images,
                rendered_texts=None, stage="recognition",
            )
            return corrected_blocks, rendered_texts

        logger.info("[Stage 6/7] Rendering %d pages...", len(corrected_blocks))
        rendering_stage = self._create_rendering_stage()
        for page_num in pages_to_process:
            rendered_texts[page_num] = rendering_stage.process(
                corrected_blocks[page_num], auxiliary_info=auxiliary_infos[page_num]
            )

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
        processed_pages: list[Page] = []
        processing_stopped = False

        if self.config.enable_page_correction:
            logger.info("[Stage 7/7] Page correction and saving %d pages...", len(rendered_texts))
            page_correction_stage = self._create_page_correction_stage()

            for page_num in pages_to_process:
                result = page_correction_stage.process(rendered_texts[page_num], page_num=page_num)
                corrected_text = result.corrected_text
                correction_ratio = result.correction_ratio

                if result.should_stop:
                    processing_stopped = True
                    break

                self._save_page_result(
                    pdf_path, page_num, page_images, sorted_blocks, corrected_blocks,
                    rendered_texts, corrected_text, correction_ratio, auxiliary_infos,
                    page_output_dir, processed_pages,
                )

            del page_correction_stage
            gc.collect()
        else:
            logger.info("[Stage 7/7] Saving %d pages (page correction skipped)...", len(rendered_texts))
            for page_num in pages_to_process:
                # No correction: use rendered text as-is
                self._save_page_result(
                    pdf_path, page_num, page_images, sorted_blocks, corrected_blocks,
                    rendered_texts, rendered_texts[page_num], 0.0, auxiliary_infos,
                    page_output_dir, processed_pages,
                )

        logger.info("[Stage 7/7] Processing complete")

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
        # Stage 1: Load images (delegates to InputLoader)
        page_images, recognition_images, auxiliary_infos, _, _ = self.input_loader.load_page_images(
            pdf_path, pages_to_process
        )

        # Stage 2: Detection
        detected_blocks = self._run_detection_stage(pdf_path, pages_to_process, page_images, page_output_dir)

        # Stage 3: Ordering
        sorted_blocks = self._run_ordering_stage(
            pdf_path, pages_to_process, detected_blocks, page_images, page_output_dir
        )

        # Stage 3.5: Image extraction (if enabled)
        if self.config.enable_image_extraction:
            self._run_image_extraction(
                pages_to_process, sorted_blocks, recognition_images, page_output_dir
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

        Delegates to OutputSaver for actual saving logic.

        Args:
            pdf_path: Path to PDF file
            pages_to_process: List of page numbers
            page_output_dir: Output directory
            detected_blocks: Blocks for each page (at current stage)
            page_images: Page images (optional)
            rendered_texts: Rendered markdown texts (optional)
            stage: Current stage name
        """
        self.output_saver.save_intermediate_results(
            pdf_path=pdf_path,
            pages_to_process=pages_to_process,
            page_output_dir=page_output_dir,
            detected_blocks=detected_blocks,
            page_images=page_images,
            rendered_texts=rendered_texts,
            stage=stage,
        )
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
            detector_name=self.config.detector,
            sorter_name=self.config.resolved_sorter,
            backend=self.config.resolved_recognizer_backend,
            model=self.config.recognizer,
            renderer=self.config.renderer,
        )

    def _build_pages_summary(self, processed_pages: list[Page]) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Build summary of processed pages.
        
        Delegates to OutputSaver.
        """
        return self.output_saver.build_pages_summary(processed_pages)

    def _determine_summary_filename(self, processing_stopped: bool, has_errors: bool) -> str:
        """Determine summary filename based on processing status.
        
        Delegates to OutputSaver.
        """
        return self.output_saver.determine_summary_filename(processing_stopped, has_errors)

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

    def _save_page_result(
        self,
        pdf_path: Path,
        page_num: int,
        page_images: dict[int, np.ndarray],
        sorted_blocks: dict[int, list[Block]],
        corrected_blocks: dict[int, list[Block]],
        rendered_texts: dict[int, str],
        corrected_text: str,
        correction_ratio: float,
        auxiliary_infos: dict[int, dict[str, Any]],
        page_output_dir: Path,
        processed_pages: list[Page],
    ) -> None:
        """Save a single page result."""
        column_layout = DetectionStage.extract_column_layout(sorted_blocks[page_num])
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

    def _save_results(self, result: dict[str, Any], output_path: Path) -> None:
        """Save processing results to JSON file.
        
        Delegates to OutputSaver.
        """
        self.output_saver.save_final_results(result, output_path)
