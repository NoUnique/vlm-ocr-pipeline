"""Component interface definitions for the VLM OCR pipeline.

This module defines Protocol interfaces for pipeline components:
- Detector: Layout detection interface
- Sorter: Reading order analysis interface
- Recognizer: Text recognition interface
- Renderer: Output rendering interface
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from .block import Block


@runtime_checkable
class Detector(Protocol):
    """Layout detection interface.

    All detectors must implement this interface and return blocks
    in the unified Block format with bbox field.

    Attributes:
        name: Detector identifier (e.g., "doclayout-yolo", "paddleocr-doclayout-v2")
        source: Source identifier for blocks (used in Block.source field)

    Methods:
        detect: Detect blocks in a single image
        detect_batch: Detect blocks in multiple images (optional, default: sequential)

    Example:
        >>> detector = DocLayoutYOLODetector()
        >>> detector.name
        'doclayout-yolo'
        >>> blocks = detector.detect(image)
    """

    name: str
    source: str

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect blocks in image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            List of detected Block objects with bbox

        Example:
            >>> detector = DocLayoutYOLODetector()
            >>> blocks = detector.detect(image)
            >>> blocks[0].type
            'text'
            >>> blocks[0].bbox
            BBox(x0=100, y0=50, x1=300, y1=200)
        """
        ...

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Block]]:
        """Detect blocks in multiple images.

        Default implementation processes images sequentially.
        Subclasses may override for true batch/parallel processing.

        Args:
            images: List of input images as numpy arrays (H, W, C)

        Returns:
            List of block lists, one per image

        Example:
            >>> detector = DocLayoutYOLODetector()
            >>> results = detector.detect_batch([image1, image2])
            >>> len(results)
            2
        """
        ...


@runtime_checkable
class Sorter(Protocol):
    """Reading order sorting interface.

    All sorters must implement this interface and add ordering information
    to blocks (order field) while maintaining the unified Block format.

    Attributes:
        name: Sorter identifier (e.g., "pymupdf", "mineru-xycut")

    Methods:
        sort: Sort blocks by reading order

    Example:
        >>> sorter = PyMuPDFSorter()
        >>> sorter.name
        'pymupdf'
        >>> sorted_blocks = sorter.sort(blocks, image)
    """

    name: str

    def sort(self, blocks: list[Block], image: np.ndarray, **kwargs: Any) -> list[Block]:
        """Sort blocks by reading order.

        Args:
            blocks: Detected blocks with bbox
            image: Page image for analysis (H, W, C)
            **kwargs: Additional context (e.g., pymupdf_page, pdf_path)

        Returns:
            Sorted blocks with order field added

        Example:
            >>> sorter = PyMuPDFSorter()
            >>> sorted_blocks = sorter.sort(blocks, image, pymupdf_page=page)
            >>> sorted_blocks[0].order
            0
        """
        ...


@runtime_checkable
class Recognizer(Protocol):
    """Text recognition interface.

    All recognizers must implement this interface to extract and correct
    text from blocks using various OCR or VLM backends.

    Attributes:
        name: Recognizer identifier (e.g., "gemini", "openai", "paddleocr-vl")
        supports_correction: Whether the recognizer supports text correction

    Methods:
        process_blocks: Extract text from blocks in an image
        correct_text: Correct raw text using VLM (optional for some backends)
        process_blocks_batch: Process multiple sets of blocks (optional)

    Example:
        >>> recognizer = GeminiClient(model="gemini-2.5-flash")
        >>> recognizer.name
        'gemini'
        >>> recognizer.supports_correction
        True
        >>> blocks_with_text = recognizer.process_blocks(image, blocks)
    """

    name: str
    supports_correction: bool

    def process_blocks(
        self,
        image: np.ndarray | None,
        blocks: Sequence[Block],
        *,
        enable_figure_description: bool = True,
    ) -> list[Block]:
        """Process blocks to extract text.

        This method processes each block in the input image to extract text content.
        The returned blocks should have their `text` field populated.

        For image/figure/chart blocks, when enable_figure_description is True,
        generates a description and stores it in the `description` field.

        Args:
            image: Full page image as numpy array (H, W, C) in RGB format.
                   Can be None for recognizers that don't need the image.
            blocks: Detected blocks with bbox field populated
            enable_figure_description: Whether to generate descriptions for image blocks.
                When True, image/figure/chart blocks will have their `description` field
                populated with VLM-generated content. Default: True.

        Returns:
            List of blocks with text field populated (and description for image blocks)

        Example:
            >>> recognizer = GeminiClient(model="gemini-2.5-flash")
            >>> blocks_with_text = recognizer.process_blocks(image, blocks)
            >>> blocks_with_text[0].text
            'Sample text content'
            >>> # For image blocks with description enabled
            >>> image_block = blocks_with_text[1]  # type='image'
            >>> image_block.description
            'A bar chart showing quarterly sales...'
        """
        ...

    def correct_text(self, text: str) -> str | dict[str, Any]:
        """Correct extracted text using VLM.

        This method takes raw extracted text and applies correction using a VLM.
        Some recognizers may not support correction (e.g., PaddleOCR-VL).
        Check `supports_correction` attribute before calling.

        Args:
            text: Raw extracted text to correct

        Returns:
            Corrected text string, or dict with correction metadata:
            - {"corrected_text": str, "correction_ratio": float}
            If correction is not supported, returns the original text unchanged.

        Example:
            >>> recognizer = GeminiClient(model="gemini-2.5-flash")
            >>> if recognizer.supports_correction:
            ...     corrected = recognizer.correct_text("sampel txt")
            ...     print(corrected)
            'sample text'

            >>> # For recognizers without correction support
            >>> recognizer = PaddleOCRVLRecognizer()
            >>> recognizer.supports_correction
            False
        """
        ...

    def process_blocks_batch(
        self,
        images: Sequence[np.ndarray | None],
        blocks_list: Sequence[Sequence[Block]],
        *,
        enable_figure_description: bool = True,
    ) -> list[list[Block]]:
        """Process multiple sets of blocks in a batch.

        Default implementation processes sequentially.
        Subclasses may override for true batch/parallel processing.

        Args:
            images: Sequence of input images
            blocks_list: Sequence of block lists, one per image
            enable_figure_description: Whether to generate descriptions for image blocks.
                Default: True.

        Returns:
            List of processed block lists

        Example:
            >>> results = recognizer.process_blocks_batch(
            ...     [image1, image2],
            ...     [blocks1, blocks2]
            ... )
        """
        ...


@runtime_checkable
class Renderer(Protocol):
    """Output rendering interface.

    Renderers convert processed blocks/pages/documents to various output
    formats (markdown, plaintext, HTML, etc.).

    This is a callable protocol - any function matching the signature can be used.
    """

    def __call__(self, blocks: Sequence[Block], **kwargs: Any) -> str:
        """Render blocks to output format.

        Args:
            blocks: Processed blocks with text
            **kwargs: Additional rendering options

        Returns:
            Rendered output string

        Example:
            >>> renderer = blocks_to_markdown
            >>> text = renderer(blocks)
            >>> print(text)
            # Title

            Sample content...
        """
        ...
