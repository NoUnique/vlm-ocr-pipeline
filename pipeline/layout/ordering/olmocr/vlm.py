"""olmOCR VLM sorter implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pipeline.types import BBox, Region, regions_to_olmocr_anchor_text

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class OlmOCRVLMSorter:
    """Sorter using olmOCR VLM model.

    This sorter uses olmOCR's VLM to process the entire page and determine
    natural reading order. It can work with or without anchor text:

    1. With anchor text: Detection bbox → anchor text → VLM
    2. Without anchor text: Pure visual understanding

    Note: This sorter changes the output format from structured regions
    to text-only output, since olmOCR VLM returns natural text without
    individual region bboxes.
    """

    def __init__(
        self,
        model: str = "allenai/olmOCR-7B-0825-FP8",
        use_anchoring: bool = True,
        use_vllm: bool = False,
    ) -> None:
        """Initialize olmOCR VLM sorter.

        Args:
            model: olmOCR model path or name
            use_anchoring: Whether to use anchor text (detection bbox → anchor)
            use_vllm: Whether to use vLLM for faster inference

        Raises:
            ImportError: If required dependencies not available
        """
        self.model_path = model
        self.use_anchoring = use_anchoring
        self.use_vllm = use_vllm
        self._vlm_client = None

        logger.info(
            "olmOCR VLM sorter initialized (model=%s, anchoring=%s)",
            model,
            use_anchoring,
        )

    def _get_vlm_client(self) -> Any:
        """Lazy load VLM client."""
        if self._vlm_client is not None:
            return self._vlm_client

        try:
            # TODO: Implement olmOCR VLM client loading
            raise NotImplementedError(
                "olmOCR VLM integration not yet implemented. "
                "This requires setting up vLLM server or transformers client."
            )
        except ImportError as e:
            raise ImportError(
                "olmOCR VLM dependencies not available. "
                "Please install vllm or transformers."
            ) from e

    def sort(self, regions: list[Region], image: np.ndarray, **kwargs: Any) -> list[Region]:
        """Sort regions using olmOCR VLM.

        This sorter processes the entire page with VLM and returns a single
        region containing the natural text output.

        Args:
            regions: Detected regions (used for anchor text if use_anchoring=True)
            image: Page image
            **kwargs: Additional context:
                - pdf_path: PDF path for PyPDF fallback anchor
                - page_num: Page number

        Returns:
            Single region containing VLM output text

        Example:
            >>> sorter = OlmOCRVLMSorter()
            >>> result = sorter.sort(regions, image)
            >>> result[0]["text"]
            'Chapter 1\n\nThis is the first paragraph...'
        """
        page_height, page_width = image.shape[:2]

        if self.use_anchoring and regions:
            anchor_text = regions_to_olmocr_anchor_text(regions, page_width, page_height)
            prompt = self._build_prompt_with_anchor(anchor_text)
        else:
            prompt = self._build_prompt_no_anchor()

        try:
            vlm_client = self._get_vlm_client()
            response = vlm_client.process(image, prompt)

            result_region = Region(
                type="text",
                coords=[0, 0, float(page_width), float(page_height)],
                confidence=1.0,
                bbox=BBox(0, 0, page_width, page_height),
                source="olmocr-vlm",
                text=response.get("natural_text", ""),
                reading_order_rank=0,
            )

            logger.debug("Processed page with olmOCR VLM")

            return [result_region]

        except NotImplementedError:
            logger.error("olmOCR VLM not implemented yet, falling back to simple sort")
            return self._fallback_sort(regions)

    def _build_prompt_with_anchor(self, anchor_text: str) -> str:
        """Build prompt with anchor text."""
        return (
            f"Below is the image of one page of a document, as well as some raw textual content "
            f"that was previously extracted for it. "
            f"Just return the plain text representation of this document as if you were reading it naturally.\n"
            f"Do not hallucinate.\n"
            f"RAW_TEXT_START\n{anchor_text}\nRAW_TEXT_END"
        )

    def _build_prompt_no_anchor(self) -> str:
        """Build prompt without anchor text."""
        return (
            "Attached is one page of a document that you must process. "
            "Just return the plain text representation of this document as if you were reading it naturally. "
            "Convert equations to LaTeX and tables to markdown.\n"
            "Return your output as markdown, with a front matter section on top specifying values "
            "for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
        )

    def _fallback_sort(self, regions: list[Region]) -> list[Region]:
        """Fallback to simple geometric sorting."""
        from pipeline.types import ensure_bbox_in_region

        if not regions:
            return regions

        regions = [ensure_bbox_in_region(r) for r in regions]
        sorted_regions = sorted(regions, key=lambda r: (r.bbox.y0, r.bbox.x0) if r.bbox else (0, 0))

        for rank, region in enumerate(sorted_regions):
            region.reading_order_rank = rank

        return sorted_regions

