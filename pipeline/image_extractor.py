"""Image extraction utilities for figure/chart blocks.

This module handles extracting and saving image regions from document pages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .types import BBox, Block, BlockType

logger = logging.getLogger(__name__)

# Block types that should be extracted as images
IMAGE_BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.IMAGE_BODY,
    BlockType.FIGURE,
    "image",
    "image_body",
    "figure",
    "chart",
}


def is_image_block(block: Block) -> bool:
    """Check if a block is an image/figure/chart type.

    Args:
        block: Block to check

    Returns:
        True if block should be extracted as an image
    """
    block_type_lower = block.type.lower()
    return block_type_lower in IMAGE_BLOCK_TYPES or block_type_lower in {
        "image",
        "image_body",
        "figure",
        "chart",
    }


def extract_block_image(
    page_image: np.ndarray,
    block: Block,
    padding: int = 2,
) -> np.ndarray:
    """Extract image region for a block from page image.

    Args:
        page_image: Full page image as numpy array (H, W, C)
        block: Block with bounding box
        padding: Padding around the block (pixels)

    Returns:
        Cropped image region as numpy array
    """
    return block.bbox.crop(page_image, padding=padding)


def save_block_image(
    page_image: np.ndarray,
    block: Block,
    output_dir: Path,
    page_num: int,
    block_idx: int,
    padding: int = 2,
) -> str | None:
    """Extract and save image block to file.

    Args:
        page_image: Full page image as numpy array (H, W, C)
        block: Block to extract
        output_dir: Base output directory
        page_num: Page number
        block_idx: Block index on the page
        padding: Padding around the block (pixels)

    Returns:
        Relative path to saved image (relative to output_dir), or None if failed
    """
    # Lazy import to avoid startup cost
    from PIL import Image  # noqa: PLC0415

    try:
        # Create images subdirectory
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Extract image region
        cropped = extract_block_image(page_image, block, padding=padding)

        # Generate filename: page_<num>_block_<idx>_<type>.png
        block_type = block.type.replace(" ", "_").lower()
        filename = f"page_{page_num}_block_{block_idx}_{block_type}.png"
        output_path = images_dir / filename

        # Save using PIL
        img = Image.fromarray(cropped)
        img.save(output_path)

        # Return relative path from output_dir
        relative_path = f"images/{filename}"
        logger.debug("Saved image block to %s", output_path)

        return relative_path

    except Exception as e:
        logger.warning("Failed to save image block: %s", e)
        return None


def extract_images_from_blocks(
    page_image: np.ndarray,
    blocks: list[Block],
    output_dir: Path,
    page_num: int,
    padding: int = 2,
) -> list[Block]:
    """Extract and save all image blocks from a page.

    This function modifies blocks in-place, adding image_path field to image blocks.

    Args:
        page_image: Full page image as numpy array (H, W, C)
        blocks: List of blocks (will be modified in-place)
        output_dir: Base output directory
        page_num: Page number
        padding: Padding around blocks (pixels)

    Returns:
        Same list of blocks with image_path populated for image blocks
    """
    for idx, block in enumerate(blocks):
        if is_image_block(block):
            image_path = save_block_image(
                page_image=page_image,
                block=block,
                output_dir=output_dir,
                page_num=page_num,
                block_idx=idx,
                padding=padding,
            )
            if image_path:
                block.image_path = image_path

    return blocks
