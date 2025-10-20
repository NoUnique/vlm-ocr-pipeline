"""Plaintext output conversion utilities.

This module provides plaintext conversion by joining blocks with double newlines.
This is a simple text concatenation strategy without any formatting.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.types import Block, Page


def blocks_to_plaintext(blocks: Sequence[Block]) -> str:
    """Convert blocks to plaintext by joining with double newlines.

    This function composes page-level raw text from processed blocks in reading order.
    Reading order: Uses order field if available, otherwise top-to-bottom (y),
    then left-to-right (x). Includes text-like blocks only and preserves internal
    newlines within each block's text.

    Args:
        blocks: List of processed blocks with text content

    Returns:
        Composed text in natural reading order, blocks separated by double newlines

    Example:
        >>> from pipeline.types import Block, BBox
        >>> blocks = [
        ...     Block(type="title", bbox=BBox(0, 0, 100, 20), text="Chapter 1", order=0),
        ...     Block(type="text", bbox=BBox(0, 30, 100, 50), text="First paragraph.", order=1),
        ... ]
        >>> text = blocks_to_plaintext(blocks)
        >>> print(text)
        Chapter 1
        <BLANKLINE>
        First paragraph.
    """
    if not blocks:
        return ""

    text_like_types = {"plain text", "text", "title", "list"}
    sortable_items: list[tuple[int, int, str, int | None]] = []

    for block in blocks:
        if block.type not in text_like_types:
            continue
        x, y = block.bbox.x0, block.bbox.y0
        text_value = block.text
        if text_value and text_value.strip():
            # Keep internal newlines; trim outer whitespace only
            order_rank = block.order
            sortable_items.append((y, x, text_value.strip(), order_rank))

    # Sort by reading order rank if available, otherwise by y then x
    if sortable_items:
        if any(item[3] is not None for item in sortable_items):
            sortable_items.sort(
                key=lambda item: (
                    0 if item[3] is not None else 1,
                    item[3] if item[3] is not None else item[0],
                    item[0],
                    item[1],
                )
            )
        else:
            sortable_items.sort(key=lambda item: (item[0], item[1]))

    # Join with a blank line between blocks to separate them
    return "\n\n".join(item[2] for item in sortable_items)


def page_to_plaintext(page: Page) -> str:
    """Convert a Page object to plaintext.

    Args:
        page: Page object with blocks

    Returns:
        Plaintext representation of the page

    Example:
        >>> from pipeline.types import Page, Block, BBox
        >>> page = Page(
        ...     page_num=1,
        ...     blocks=[Block(type="text", bbox=BBox(0, 0, 100, 20), text="Hello")]
        ... )
        >>> text = page_to_plaintext(page)
        >>> print(text)
        Hello
    """
    return blocks_to_plaintext(page.blocks)
