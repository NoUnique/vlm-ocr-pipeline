"""JSON output conversion utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_blocks_to_json(
    blocks: list[dict[str, Any]],
    output_path: Path,
    indent: int = 2,
) -> None:
    """Save blocks to JSON file.

    Args:
        blocks: List of block dictionaries
        output_path: Output JSON file path
        indent: JSON indentation level (default: 2)

    Raises:
        OSError: If file cannot be written

    Example:
        >>> blocks = [
        ...     {"type": "text", "bbox": [100, 50, 200, 150], "confidence": 0.95},
        ...     {"type": "title", "bbox": [100, 10, 300, 40], "confidence": 0.98},
        ... ]
        >>> save_blocks_to_json(blocks, Path("output.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=indent, ensure_ascii=False)

    logger.info("Saved %d blocks to JSON: %s", len(blocks), output_path)


def save_pipeline_result_to_json(
    result: dict[str, Any],
    output_path: Path,
    indent: int = 2,
) -> None:
    """Save pipeline result to JSON file.

    Args:
        result: Pipeline result dictionary (may contain blocks, metadata, etc.)
        output_path: Output JSON file path
        indent: JSON indentation level (default: 2)

    Raises:
        OSError: If file cannot be written

    Example:
        >>> result = {
        ...     "metadata": {"total_pages": 5, "source": "document.pdf"},
        ...     "pages": [
        ...         {"page_num": 1, "blocks": [...]},
        ...     ]
        ... }
        >>> save_pipeline_result_to_json(result, Path("result.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=indent, ensure_ascii=False)

    logger.info("Saved pipeline result to JSON: %s", output_path)


def load_blocks_from_json(json_path: Path) -> list[dict[str, Any]]:
    """Load blocks from JSON file.

    Args:
        json_path: Input JSON file path

    Returns:
        List of block dictionaries

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON

    Example:
        >>> blocks = load_blocks_from_json(Path("output.json"))
        >>> len(blocks)
        10
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Support both direct list and wrapped format
    if isinstance(data, list):
        blocks = data
    elif isinstance(data, dict) and "blocks" in data:
        blocks = data["blocks"]
    elif isinstance(data, dict) and "pages" in data:
        # Multi-page format: extract all blocks
        blocks = []
        for page in data["pages"]:
            if "blocks" in page:
                blocks.extend(page["blocks"])
    else:
        raise ValueError(f"Unsupported JSON format in {json_path}")

    logger.info("Loaded %d blocks from JSON: %s", len(blocks), json_path)
    return blocks


def load_pipeline_result_from_json(json_path: Path) -> dict[str, Any]:
    """Load pipeline result from JSON file.

    Args:
        json_path: Input JSON file path

    Returns:
        Pipeline result dictionary

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON

    Example:
        >>> result = load_pipeline_result_from_json(Path("result.json"))
        >>> result["metadata"]["total_pages"]
        5
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded pipeline result from JSON: %s", json_path)
    return data
