"""Stage result serialization and deserialization.

This module handles saving and loading intermediate results between pipeline stages.
Each stage's output is serialized to JSON-compatible format.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pipeline.types import Page

logger = logging.getLogger(__name__)


def serialize_stage_result(
    stage: str,
    data: Any,
    output_file: Path,
) -> None:
    """Serialize stage result to JSON file.

    Args:
        stage: Stage name ("rendering", "detection", "ordering", "recognition", "output")
        data: Stage result data
        output_file: Output JSON file path

    Raises:
        ValueError: If stage is unknown
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if stage == "rendering":
        # data = (image: np.ndarray, auxiliary_info: dict | None)
        _serialize_rendering_stage(data, output_file)
    elif stage in ["detection", "ordering", "recognition", "correction"]:
        # data = Page object
        _serialize_page(data, output_file)
    elif stage == "output":
        # data = str (markdown)
        _serialize_output(data, output_file)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    logger.debug("Serialized %s stage to %s", stage, output_file)


def deserialize_stage_result(
    stage: str,
    input_file: Path,
) -> Any:
    """Deserialize stage result from JSON file.

    Args:
        stage: Stage name
        input_file: Input JSON file path

    Returns:
        Deserialized stage result

    Raises:
        ValueError: If stage is unknown
        FileNotFoundError: If input file not found
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {input_file}")

    if stage == "rendering":
        return _deserialize_rendering_stage(input_file)
    elif stage in ["detection", "ordering", "recognition", "correction"]:
        return _deserialize_page(input_file)
    elif stage == "output":
        return _deserialize_output(input_file)
    else:
        raise ValueError(f"Unknown stage: {stage}")


# ==================== Rendering Stage ====================


def _serialize_rendering_stage(
    data: tuple[np.ndarray, dict[str, Any] | None],
    output_file: Path,
) -> None:
    """Serialize rendering stage result (image + auxiliary info).

    Saves image as separate PNG file and auxiliary info in JSON.

    Args:
        data: (image, auxiliary_info) tuple
        output_file: Output JSON file path
    """
    image, auxiliary_info = data

    # Save image as PNG
    image_file = output_file.with_suffix(".png")
    try:
        from PIL import Image

        Image.fromarray(image).save(image_file)
        logger.debug("Saved image to %s", image_file)
    except Exception as e:
        logger.warning("Failed to save image: %s. Using base64 encoding.", e)
        # Fallback: encode as base64 in JSON
        image_b64 = base64.b64encode(image.tobytes()).decode("utf-8")
        result = {
            "image_shape": image.shape,
            "image_dtype": str(image.dtype),
            "image_data": image_b64,
            "auxiliary_info": auxiliary_info,
        }
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        return

    # Save JSON with image reference
    result = {
        "image_file": str(image_file.name),  # Relative path
        "image_shape": list(image.shape),
        "auxiliary_info": auxiliary_info,
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def _deserialize_rendering_stage(
    input_file: Path,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Deserialize rendering stage result.

    Args:
        input_file: Input JSON file path

    Returns:
        (image, auxiliary_info) tuple
    """
    with open(input_file) as f:
        data = json.load(f)

    # Load image
    if "image_file" in data:
        # Image saved as separate file
        image_file = input_file.parent / data["image_file"]
        from PIL import Image

        image = np.array(Image.open(image_file))
        logger.debug("Loaded image from %s", image_file)
    else:
        # Image encoded in JSON (fallback)
        image_data = base64.b64decode(data["image_data"])
        image = np.frombuffer(image_data, dtype=data["image_dtype"]).reshape(data["image_shape"])
        logger.debug("Decoded image from base64")

    auxiliary_info = data.get("auxiliary_info")

    return (image, auxiliary_info)


# ==================== Page Stages (detection, ordering, recognition) ====================


def _serialize_page(page: Page, output_file: Path) -> None:
    """Serialize Page object to JSON.

    Args:
        page: Page object
        output_file: Output JSON file path
    """
    # Page already has to_dict() method
    data = page.to_dict()

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def _deserialize_page(input_file: Path) -> Page:
    """Deserialize Page object from JSON.

    Args:
        input_file: Input JSON file path

    Returns:
        Page object
    """
    from pipeline.types import Page

    with open(input_file) as f:
        data = json.load(f)

    # Page already has from_dict() method
    return Page.from_dict(data)


# ==================== Output Stage ====================


def _serialize_output(markdown: str, output_file: Path) -> None:
    """Serialize output stage result (markdown string).

    Args:
        markdown: Markdown text
        output_file: Output JSON file path
    """
    # Also save as .md file for convenience
    md_file = output_file.with_suffix(".md")
    with open(md_file, "w") as f:
        f.write(markdown)
    logger.debug("Saved markdown to %s", md_file)

    # Save JSON with metadata
    result = {
        "markdown_file": str(md_file.name),
        "markdown_length": len(markdown),
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def _deserialize_output(input_file: Path) -> str:
    """Deserialize output stage result.

    Args:
        input_file: Input JSON file path

    Returns:
        Markdown string
    """
    with open(input_file) as f:
        data = json.load(f)

    # Load markdown from separate file
    md_file = input_file.parent / data["markdown_file"]
    with open(md_file) as f:
        markdown = f.read()

    logger.debug("Loaded markdown from %s", md_file)
    return markdown


# ==================== Utility Functions ====================


def save_checkpoint(
    stage: str,
    data: Any,
    output_dir: Path,
    page_num: int | None = None,
) -> str:
    """Save checkpoint for a stage.

    Args:
        stage: Stage name
        data: Stage result data
        output_dir: Output directory
        page_num: Optional page number for multi-page documents

    Returns:
        Relative path to saved checkpoint file (filename only)

    Example:
        >>> save_checkpoint("detection", page, Path("results"), page_num=1)
        "stage2_detection_page1.json"
    """
    # Generate filename
    stage_num = _get_stage_number(stage)
    if page_num is not None:
        filename = f"stage{stage_num}_{stage}_page{page_num}.json"
    else:
        filename = f"stage{stage_num}_{stage}.json"

    output_file = output_dir / filename

    # Serialize
    serialize_stage_result(stage, data, output_file)

    # Return relative path (filename only) for checkpoint tracking
    return filename


def load_checkpoint(
    stage: str,
    checkpoint_dir: Path,
    page_num: int | None = None,
) -> Any:
    """Load checkpoint for a stage.

    Args:
        stage: Stage name
        checkpoint_dir: Checkpoint directory
        page_num: Optional page number for multi-page documents

    Returns:
        Deserialized stage result

    Raises:
        FileNotFoundError: If checkpoint not found
    """
    # Generate filename
    stage_num = _get_stage_number(stage)
    if page_num is not None:
        filename = f"stage{stage_num}_{stage}_page{page_num}.json"
    else:
        filename = f"stage{stage_num}_{stage}.json"

    input_file = checkpoint_dir / filename

    # Deserialize
    return deserialize_stage_result(stage, input_file)


def _get_stage_number(stage: str) -> int:
    """Get stage number for filename.

    Args:
        stage: Stage name

    Returns:
        Stage number (1-indexed)
    """
    stage_order = {
        "rendering": 1,
        "detection": 2,
        "ordering": 3,
        "recognition": 4,
        "correction": 5,
        "output": 6,
    }

    return stage_order.get(stage, 0)
