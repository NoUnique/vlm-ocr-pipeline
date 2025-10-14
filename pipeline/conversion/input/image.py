"""Image file loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_image(image_path: Path) -> np.ndarray:
    """Load an image file.

    Supports common image formats: JPEG, PNG, BMP, TIFF, WebP, etc.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array in BGR format (OpenCV convention)

    Raises:
        ValueError: If image cannot be loaded

    Example:
        >>> image = load_image(Path("photo.jpg"))
        >>> image.shape
        (1080, 1920, 3)
    """
    image_np = cv2.imread(str(image_path))
    if image_np is None:
        raise ValueError(f"Could not load image: {image_path}")

    logger.info("Loaded image: %s, shape: %s", image_path, image_np.shape)
    return image_np


def load_jpeg(image_path: Path) -> np.ndarray:
    """Load a JPEG image file.

    Args:
        image_path: Path to the JPEG file

    Returns:
        Image as numpy array in BGR format

    Raises:
        ValueError: If image cannot be loaded

    Example:
        >>> image = load_jpeg(Path("photo.jpg"))
        >>> image.dtype
        dtype('uint8')
    """
    if image_path.suffix.lower() not in [".jpg", ".jpeg"]:
        logger.warning(
            "File extension %s is not JPEG, but attempting to load anyway",
            image_path.suffix,
        )

    return load_image(image_path)


def load_png(image_path: Path) -> np.ndarray:
    """Load a PNG image file.

    Args:
        image_path: Path to the PNG file

    Returns:
        Image as numpy array in BGR format (alpha channel removed if present)

    Raises:
        ValueError: If image cannot be loaded

    Example:
        >>> image = load_png(Path("diagram.png"))
        >>> image.ndim
        3
    """
    if image_path.suffix.lower() != ".png":
        logger.warning(
            "File extension %s is not PNG, but attempting to load anyway",
            image_path.suffix,
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load PNG image: {image_path}")

    # Handle alpha channel if present
    if image.ndim == 3 and image.shape[2] == 4:  # noqa: PLR2004
        # Convert RGBA to RGB (remove alpha)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        logger.info("Converted PNG with alpha channel to BGR")

    logger.info("Loaded PNG image: %s, shape: %s", image_path, image.shape)
    return image

