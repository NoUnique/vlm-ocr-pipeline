"""Image processing utilities for VLM API clients.

This module provides common image processing functions used by API clients
to prepare images for VLM inference.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "prepare_image_for_api",
    "resize_image",
    "numpy_to_jpeg_bytes",
    "DEFAULT_MAX_DIMENSION",
    "DEFAULT_JPEG_QUALITY",
]

# Constants for image processing
DEFAULT_MAX_DIMENSION: int = 1024
DEFAULT_JPEG_QUALITY: int = 85


def resize_image(
    image: np.ndarray,
    max_dim: int = DEFAULT_MAX_DIMENSION,
) -> np.ndarray:
    """Resize image if it exceeds maximum dimension.

    Maintains aspect ratio while ensuring the largest dimension
    does not exceed max_dim.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        max_dim: Maximum dimension (width or height)

    Returns:
        Resized image (or original if already within bounds)

    Example:
        >>> img = np.zeros((2000, 1000, 3), dtype=np.uint8)
        >>> resized = resize_image(img, max_dim=1024)
        >>> resized.shape
        (1024, 512, 3)
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h))


def numpy_to_jpeg_bytes(
    image: np.ndarray,
    quality: int = DEFAULT_JPEG_QUALITY,
    optimize: bool = True,
) -> bytes:
    """Convert numpy array to JPEG bytes.

    Handles BGR to RGB conversion for OpenCV images.

    Args:
        image: Input image as numpy array (H, W, C) in BGR format
        quality: JPEG quality (1-100)
        optimize: Whether to optimize JPEG encoding

    Returns:
        JPEG-encoded image bytes

    Example:
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> jpeg_bytes = numpy_to_jpeg_bytes(img)
        >>> len(jpeg_bytes) > 0
        True
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    pil_image = Image.fromarray(image_rgb)

    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG", quality=quality, optimize=optimize)

    return img_byte_arr.getvalue()


def prepare_image_for_api(
    image: np.ndarray,
    max_dim: int = DEFAULT_MAX_DIMENSION,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Prepare image for VLM API: resize and convert to JPEG bytes.

    This is the main function to use for preparing images before sending
    to VLM APIs (Gemini, OpenAI, etc.).

    Args:
        image: Input image as numpy array (H, W, C) in BGR format
        max_dim: Maximum dimension for resizing
        quality: JPEG quality (1-100)

    Returns:
        JPEG-encoded image bytes, ready for API upload

    Example:
        >>> img = np.zeros((2000, 1500, 3), dtype=np.uint8)
        >>> jpeg_bytes = prepare_image_for_api(img, max_dim=1024)
        >>> len(jpeg_bytes) > 0
        True
    """
    resized = resize_image(image, max_dim=max_dim)
    return numpy_to_jpeg_bytes(resized, quality=quality)

