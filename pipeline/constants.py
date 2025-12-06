"""Shared constants for the VLM OCR pipeline."""

# =============================================================================
# Rate Limiting
# =============================================================================
REQUEST_WINDOW_SECONDS = 60
"""Sliding window size (seconds) for per-minute rate limiting."""

# =============================================================================
# LayoutReader Scaling
# =============================================================================
LAYOUTREADER_SCALE = 1000
"""Coordinate scaling factor for LayoutReader model (scales to 1000x1000)."""

# =============================================================================
# Memory Management
# =============================================================================
DEFAULT_TARGET_MEMORY_FRACTION = 0.85
"""Default target GPU memory fraction for batch size optimization."""

DEFAULT_NUM_GPUS = 1.0
"""Default number of GPUs to use for distributed processing."""

# API Token Defaults (used when config file is unavailable)
DEFAULT_MAX_TOKENS = 2000
"""Default max_tokens for regular text extraction."""

SPECIAL_BLOCK_MAX_TOKENS = 3000
"""Default max_tokens for special blocks (table, formula, figure)."""

TEXT_CORRECTION_MAX_TOKENS = 4000
"""Default max_tokens for page-level text correction."""

DEFAULT_TEMPERATURE = 0.1
"""Default temperature for API calls (0.1 for mostly deterministic output)."""

ESTIMATED_IMAGE_TOKENS = 2000
"""Estimated tokens for image + text (used for Gemini rate limiting)."""

DEFAULT_ESTIMATED_TOKENS = 1000
"""Default estimated tokens for rate limiting calculations."""

# Detection Defaults (used when config file is unavailable)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
"""Default confidence threshold for layout detection."""

DEFAULT_OVERLAP_THRESHOLD = 0.7
"""Default overlap threshold for duplicate filtering."""

# =============================================================================
# Global Settings
# =============================================================================
MIN_BOX_SIZE = 10
"""Minimum bounding box dimension (width or height) in pixels."""

DEFAULT_IOU_THRESHOLD = 0.3
"""Default IoU threshold for block overlap matching."""

# =============================================================================
# Image Processing
# =============================================================================
MAX_IMAGE_DIMENSION = 1024
"""Maximum dimension (width or height) for images sent to VLM APIs."""

MIN_HEADER_FONT_SIZE = 12.0
"""Minimum font size to be considered a header."""

# Text Correction
TEXT_CORRECTION_TEMPERATURE = 0.0
"""Temperature for text correction API calls (0.0 for fully deterministic output)."""
