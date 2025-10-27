"""Backend validation and resolution for detectors, sorters, and recognizers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_models_config() -> dict[str, Any]:
    """Load models configuration from YAML file.

    Returns:
        Configuration dictionary with detectors, sorters, recognizers sections
    """
    config_path = Path("settings") / "models.yaml"
    try:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.warning("Models config file not found: %s", config_path)
        return {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse models config: %s", e)
        return {}
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to read models config: %s", e)
        return {}


def resolve_detector_backend(detector: str, backend: str | None) -> tuple[str | None, str | None]:
    """Resolve detector backend with validation.

    Args:
        detector: Detector name (e.g., "doclayout-yolo", "mineru-vlm")
        backend: User-specified backend (None for auto-selection)

    Returns:
        (resolved_backend, error_message)
        If successful: (backend_name, None)
        If failed: (default_or_none, error_message)

    Examples:
        >>> resolve_detector_backend("doclayout-yolo", None)
        ("pytorch", None)
        >>> resolve_detector_backend("mineru-vlm", "vllm")
        ("vllm", None)
        >>> resolve_detector_backend("doclayout-yolo", "vllm")
        ("pytorch", "Backend 'vllm' not supported...")
    """
    config = _load_models_config()
    detectors = config.get("detectors", {})

    if detector not in detectors:
        return None, f"Unknown detector: {detector}"

    detector_config = detectors[detector]
    supported_backends = detector_config.get("supported_backends", [])
    default_backend = detector_config.get("default_backend")

    # Rule-based/algorithm-based models don't use backends
    if not supported_backends:
        if backend is not None:
            logger.warning(
                "Detector '%s' does not use inference backend (rule-based/algorithm-based). "
                "Ignoring --detector-backend=%s",
                detector,
                backend,
            )
        return None, None

    # Auto-select backend
    if backend is None:
        logger.info("Auto-selected detector backend: %s (for %s)", default_backend, detector)
        return default_backend, None

    # Validate user-specified backend
    if backend not in supported_backends:
        error_msg = (
            f"Backend '{backend}' not supported for detector '{detector}'. "
            f"Supported backends: {supported_backends}. "
            f"Using default: {default_backend}"
        )
        logger.error(error_msg)
        return default_backend, error_msg

    return backend, None


def resolve_sorter_backend(sorter: str, backend: str | None) -> tuple[str | None, str | None]:
    """Resolve sorter backend with validation.

    Args:
        sorter: Sorter name (e.g., "pymupdf", "mineru-vlm")
        backend: User-specified backend (None for auto-selection)

    Returns:
        (resolved_backend, error_message)
        If successful: (backend_name, None)
        If failed: (default_or_none, error_message)

    Examples:
        >>> resolve_sorter_backend("pymupdf", None)
        (None, None)  # Rule-based, no backend
        >>> resolve_sorter_backend("mineru-vlm", "vllm")
        ("vllm", None)
        >>> resolve_sorter_backend("mineru-layoutreader", "vllm")
        ("hf", "Backend 'vllm' not supported...")
    """
    config = _load_models_config()
    sorters = config.get("sorters", {})

    if sorter not in sorters:
        return None, f"Unknown sorter: {sorter}"

    sorter_config = sorters[sorter]
    supported_backends = sorter_config.get("supported_backends", [])
    default_backend = sorter_config.get("default_backend")

    # Rule-based/algorithm-based models don't use backends
    if not supported_backends:
        if backend is not None:
            logger.warning(
                "Sorter '%s' does not use inference backend (rule-based/algorithm-based). Ignoring --sorter-backend=%s",
                sorter,
                backend,
            )
        return None, None

    # Auto-select backend
    if backend is None:
        logger.info("Auto-selected sorter backend: %s (for %s)", default_backend, sorter)
        return default_backend, None

    # Validate user-specified backend
    if backend not in supported_backends:
        error_msg = (
            f"Backend '{backend}' not supported for sorter '{sorter}'. "
            f"Supported backends: {supported_backends}. "
            f"Using default: {default_backend}"
        )
        logger.error(error_msg)
        return default_backend, error_msg

    return backend, None


def resolve_recognizer_backend(recognizer: str, backend: str | None) -> tuple[str | None, str | None]:
    """Resolve recognizer backend with validation.

    For API-based recognizers (openai, gemini), auto-infers backend from recognizer name.

    Args:
        recognizer: Recognizer name (e.g., "openai", "gemini", "paddleocr-vl", "gemini-2.5-flash")
        backend: User-specified backend (None for auto-selection)

    Returns:
        (resolved_backend, error_message)
        If successful: (backend_name, None)
        If failed: (default_or_none, error_message)

    Examples:
        >>> resolve_recognizer_backend("openai", None)
        ("openai", None)  # API recognizer, auto-inferred
        >>> resolve_recognizer_backend("gemini", None)
        ("gemini", None)  # API recognizer, auto-inferred
        >>> resolve_recognizer_backend("paddleocr-vl", "vllm")
        ("vllm", None)
        >>> resolve_recognizer_backend("gemini-2.5-flash", None)
        ("gemini", None)  # Model name implies gemini backend
    """
    config = _load_models_config()
    recognizers = config.get("recognizers", {})

    # Handle model names that imply API backend (e.g., "gemini-2.5-flash", "gpt-4o")
    # Infer recognizer type from model name
    inferred_recognizer = None
    if recognizer.startswith("gemini"):
        inferred_recognizer = "gemini"
    elif recognizer.startswith("gpt-") or recognizer in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
        inferred_recognizer = "openai"
    elif recognizer == "paddleocr-vl" or recognizer.startswith("PaddlePaddle/PaddleOCR"):
        inferred_recognizer = "paddleocr-vl"

    # Use inferred recognizer if available
    lookup_recognizer = inferred_recognizer or recognizer

    if lookup_recognizer not in recognizers:
        # If not found in config, try direct lookup (for backward compatibility)
        if recognizer in recognizers:
            lookup_recognizer = recognizer
        else:
            return (
                None,
                f"Unknown recognizer: {recognizer}. Available: {list(recognizers.keys())}",
            )

    recognizer_config = recognizers[lookup_recognizer]
    supported_backends = recognizer_config.get("supported_backends", [])
    default_backend = recognizer_config.get("default_backend")

    # Auto-select backend
    if backend is None:
        if default_backend:
            logger.info("Auto-selected recognizer backend: %s (for %s)", default_backend, recognizer)
            return default_backend, None
        return None, f"No default backend for recognizer: {recognizer}"

    # Validate user-specified backend
    if backend not in supported_backends:
        error_msg = (
            f"Backend '{backend}' not supported for recognizer '{recognizer}'. "
            f"Supported backends: {supported_backends}. "
            f"Using default: {default_backend}"
        )
        logger.error(error_msg)
        return default_backend, error_msg

    return backend, None


def get_model_info(stage: str, model_name: str) -> dict[str, Any] | None:
    """Get model configuration information.

    Args:
        stage: "detector", "sorter", or "recognizer"
        model_name: Model name to look up

    Returns:
        Model configuration dict, or None if not found

    Examples:
        >>> get_model_info("detector", "doclayout-yolo")
        {'default_model': 'models/...', 'supported_backends': ['pytorch'], ...}
    """
    config = _load_models_config()
    stage_key = f"{stage}s"  # detector -> detectors

    if stage_key not in config:
        return None

    return config[stage_key].get(model_name)
