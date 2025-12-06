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


def _infer_backend_from_model_pattern(recognizer: str) -> str | None:
    """Infer backend from model name pattern.

    Args:
        recognizer: Model name

    Returns:
        Inferred backend name or None if no pattern matches
    """
    # Gemini models (gemini-2.5-flash, gemini-2.0-pro, etc.)
    if recognizer.startswith("gemini"):
        return "gemini"

    # GPT/ChatGPT models (gpt-4o, gpt-4-turbo, chatgpt, etc.)
    if recognizer.startswith(("gpt-", "chatgpt")):
        return "openai"

    # HuggingFace format: {org}/{model_name} → OpenRouter via openai backend
    if "/" in recognizer:
        return "openai"

    return None


def _infer_recognizer_type(recognizer: str) -> str | None:
    """Infer recognizer type from model name for config lookup.

    Args:
        recognizer: Model name

    Returns:
        Recognizer type for config lookup or None
    """
    pattern_map = {
        "gemini": lambda r: r.startswith("gemini"),
        "openai": lambda r: r.startswith(("gpt-", "chatgpt")),
        "paddleocr-vl": lambda r: r == "paddleocr-vl" or r.startswith("PaddlePaddle/PaddleOCR"),
        "deepseek-ocr": lambda r: r == "deepseek-ocr" or r.startswith("deepseek-ai/DeepSeek-OCR"),
    }

    for recognizer_type, matcher in pattern_map.items():
        if matcher(recognizer):
            return recognizer_type

    return None


def _lookup_recognizer_config(
    recognizer: str, recognizers: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Lookup recognizer configuration from models.yaml.

    Args:
        recognizer: Model name
        recognizers: Recognizers config dictionary

    Returns:
        (recognizer_config, error_message) - config is None if not found
    """
    # Try inferred type first
    inferred_type = _infer_recognizer_type(recognizer)
    lookup_key = inferred_type or recognizer

    if lookup_key in recognizers:
        return recognizers[lookup_key], None

    # Fallback to direct lookup
    if recognizer in recognizers:
        return recognizers[recognizer], None

    # HuggingFace format fallback - return None config but no error
    if "/" in recognizer:
        return None, None

    return None, f"Unknown recognizer: {recognizer}. Available: {list(recognizers.keys())}"


def _resolve_backend_for_config(
    backend: str | None,
    recognizer_config: dict[str, Any],
    recognizer: str,
) -> tuple[str | None, str | None]:
    """Resolve backend using recognizer config.

    Args:
        backend: User-specified backend (None for auto-select)
        recognizer_config: Config from models.yaml
        recognizer: Original recognizer name (for logging)

    Returns:
        (resolved_backend, error_message)
    """
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


def resolve_recognizer_backend(recognizer: str, backend: str | None) -> tuple[str | None, str | None]:
    """Resolve recognizer backend with validation and auto-inference.

    Auto-infers backend based on recognizer name pattern:
    - Gemini models (gemini-*) → gemini backend
    - GPT/ChatGPT models (gpt-*, chatgpt*) → openai backend
    - HuggingFace format ({org}/{model}) → openai backend (for OpenRouter)
    - Registered models (paddleocr-vl, deepseek-ocr) → default_backend from models.yaml

    Args:
        recognizer: Recognizer name (e.g., "gemini-2.5-flash", "gpt-4o", "meta-llama/Llama-3-8b")
        backend: User-specified backend (None for auto-selection)

    Returns:
        (resolved_backend, error_message)
        If successful: (backend_name, None)
        If failed: (default_or_none, error_message)
    """
    # Step 1: Try pattern-based inference for API models (backend=None only)
    if backend is None:
        inferred = _infer_backend_from_model_pattern(recognizer)
        if inferred:
            logger.info("Auto-selected recognizer backend: %s (for model: %s)", inferred, recognizer)
            return inferred, None

    # Step 2: Lookup in models.yaml config
    config = _load_models_config()
    recognizers = config.get("recognizers", {})

    recognizer_config, error = _lookup_recognizer_config(recognizer, recognizers)

    if error:
        return None, error

    # Step 3: Handle HuggingFace format without config (fallback to openai)
    if recognizer_config is None:
        if "/" in recognizer:
            logger.info("Using openai backend for HuggingFace model via OpenRouter: %s", recognizer)
            return "openai", None
        return None, f"No configuration found for recognizer: {recognizer}"

    # Step 4: Resolve backend using config
    return _resolve_backend_for_config(backend, recognizer_config, recognizer)


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
