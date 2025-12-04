"""Pipeline configuration module.

This module provides:
- PipelineConfig: Dataclass for all pipeline configuration options
- ConfigLoader: YAML configuration file loader
- Validation for detector/sorter/recognizer combinations
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file with error handling.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dict, or empty dict if file not found or invalid
    """
    try:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.debug("Config file not found: %s", config_path)
        return {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return {}
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Failed to read config file %s: %s", config_path, e)
        return {}


@dataclass
class PipelineConfig:
    """Pipeline configuration with validation.

    This dataclass encapsulates all configuration options for the VLM OCR pipeline,
    providing type safety, validation, and easy serialization.

    Configuration Sources (in order of precedence):
    1. Constructor arguments (highest priority)
    2. CLI arguments via from_cli()
    3. YAML configuration files via from_yaml()
    4. Default values (lowest priority)

    Example:
        >>> # Direct construction
        >>> config = PipelineConfig(detector="paddleocr-doclayout-v2")
        >>> config.validate()

        >>> # From CLI arguments
        >>> args = parser.parse_args()
        >>> config = PipelineConfig.from_cli(args)

        >>> # From YAML file
        >>> config = PipelineConfig.from_yaml(Path("settings/config.yaml"))
    """

    # ==================== Detection Stage ====================
    detector: str = "paddleocr-doclayout-v2"
    detector_backend: str | None = None
    detector_model_path: str | Path | None = None
    confidence_threshold: float | None = None  # None = load from config

    # ==================== Batch Processing ====================
    auto_batch_size: bool = False
    batch_size: int | None = None
    target_memory_fraction: float = 0.85

    # ==================== Ordering Stage ====================
    sorter: str | None = None  # None = auto-select based on detector
    sorter_backend: str | None = None
    sorter_model_path: str | Path | None = None

    # ==================== Recognition Stage ====================
    recognizer: str = "paddleocr-vl"
    recognizer_backend: str | None = None

    # ==================== API Options ====================
    gemini_tier: str = "free"

    # ==================== Output Options ====================
    renderer: str = "markdown"

    # ==================== Performance Options ====================
    use_async: bool = False

    # ==================== DPI Options ====================
    dpi: int | None = None
    detection_dpi: int | None = None
    recognition_dpi: int | None = None
    use_dual_resolution: bool = False

    # ==================== Path Options ====================
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    temp_dir: Path = field(default_factory=lambda: Path(".tmp"))

    # ==================== Caching Options ====================
    use_cache: bool = True

    # ==================== Internal State (populated during validation) ====================
    # These are resolved during validate() and used internally
    _resolved_detector_backend: str | None = field(default=None, repr=False)
    _resolved_sorter: str | None = field(default=None, repr=False)
    _resolved_sorter_backend: str | None = field(default=None, repr=False)
    _resolved_recognizer_backend: str | None = field(default=None, repr=False)
    _models_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _detection_config: dict[str, Any] = field(default_factory=dict, repr=False)
    _pipeline_config: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Convert path strings to Path objects."""
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.temp_dir, str):
            self.temp_dir = Path(self.temp_dir)
        if isinstance(self.detector_model_path, str):
            self.detector_model_path = Path(self.detector_model_path)
        if isinstance(self.sorter_model_path, str):
            self.sorter_model_path = Path(self.sorter_model_path)

    @classmethod
    def from_yaml(cls, config_path: Path, **overrides: Any) -> PipelineConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
            **overrides: Values to override from file

        Returns:
            PipelineConfig instance

        Example:
            >>> config = PipelineConfig.from_yaml(Path("settings/config.yaml"))
            >>> config = PipelineConfig.from_yaml(
            ...     Path("settings/config.yaml"),
            ...     detector="mineru-vlm"
            ... )
        """
        yaml_config = _load_yaml_config(config_path)

        # Map YAML keys to dataclass field names
        field_mapping = {
            "confidence_threshold": "confidence_threshold",
            "use_cache": "use_cache",
            "cache_dir": "cache_dir",
            "output_dir": "output_dir",
            "temp_dir": "temp_dir",
            "detector": "detector",
            "detector_backend": "detector_backend",
            "detector_model_path": "detector_model_path",
            "auto_batch_size": "auto_batch_size",
            "batch_size": "batch_size",
            "target_memory_fraction": "target_memory_fraction",
            "sorter": "sorter",
            "sorter_backend": "sorter_backend",
            "sorter_model_path": "sorter_model_path",
            "recognizer": "recognizer",
            "recognizer_backend": "recognizer_backend",
            "gemini_tier": "gemini_tier",
            "renderer": "renderer",
            "use_async": "use_async",
            "dpi": "dpi",
            "detection_dpi": "detection_dpi",
            "recognition_dpi": "recognition_dpi",
            "use_dual_resolution": "use_dual_resolution",
        }

        # Extract values from YAML
        kwargs: dict[str, Any] = {}
        for yaml_key, field_name in field_mapping.items():
            if yaml_key in yaml_config:
                kwargs[field_name] = yaml_config[yaml_key]

        # Apply overrides
        kwargs.update(overrides)

        return cls(**kwargs)

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> PipelineConfig:
        """Create configuration from CLI arguments.

        Args:
            args: Parsed CLI arguments from argparse

        Returns:
            PipelineConfig instance

        Example:
            >>> parser = argparse.ArgumentParser()
            >>> # ... add arguments ...
            >>> args = parser.parse_args()
            >>> config = PipelineConfig.from_cli(args)
        """
        # Map CLI args to config fields (only include non-None values)
        kwargs: dict[str, Any] = {}

        # Detection
        if hasattr(args, "detector") and args.detector is not None:
            kwargs["detector"] = args.detector
        if hasattr(args, "detector_backend") and args.detector_backend is not None:
            kwargs["detector_backend"] = args.detector_backend
        if hasattr(args, "detector_model_path") and args.detector_model_path is not None:
            kwargs["detector_model_path"] = args.detector_model_path
        if hasattr(args, "confidence") and args.confidence is not None:
            kwargs["confidence_threshold"] = args.confidence

        # Batch processing
        if hasattr(args, "auto_batch_size") and args.auto_batch_size:
            kwargs["auto_batch_size"] = args.auto_batch_size
        if hasattr(args, "batch_size") and args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
        if hasattr(args, "target_memory_fraction") and args.target_memory_fraction is not None:
            kwargs["target_memory_fraction"] = args.target_memory_fraction

        # Ordering
        if hasattr(args, "sorter") and args.sorter is not None:
            kwargs["sorter"] = args.sorter
        if hasattr(args, "sorter_backend") and args.sorter_backend is not None:
            kwargs["sorter_backend"] = args.sorter_backend
        if hasattr(args, "sorter_model_path") and args.sorter_model_path is not None:
            kwargs["sorter_model_path"] = args.sorter_model_path

        # Recognition
        if hasattr(args, "recognizer") and args.recognizer is not None:
            kwargs["recognizer"] = args.recognizer
        if hasattr(args, "recognizer_backend") and args.recognizer_backend is not None:
            kwargs["recognizer_backend"] = args.recognizer_backend
        if hasattr(args, "gemini_tier") and args.gemini_tier is not None:
            kwargs["gemini_tier"] = args.gemini_tier

        # Output
        if hasattr(args, "output") and args.output is not None:
            kwargs["output_dir"] = Path(args.output)
        if hasattr(args, "renderer") and args.renderer is not None:
            kwargs["renderer"] = args.renderer

        # Paths
        if hasattr(args, "cache_dir") and args.cache_dir is not None:
            kwargs["cache_dir"] = Path(args.cache_dir)
        if hasattr(args, "temp_dir") and args.temp_dir is not None:
            kwargs["temp_dir"] = Path(args.temp_dir)
        if hasattr(args, "no_cache") and args.no_cache:
            kwargs["use_cache"] = False

        # DPI
        if hasattr(args, "dpi") and args.dpi is not None:
            kwargs["dpi"] = args.dpi
        if hasattr(args, "detection_dpi") and args.detection_dpi is not None:
            kwargs["detection_dpi"] = args.detection_dpi
        if hasattr(args, "recognition_dpi") and args.recognition_dpi is not None:
            kwargs["recognition_dpi"] = args.recognition_dpi
        if hasattr(args, "use_dual_resolution") and args.use_dual_resolution:
            kwargs["use_dual_resolution"] = args.use_dual_resolution

        return cls(**kwargs)

    def validate(self) -> None:
        """Validate configuration and resolve auto-selections.

        This method:
        1. Loads YAML configuration files
        2. Resolves DPI settings from config
        3. Validates and resolves backend selections
        4. Validates detector/sorter combinations
        5. Validates renderer option

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = PipelineConfig(detector="invalid-detector")
            >>> config.validate()  # Raises ValueError
        """
        # Load configuration files
        self._load_config_files()

        # Resolve DPI settings
        self._resolve_dpi_settings()

        # Resolve and validate backends
        self._resolve_backends()

        # Resolve and validate detector/sorter combination
        self._resolve_detector_sorter_combination()

        # Validate renderer
        self._validate_renderer()

        # Resolve confidence threshold from config if not set
        self._resolve_confidence_threshold()

        logger.info(
            "Configuration validated: detector=%s (backend=%s), sorter=%s (backend=%s), "
            "recognizer=%s (backend=%s)",
            self.detector,
            self._resolved_detector_backend or "auto",
            self._resolved_sorter,
            self._resolved_sorter_backend or "auto",
            self.recognizer,
            self._resolved_recognizer_backend,
        )

    def _load_config_files(self) -> None:
        """Load YAML configuration files."""
        settings_dir = Path("settings")
        self._models_config = _load_yaml_config(settings_dir / "models.yaml")
        self._detection_config = _load_yaml_config(settings_dir / "detection_config.yaml")
        self._pipeline_config = _load_yaml_config(settings_dir / "config.yaml")

    def _resolve_dpi_settings(self) -> None:
        """Resolve DPI settings from config files."""
        conversion_config = self._pipeline_config.get("conversion", {})

        if self.dpi is None:
            self.dpi = conversion_config.get("default_dpi", 200)
        if self.detection_dpi is None:
            self.detection_dpi = conversion_config.get("detection_dpi", 150)
        if self.recognition_dpi is None:
            self.recognition_dpi = conversion_config.get("recognition_dpi", 300)
        if not self.use_dual_resolution:
            self.use_dual_resolution = conversion_config.get("use_dual_resolution", False)

        logger.info(
            "DPI configuration: default=%d, detection=%d, recognition=%d, dual_resolution=%s",
            self.dpi,
            self.detection_dpi,
            self.recognition_dpi,
            self.use_dual_resolution,
        )

    def _resolve_backends(self) -> None:
        """Resolve and validate all backend selections."""
        from .backend_validator import (
            resolve_detector_backend,
            resolve_recognizer_backend,
            resolve_sorter_backend,
        )

        # Resolve detector backend
        self._resolved_detector_backend, detector_error = resolve_detector_backend(
            self.detector, self.detector_backend
        )
        if detector_error:
            logger.warning("Detector backend validation: %s", detector_error)

        # Resolve recognizer backend
        self._resolved_recognizer_backend, recognizer_error = resolve_recognizer_backend(
            self.recognizer, self.recognizer_backend
        )
        if recognizer_error:
            raise ValueError(f"Recognizer backend validation failed: {recognizer_error}")
        if self._resolved_recognizer_backend is None:
            raise ValueError(f"No backend available for recognizer: {self.recognizer}")

        # Resolve sorter backend (after sorter is resolved)
        # This is called again in _resolve_detector_sorter_combination after sorter is determined
        if self.sorter is not None:
            self._resolved_sorter_backend, sorter_error = resolve_sorter_backend(
                self.sorter, self.sorter_backend
            )
            if sorter_error:
                logger.warning("Sorter backend validation: %s", sorter_error)

    def _resolve_detector_sorter_combination(self) -> None:
        """Resolve and validate detector/sorter combination."""
        from .backend_validator import resolve_sorter_backend
        from .layout.ordering import REQUIRED_COMBINATIONS, validate_combination

        detector = self.detector
        sorter = self.sorter

        # Default detector for comparison
        detector_default = "doclayout-yolo"
        detector_is_default = detector == detector_default

        # Case 1: Sorter is specified and requires a specific detector
        if sorter is not None and sorter in REQUIRED_COMBINATIONS:
            required_detector = REQUIRED_COMBINATIONS[sorter]
            if not detector_is_default and detector != required_detector:
                raise ValueError(
                    f"Sorter '{sorter}' requires detector '{required_detector}' (tightly coupled), "
                    f"but detector '{detector}' was specified. "
                    f"Either omit --detector or use --detector {required_detector}."
                )
            if detector_is_default:
                self.detector = required_detector
                detector = required_detector
                logger.info(
                    "Auto-selected detector='%s' for '%s' sorter (tightly coupled)",
                    detector,
                    sorter,
                )

        # Case 2: Sorter is not specified, auto-select based on detector
        if sorter is None:
            if detector == "paddleocr-doclayout-v2":
                sorter = "paddleocr-doclayout-v2"
                logger.info("Auto-selected sorter='paddleocr-doclayout-v2' for paddleocr-doclayout-v2 detector")
            elif detector == "mineru-vlm":
                sorter = "mineru-vlm"
                logger.info("Auto-selected sorter='mineru-vlm' for mineru-vlm detector")
            else:
                sorter = "mineru-xycut"
                logger.info("Using default sorter='mineru-xycut' (fast and accurate)")

        self._resolved_sorter = sorter

        # Resolve sorter backend now that sorter is determined
        self._resolved_sorter_backend, sorter_error = resolve_sorter_backend(
            sorter, self.sorter_backend
        )
        if sorter_error:
            logger.warning("Sorter backend validation: %s", sorter_error)

        # Validate combination
        is_valid, message = validate_combination(detector, sorter)
        if not is_valid:
            raise ValueError(f"Invalid detector/sorter combination: {message}")

        logger.info("Pipeline combination: %s", message)

        # Check PyMuPDF availability
        if sorter == "pymupdf":
            try:
                import fitz  # noqa: F401
            except ImportError:
                logger.warning("PyMuPDF is not installed. Falling back to sorter='mineru-xycut'")
                self._resolved_sorter = "mineru-xycut"

    def _validate_renderer(self) -> None:
        """Validate renderer option."""
        valid_renderers = ["markdown", "plaintext"]
        if self.renderer.lower() not in valid_renderers:
            raise ValueError(
                f"Invalid renderer: {self.renderer}. Must be one of: {valid_renderers}"
            )
        self.renderer = self.renderer.lower()

    def _resolve_confidence_threshold(self) -> None:
        """Resolve confidence threshold from config if not set."""
        from .constants import DEFAULT_CONFIDENCE_THRESHOLD

        if self.confidence_threshold is None:
            detectors_config = self._detection_config.get("detectors", {})
            detector_config = detectors_config.get(self.detector, {})
            self.confidence_threshold = detector_config.get(
                "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
            )
            logger.debug(
                "Using confidence_threshold=%.2f from config for detector=%s",
                self.confidence_threshold,
                self.detector,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            # Detection
            "detector": self.detector,
            "detector_backend": self.detector_backend,
            "detector_model_path": str(self.detector_model_path) if self.detector_model_path else None,
            "confidence_threshold": self.confidence_threshold,
            # Batch
            "auto_batch_size": self.auto_batch_size,
            "batch_size": self.batch_size,
            "target_memory_fraction": self.target_memory_fraction,
            # Ordering
            "sorter": self.sorter,
            "sorter_backend": self.sorter_backend,
            "sorter_model_path": str(self.sorter_model_path) if self.sorter_model_path else None,
            # Recognition
            "recognizer": self.recognizer,
            "recognizer_backend": self.recognizer_backend,
            "gemini_tier": self.gemini_tier,
            # Output
            "renderer": self.renderer,
            # Performance
            "use_async": self.use_async,
            # DPI
            "dpi": self.dpi,
            "detection_dpi": self.detection_dpi,
            "recognition_dpi": self.recognition_dpi,
            "use_dual_resolution": self.use_dual_resolution,
            # Paths
            "cache_dir": str(self.cache_dir),
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            # Cache
            "use_cache": self.use_cache,
        }

    # ==================== Resolved Property Accessors ====================

    @property
    def resolved_detector_backend(self) -> str | None:
        """Get resolved detector backend (after validation)."""
        return self._resolved_detector_backend

    @property
    def resolved_sorter(self) -> str:
        """Get resolved sorter name (after validation)."""
        if self._resolved_sorter is None:
            raise RuntimeError("Configuration not validated. Call validate() first.")
        return self._resolved_sorter

    @property
    def resolved_sorter_backend(self) -> str | None:
        """Get resolved sorter backend (after validation)."""
        return self._resolved_sorter_backend

    @property
    def resolved_recognizer_backend(self) -> str:
        """Get resolved recognizer backend (after validation)."""
        if self._resolved_recognizer_backend is None:
            raise RuntimeError("Configuration not validated. Call validate() first.")
        return self._resolved_recognizer_backend

    @property
    def models_config(self) -> dict[str, Any]:
        """Get loaded models configuration."""
        return self._models_config

    @property
    def detection_config(self) -> dict[str, Any]:
        """Get loaded detection configuration."""
        return self._detection_config

    @property
    def pipeline_yaml_config(self) -> dict[str, Any]:
        """Get loaded pipeline YAML configuration."""
        return self._pipeline_config

