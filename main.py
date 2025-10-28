#!/usr/bin/env python3
"""
Main entry point for the VLM OCR Pipeline
Provides command-line interface for processing images and PDFs using Vision Language Models
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Note: Pipeline and related imports are moved to function-level
# to improve CLI startup time (--help, argument validation, etc.)
# Type hints use TYPE_CHECKING to avoid runtime import cost
if TYPE_CHECKING:
    from pipeline import Pipeline


def parse_dpi_config(dpi_arg: str | None) -> tuple[int | None, int | None, int | None, bool]:
    """Parse DPI configuration from CLI argument.

    Supports three formats:
    1. Presets: "fast", "balanced", "quality"
    2. Single DPI: "200"
    3. Dual resolution: "150,300" (detection,recognition)

    Args:
        dpi_arg: DPI argument from CLI (None, preset, single, or comma-separated)

    Returns:
        Tuple of (dpi, detection_dpi, recognition_dpi, use_dual_resolution)

    Examples:
        >>> parse_dpi_config(None)
        (None, None, None, False)
        >>> parse_dpi_config("fast")
        (150, 150, 150, False)
        >>> parse_dpi_config("balanced")
        (200, 150, 300, True)
        >>> parse_dpi_config("quality")
        (300, 300, 300, False)
        >>> parse_dpi_config("200")
        (200, 200, 200, False)
        >>> parse_dpi_config("150,300")
        (225, 150, 300, True)
    """
    if dpi_arg is None:
        return (None, None, None, False)

    dpi_lower = dpi_arg.lower().strip()

    # Preset configurations
    presets = {
        "fast": (150, 150, 150, False),  # Fast processing, lower quality
        "balanced": (200, 150, 300, True),  # Balanced: fast detection, accurate recognition
        "quality": (300, 300, 300, False),  # High quality, slower processing
    }

    if dpi_lower in presets:
        return presets[dpi_lower]

    # Dual resolution: "150,300"
    if "," in dpi_arg:
        try:
            parts = dpi_arg.split(",")
            if len(parts) != 2:  # noqa: PLR2004
                raise ValueError(f"Invalid dual DPI format: {dpi_arg}. Expected format: '150,300'")
            detection_dpi = int(parts[0].strip())
            recognition_dpi = int(parts[1].strip())
            # Use average as default DPI
            avg_dpi = (detection_dpi + recognition_dpi) // 2
            return (avg_dpi, detection_dpi, recognition_dpi, True)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid dual DPI format: {dpi_arg}. Error: {e}") from e

    # Single DPI value: "200"
    try:
        single_dpi = int(dpi_arg)
        return (single_dpi, single_dpi, single_dpi, False)
    except ValueError as e:
        raise ValueError(
            f"Invalid DPI value: {dpi_arg}. "
            "Expected preset (fast/balanced/quality), single value (200), or dual (150,300)"
        ) from e


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration with timestamped log files."""
    from pipeline.misc import tz_now  # noqa: PLC0415 - lazy import for startup performance

    logs_dir = Path(".logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = tz_now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = logs_dir / f"{timestamp}_ocr_pipeline.log"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_filename, encoding="utf-8")],
    )


def parse_page_range(page_range_str: str) -> tuple[int, int] | None:
    """Parse page range string like '1-10' into (1, 10)."""
    try:
        if "-" not in page_range_str:
            raise ValueError("Page range must contain '-'")

        start_str, end_str = page_range_str.split("-", 1)
        start_page = int(start_str.strip())
        end_page = int(end_str.strip())

        if start_page < 1 or end_page < 1 or start_page > end_page:
            raise ValueError("Invalid page range")

        return (start_page, end_page)
    except ValueError as exc:
        logging.getLogger(__name__).error("Invalid page range format '%s': %s", page_range_str, exc)
        return None


def parse_specific_pages(pages_str: str) -> list[int] | None:
    """Parse comma-separated pages like '1,3,5,10' into [1, 3, 5, 10]."""
    try:
        pages: list[int] = []
        for page_str in pages_str.split(","):
            page_num = int(page_str.strip())
            if page_num < 1:
                raise ValueError(f"Page number must be positive: {page_num}")
            pages.append(page_num)

        if not pages:
            raise ValueError("No valid page numbers found")

        return sorted(set(pages))
    except ValueError as exc:
        logging.getLogger(__name__).error("Invalid pages format '%s': %s", pages_str, exc)
        return None


def main() -> int:
    """CLI entry point."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    return _execute_command(args, parser, logger)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VLM OCR Pipeline - Process images and PDFs with layout detection and VLM-powered text correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Basic usage (default: doclayout-yolo detector + gemini recognizer)
              python main.py --input document.pdf

              # Different recognizers
              python main.py --input document.pdf --recognizer gpt-4o
              python main.py --input document.pdf --recognizer paddleocr-vl

              # Custom backends
              python main.py --input document.pdf \
                  --detector mineru-vlm --detector-backend vllm \
                  --recognizer paddleocr-vl --recognizer-backend sglang

              # Traditional pipeline with ordering
              python main.py --input document.pdf --sorter pymupdf
              python main.py --input document.pdf --sorter mineru-xycut

              # MinerU VLM (detection + ordering)
              python main.py --input document.pdf \
                  --detector mineru-vlm --sorter mineru-vlm

              # Advanced options
              python main.py --input /path/to/images/
              python main.py --input document.pdf --output /custom/output/
              python main.py --input document.pdf --max-pages 5
              python main.py --input document.pdf --page-range 10-20
              python main.py --input document.pdf --pages 1,5,10,15
              python main.py --input document.pdf --no-cache --confidence 0.7
            """
        ),
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file or directory path (PDF, image, or directory containing PDFs)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output",
        help="Output directory path (default: ./output)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable caching (default: caching enabled)")
    parser.add_argument("--cache-dir", type=str, default=".cache", help="Cache directory path (default: ./.cache)")
    parser.add_argument("--temp-dir", type=str, default=".tmp", help="Temporary files directory path (default: ./.tmp)")

    # DPI Settings
    parser.add_argument(
        "--dpi",
        type=str,
        help=(
            "DPI for PDF-to-image conversion. Supports: "
            "presets (fast/balanced/quality), "
            "single value (200), "
            "or dual resolution (150,300 for detection,recognition)"
        ),
    )

    page_group = parser.add_mutually_exclusive_group()
    page_group.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to process from the beginning (e.g., --max-pages 5)",
    )
    page_group.add_argument(
        "--page-range",
        type=str,
        help='Page range to process in format "start-end" (e.g., --page-range 1-10)',
    )
    page_group.add_argument(
        "--pages",
        type=str,
        help="Specific pages to process, comma-separated (e.g., --pages 1,3,5,10)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    # Layout Detection Stage
    detection_group = parser.add_argument_group("Layout Detection")
    detection_group.add_argument(
        "--detector",
        type=str,
        default="doclayout-yolo",
        help=(
            "Detector model name or alias (default: doclayout-yolo). "
            "Options: doclayout-yolo, mineru-doclayout-yolo, mineru-vlm, paddleocr-doclayout-v2, "
            "or HuggingFace model path (e.g., opendatalab/PDF-Extract-Kit-1.0)"
        ),
    )
    detection_group.add_argument(
        "--detector-backend",
        choices=["pytorch", "hf", "pt-ray", "hf-ray", "vllm", "sglang"],
        help="Inference backend for detector (auto-selected if not specified)",
    )
    detection_group.add_argument(
        "--detector-model-path",
        type=str,
        help="Custom detector model path (overrides model name resolution)",
    )
    detection_group.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="Auto-calibrate optimal batch size for detector (recommended for multi-image processing)",
    )
    detection_group.add_argument(
        "--batch-size",
        type=int,
        help="Manual batch size for detector (default: 1, ignored if --auto-batch-size is set)",
    )
    detection_group.add_argument(
        "--target-memory-fraction",
        type=float,
        default=0.85,
        help="Target GPU memory usage fraction for auto batch size calibration (default: 0.85)",
    )

    # Reading Order Stage
    ordering_group = parser.add_argument_group("Reading Order")
    ordering_group.add_argument(
        "--sorter",
        type=str,
        help=(
            "Sorter model name or alias (auto-selected if not specified). "
            "Options: pymupdf, mineru-xycut, mineru-layoutreader, mineru-vlm, olmocr-vlm, paddleocr-doclayout-v2, "
            "or HuggingFace model path (e.g., opendatalab/PDF-Extract-Kit-1.0)"
        ),
    )
    ordering_group.add_argument(
        "--sorter-backend",
        choices=["pytorch", "hf", "pt-ray", "hf-ray", "vllm", "sglang"],
        help="Inference backend for sorter (auto-selected if not specified)",
    )
    ordering_group.add_argument(
        "--sorter-model-path",
        type=str,
        help="Custom sorter model path (overrides model name resolution)",
    )

    # Text Recognition Stage
    recognition_group = parser.add_argument_group("Text Recognition")
    recognition_group.add_argument(
        "--recognizer",
        type=str,
        default="gemini-2.5-flash",
        help=(
            "Recognizer model name (default: gemini-2.5-flash). "
            "Examples: gemini-2.5-flash, gpt-4o, paddleocr-vl, "
            "or full model name (e.g., PaddlePaddle/PaddleOCR-VL-0.9B)"
        ),
    )
    recognition_group.add_argument(
        "--recognizer-backend",
        choices=["pytorch", "hf", "pt-ray", "hf-ray", "vllm", "sglang", "openai", "gemini"],
        help="Inference backend for recognizer (auto-selected if not specified)",
    )

    # API-specific options
    gemini_group = parser.add_argument_group("Gemini API Options")
    gemini_group.add_argument(
        "--gemini-tier",
        type=str,
        choices=["free", "tier1", "tier2", "tier3"],
        default="free",
        help="Gemini API tier for rate limiting (only for gemini-* models, default: free)",
    )

    parser.add_argument(
        "--rate-limit-status",
        action="store_true",
        help="Show current rate limit status and exit",
    )

    return parser


def _execute_command(args: argparse.Namespace, parser: argparse.ArgumentParser, logger: logging.Logger) -> int:
    if _handle_rate_limit_status(args):
        return 0

    if not args.input:
        parser.error("the following arguments are required: --input/-i")
        return 1  # pragma: no cover - parser.error raises SystemExit

    try:
        # Lazy import: only load Pipeline when actually processing input
        from pipeline import Pipeline  # noqa: PLC0415

        # Parse DPI configuration
        dpi, detection_dpi, recognition_dpi, use_dual_resolution = parse_dpi_config(args.dpi)

        pipeline = Pipeline(
            confidence_threshold=args.confidence,
            use_cache=not args.no_cache,
            cache_dir=args.cache_dir,
            output_dir=args.output,
            temp_dir=args.temp_dir,
            # Stage-specific options
            detector=args.detector,
            detector_backend=args.detector_backend,
            detector_model_path=args.detector_model_path,
            auto_batch_size=args.auto_batch_size,
            batch_size=args.batch_size,
            target_memory_fraction=args.target_memory_fraction,
            sorter=args.sorter,
            sorter_backend=args.sorter_backend,
            sorter_model_path=args.sorter_model_path,
            recognizer=args.recognizer,
            recognizer_backend=args.recognizer_backend,
            gemini_tier=args.gemini_tier,
            # DPI options
            dpi=dpi,
            detection_dpi=detection_dpi,
            recognition_dpi=recognition_dpi,
            use_dual_resolution=use_dual_resolution,
        )

        return _run_pipeline(pipeline, args, logger)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as exc:  # noqa: BLE001 - retain broad logging for CLI
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return 1


def _handle_rate_limit_status(args: argparse.Namespace) -> bool:
    if not args.rate_limit_status:
        return False

    # Check if using Gemini recognizer
    recognizer_backend = args.recognizer_backend
    if recognizer_backend is None:
        # Try to infer from recognizer name
        if args.recognizer.startswith("gemini"):
            recognizer_backend = "gemini"
        else:
            recognizer_backend = "unknown"

    if recognizer_backend != "gemini":
        print("\n⚠️  Rate limit status is only available for Gemini backend")
        print(f"Current recognizer backend: {recognizer_backend}")
        print("Use --recognizer gemini-2.5-flash or similar to use Gemini")
        return True

    from pipeline.recognition.api.ratelimit import rate_limiter  # noqa: PLC0415

    rate_limiter.set_tier_and_model(args.gemini_tier, args.recognizer)
    status = rate_limiter.get_status()
    _print_rate_limit_status(status)
    return True


def _print_rate_limit_status(status: dict[str, Any]) -> None:
    print("\n📊 Gemini API Rate Limit Status")
    print(f"{'=' * 50}")
    print(f"Tier: {status['tier']}")
    print(f"Current Model: {status['model']}")
    print("\n🔢 Current Model Usage:")
    limits = status.get("limits", {})
    current = status.get("current", {})
    print(f"  Requests per minute: {current.get('rpm')}/{limits.get('rpm') or 'unlimited'}")
    if limits.get("tpm"):
        print(f"  Tokens per minute: {current.get('tpm', 0):,}/{limits['tpm']:,}")
    else:
        print(f"  Tokens per minute: {current.get('tpm', 0):,}/unlimited")
    print(f"  Requests per day: {current.get('rpd')}/{limits.get('rpd') or 'unlimited'}")

    utilization = status.get("utilization", {})
    print("\n📈 Current Model Utilization:")
    print(f"  RPM: {utilization.get('rpm_percent', 0.0):.1f}%")
    print(f"  TPM: {utilization.get('tpm_percent', 0.0):.1f}%")
    print(f"  RPD: {utilization.get('rpd_percent', 0.0):.1f}%")

    all_models = status.get("all_models") or {}
    if len(all_models) > 1:
        print("\n📋 All Models Summary:")
        for model_name, model_usage in all_models.items():
            is_current = "← CURRENT" if model_name == status.get("model") else ""
            print(
                f"  {model_name}: RPM:{model_usage.get('rpm')} "
                f"TPM:{model_usage.get('tpm', 0):,} RPD:{model_usage.get('rpd')} {is_current}"
            )


def _run_pipeline(pipeline: Pipeline, args: argparse.Namespace, logger: logging.Logger) -> int:
    logger.info("Starting VLM OCR Pipeline")
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", args.output)
    logger.info("Detector: %s (backend: %s)", args.detector, args.detector_backend or "auto")
    logger.info("Sorter: %s (backend: %s)", pipeline.sorter_name, args.sorter_backend or "auto")
    logger.info("Recognizer: %s (backend: %s)", args.recognizer, args.recognizer_backend or "auto")
    if args.recognizer.startswith("gemini"):
        logger.info("Gemini Tier: %s", args.gemini_tier)
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    page_options = _parse_page_options(args, logger)
    if page_options is None:
        return 1

    result = _process_input_path(pipeline, input_path, args, page_options, logger)
    if result is None:
        return 1

    if "error" in result:
        logger.error("Processing failed: %s", result["error"])
        return 1

    _log_output_location(input_path, args, result, pipeline, logger)
    logger.info("VLM OCR Pipeline completed successfully")
    return 0


def _parse_page_options(
    args: argparse.Namespace, logger: logging.Logger
) -> tuple[int | None, tuple[int, int] | None, list[int] | None] | None:
    max_pages = args.max_pages
    page_range: tuple[int, int] | None = None
    specific_pages: list[int] | None = None

    if args.page_range:
        page_range = parse_page_range(args.page_range)
        if page_range is None:
            return None

    if args.pages:
        specific_pages = parse_specific_pages(args.pages)
        if specific_pages is None:
            return None

    if max_pages:
        logger.info("Limiting to maximum %d pages", max_pages)
    elif page_range:
        logger.info("Processing page range: %d-%d", page_range[0], page_range[1])
    elif specific_pages:
        logger.info("Processing specific pages: %s", specific_pages)

    return max_pages, page_range, specific_pages


def _process_input_path(
    pipeline: Pipeline,
    input_path: Path,
    args: argparse.Namespace,
    page_options: tuple[int | None, tuple[int, int] | None, list[int] | None],
    logger: logging.Logger,
) -> dict[str, Any] | None:
    max_pages, page_range, specific_pages = page_options

    if input_path.is_file():
        return _process_input_file(
            pipeline,
            input_path,
            args,
            max_pages,
            page_range,
            specific_pages,
            logger,
        )

    if input_path.is_dir():
        return _process_input_directory(
            pipeline,
            input_path,
            args,
            max_pages,
            page_range,
            specific_pages,
            logger,
        )

    logger.error("Invalid input path: %s", input_path)
    return None


def _process_input_file(
    pipeline: Pipeline,
    input_path: Path,
    args: argparse.Namespace,
    max_pages: int | None,
    page_range: tuple[int, int] | None,
    specific_pages: list[int] | None,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    file_ext = input_path.suffix.lower()

    if file_ext == ".pdf":
        logger.info("Processing PDF file: %s", input_path)
        document = pipeline.process_pdf(
            input_path,
            max_pages=max_pages,
            page_range=page_range,
            pages=specific_pages,
        )
        return document.to_dict() if document else None

    if file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
        from pipeline.types import Document  # noqa: PLC0415

        logger.info("Processing image file: %s", input_path)
        result = pipeline.process_image(input_path)

        model_output_dir = Path(args.output) / pipeline.model
        output_path = model_output_dir / f"{input_path.stem}.json"

        # Handle both Document and dict return types
        if result is None:
            return None
        result_dict: dict[str, Any] = result.to_dict() if isinstance(result, Document) else result
        pipeline._save_results(result_dict, output_path)
        return result_dict

    logger.error("Unsupported file format: %s", file_ext)
    return None


def _process_input_directory(
    pipeline: Pipeline,
    input_path: Path,
    args: argparse.Namespace,
    max_pages: int | None,
    page_range: tuple[int, int] | None,
    specific_pages: list[int] | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    logger.info("Processing directory: %s", input_path)

    if max_pages or page_range or specific_pages:
        logger.warning("Page limiting options are applied to each PDF individually in directory mode")

    return pipeline.process_directory(
        input_path,
        args.output,
        max_pages=max_pages,
        page_range=page_range,
        specific_pages=specific_pages,
    )


def _log_output_location(
    input_path: Path,
    args: argparse.Namespace,
    result: dict[str, Any],
    pipeline: Pipeline,
    logger: logging.Logger,
) -> None:
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            logger.info("Results saved to: %s", result.get("output_directory", args.output))
        else:
            logger.info("Results saved to: %s", Path(args.output) / pipeline.model)
    elif input_path.is_dir():
        logger.info("Results saved to: %s", result.get("output_directory", args.output))
    else:
        logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    sys.exit(main())
