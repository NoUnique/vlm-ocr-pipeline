#!/usr/bin/env python3
"""Render markdown/plaintext from existing JSON output files.

Usage:
    python scripts/render_from_json.py --input output/doc/json/page_1.json
    python scripts/render_from_json.py --input output/doc/json/
    python scripts/render_from_json.py --input output/doc/json/ --renderer plaintext
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.stages.rendering_stage import RenderingStage
from pipeline.types import Page

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_page_from_json(json_path: Path) -> Page:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return Page.from_dict(data)


def render_page(page: Page, renderer: str, image_render_mode: str) -> str:
    rendering_stage = RenderingStage(renderer=renderer, image_render_mode=image_render_mode)
    return rendering_stage.process(page.blocks)


def process_json_file(json_path: Path, output_dir: Path | None, renderer: str, image_render_mode: str) -> None:
    logger.info("Processing %s", json_path)
    page = load_page_from_json(json_path)
    rendered = render_page(page, renderer, image_render_mode)

    output_path = (output_dir / json_path.stem) if output_dir else (json_path.parent.parent / json_path.stem)
    ext = ".md" if renderer == "markdown" else ".txt"
    output_path = output_path.with_suffix(ext)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)
    logger.info("Saved to %s", output_path)


def process_directory(input_dir: Path, output_dir: Path | None, renderer: str, image_render_mode: str) -> None:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", input_dir)
        return

    logger.info("Found %d JSON files", len(json_files))
    for json_path in json_files:
        if json_path.stem.startswith("summary"):
            continue
        process_json_file(json_path, output_dir, renderer, image_render_mode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render from existing JSON files")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file or directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--renderer", choices=["markdown", "plaintext"], default="markdown")
    parser.add_argument(
        "--image-render-mode",
        choices=["image_only", "image_and_description", "description_only"],
        default="image_and_description",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None

    if not input_path.exists():
        logger.error("Input does not exist: %s", input_path)
        return 1

    if input_path.is_file():
        process_json_file(input_path, output_dir, args.renderer, args.image_render_mode)
    else:
        process_directory(input_path, output_dir, args.renderer, args.image_render_mode)

    logger.info("Rendering complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
