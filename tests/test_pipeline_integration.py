from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest

from pipeline import Pipeline
from pipeline.conversion.converter import DocumentConverter
from pipeline.layout.detection import LayoutDetector
from pipeline.recognition import TextRecognizer
from pipeline.recognition.cache import RecognitionCache

EXPECTED_REGION_COUNT = 3
EXPECTED_PAGE_COUNT = 2
EXPECTED_STATUS_SUMMARY = {"complete": EXPECTED_PAGE_COUNT}


class FakePromptManager:
    def get_prompt(self, category, prompt_type, **kwargs):
        return f"{category}-{prompt_type}"

    def get_prompt_for_region_type(self, region_type: str) -> str:
        return f"prompt-{region_type}"

    def get_gemini_prompt_for_region_type(self, region_type: str) -> str:
        return f"prompt-{region_type}"


class FakeAIClient:
    def __init__(self):
        self.correct_calls: list[str] = []

    def extract_text(self, region_img, region_info, prompt):
        text = f"text-{region_info['type']}"
        return {
            "type": region_info["type"],
            "coords": region_info["coords"],
            "text": text,
        }

    def process_special_region(self, region_img, region_info, prompt):
        return {
            "type": region_info["type"],
            "coords": region_info["coords"],
            "structured_text": f"structured-{region_info['type']}",
        }

    def correct_text(self, text, system_prompt, user_prompt):
        self.correct_calls.append(text)
        return {"corrected_text": f"corrected-{text}", "confidence": 0.95}

    def is_available(self):
        return True


class FakeDocLayoutModel:
    def __init__(self, regions_per_call: list[list[dict]] | None = None):
        self.regions_per_call = regions_per_call or []
        self.call_count = 0

    def predict(self, image, conf=0.25):
        if not self.regions_per_call:
            return []
        index = min(self.call_count, len(self.regions_per_call) - 1)
        self.call_count += 1
        return copy.deepcopy(self.regions_per_call[index])


@pytest.fixture
def pipeline_fixture(tmp_path):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.backend = "openai"
    pipeline.model = "test-model"
    pipeline.cache_dir = tmp_path / "cache"
    pipeline.output_dir = tmp_path / "output"
    pipeline.temp_dir = tmp_path / "tmp"
    for directory in [pipeline.cache_dir, pipeline.output_dir, pipeline.temp_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Initialize components
    pipeline.converter = DocumentConverter(temp_dir=pipeline.temp_dir)
    pipeline.detector = LayoutDetector.__new__(LayoutDetector)
    pipeline.recognizer = TextRecognizer.__new__(TextRecognizer)

    fake_prompt_manager = FakePromptManager()
    fake_ai_client = FakeAIClient()

    # Set up recognizer with fake clients
    cast(Any, pipeline.recognizer).prompt_manager = fake_prompt_manager
    cast(Any, pipeline.recognizer).ai_client = fake_ai_client
    cast(Any, pipeline.recognizer).gemini_client = fake_ai_client
    cast(Any, pipeline.recognizer).openai_client = fake_ai_client
    cast(Any, pipeline.recognizer).cache = RecognitionCache(pipeline.cache_dir, use_cache=False)

    cast(Any, pipeline)._test_ai_client = fake_ai_client
    return pipeline


@pytest.fixture
def sample_regions():
    return [
        {
            "type": "plain text",
            "coords": [10, 20, 40, 18],
            "confidence": 0.95,
        },
        {
            "type": "table",
            "coords": [15, 80, 60, 25],
            "confidence": 0.92,
        },
        {
            "type": "title",
            "coords": [8, 130, 50, 20],
            "confidence": 0.9,
        },
    ]


def test_process_single_image_integration(pipeline_fixture, sample_regions):
    cast(Any, pipeline_fixture.detector).model = FakeDocLayoutModel([sample_regions])
    image_path = Path("samples/98A-004_origin_page-0001.jpg")

    result = pipeline_fixture.process_single_image(image_path)

    assert result["image_path"].endswith("98A-004_origin_page-0001.jpg")
    assert len(result["regions"]) == EXPECTED_REGION_COUNT

    plain_text_region = next(r for r in result["regions"] if r["type"] == "plain text")
    assert plain_text_region["text"] == "text-plain text"
    assert plain_text_region["corrected_text"] == "corrected-text-plain text"

    table_region = next(r for r in result["regions"] if r["type"] == "table")
    assert table_region["structured_text"] == "structured-table"
    assert "corrected_text" not in table_region

    assert "processed_at" in result


def test_process_pdf_integration(monkeypatch, pipeline_fixture, sample_regions, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")

    pages_regions = [sample_regions, sample_regions]
    cast(Any, pipeline_fixture.detector).model = FakeDocLayoutModel(pages_regions)

    def fake_pdfinfo(_path):
        return {"Pages": EXPECTED_PAGE_COUNT}

    def fake_render_pdf_page(self, _pdf_path, page_num):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        temp_image_path = self.temp_dir / f"{_pdf_path.stem}_page_{page_num}.jpg"
        cv2.imwrite(str(temp_image_path), image)
        return image, temp_image_path

    monkeypatch.setattr("pipeline.conversion.converter.pdfinfo_from_path", fake_pdfinfo)
    monkeypatch.setattr("pipeline.conversion.DocumentConverter.render_pdf_page", fake_render_pdf_page)

    summary = pipeline_fixture.process_pdf(pdf_path)

    assert summary["num_pages"] == EXPECTED_PAGE_COUNT
    assert summary["processed_pages"] == EXPECTED_PAGE_COUNT
    assert summary["status_summary"] == EXPECTED_STATUS_SUMMARY

    expected_output_dir = pipeline_fixture.output_dir / pipeline_fixture.model / pdf_path.stem
    assert summary["output_directory"] == str(expected_output_dir)
    assert all(page["status"] == "complete" for page in summary["pages"])
    fake_ai_client = cast(FakeAIClient, cast(Any, pipeline_fixture)._test_ai_client)
    assert len(fake_ai_client.correct_calls) == EXPECTED_PAGE_COUNT

    summary_file = expected_output_dir / "summary.json"
    assert summary_file.exists()
