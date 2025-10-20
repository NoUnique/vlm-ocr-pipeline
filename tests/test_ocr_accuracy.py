"""Test OCR accuracy using Levenshtein distance metric."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pipeline import Pipeline
from pipeline.conversion.output.markdown import document_dict_to_markdown


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance (minimum number of edits needed to transform s1 into s2)
        
    Examples:
        >>> levenshtein_distance("kitten", "sitting")
        3
        >>> levenshtein_distance("hello", "hello")
        0
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_accuracy(predicted: str, expected: str) -> dict[str, Any]:
    """Calculate accuracy metrics using Levenshtein distance.
    
    Args:
        predicted: Predicted text from OCR
        expected: Expected (ground truth) text
        
    Returns:
        Dictionary containing:
            - distance: Levenshtein distance
            - similarity: Similarity ratio (0.0 to 1.0)
            - accuracy: Accuracy percentage (0.0 to 100.0)
            - predicted_length: Length of predicted text
            - expected_length: Length of expected text
    """
    distance = levenshtein_distance(predicted, expected)
    max_length = max(len(predicted), len(expected))

    # Avoid division by zero
    if max_length == 0:
        similarity = 1.0
    else:
        similarity = 1.0 - (distance / max_length)

    accuracy = similarity * 100.0

    return {
        "distance": distance,
        "similarity": similarity,
        "accuracy": accuracy,
        "predicted_length": len(predicted),
        "expected_length": len(expected),
    }


def normalize_json(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize JSON data for comparison by removing volatile fields.
    
    Args:
        data: JSON data to normalize
        
    Returns:
        Normalized JSON data with volatile fields removed
    """
    normalized = copy.deepcopy(data)

    # Remove timestamp fields
    if "processed_at" in normalized:
        del normalized["processed_at"]

    # Remove volatile fields from pages
    if "pages" in normalized:
        for page in normalized["pages"]:
            if "processed_at" in page:
                del page["processed_at"]
            if "image_path" in page:
                # Keep only filename, not full path
                page["image_path"] = Path(page["image_path"]).name

    # Remove output directory path (keep only relative structure)
    if "output_directory" in normalized:
        normalized["output_directory"] = Path(normalized["output_directory"]).name

    return normalized


def extract_text_from_result(result: dict[str, Any]) -> str:
    """Extract all text from OCR result.
    
    Args:
        result: OCR result dictionary
        
    Returns:
        Concatenated text from all regions
    """
    if "pages" not in result or not result["pages"]:
        return ""

    page_result = result["pages"][0]
    regions = page_result.get("blocks", [])

    extracted_texts: list[str] = []
    for block in regions:
        if "corrected_text" in block:
            extracted_texts.append(block["corrected_text"])
        elif "text" in block:
            extracted_texts.append(block["text"])
        elif "structured_text" in block:
            extracted_texts.append(block["structured_text"])

    return "\n".join(extracted_texts)


@pytest.fixture
def test_output_dir() -> Path:
    """Create timestamp-based test output directory."""
    # Use tests/output/<timestamp> for persistent storage with unique runs
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("tests/output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to sample PDF for testing."""
    # Get project root directory (two levels up from tests/)
    project_root = Path(__file__).parent.parent
    return project_root / "samples" / "98A-004.pdf"


@pytest.fixture
def expected_json_path(test_output_dir: Path) -> Path:
    """Path to expected JSON output (ground truth).
    
    This file should be created by running the baseline test first,
    then manually verifying and saving it as ground truth.
    """
    # Get fixtures directory relative to this test file
    fixtures_dir = Path(__file__).parent / "fixtures"
    return fixtures_dir / "98A-004_page1_expected.json"


@pytest.fixture
def expected_markdown_path(test_output_dir: Path) -> Path:
    """Path to expected Markdown output (ground truth).
    
    This file should be created after implementing markdown conversion logic.
    """
    # Get fixtures directory relative to this test file
    fixtures_dir = Path(__file__).parent / "fixtures"
    return fixtures_dir / "98A-004_page1_expected.md"


@pytest.fixture
def expected_text() -> str:
    """Expected text from the first page of sample PDF.
    
    This should be manually verified and updated with the actual expected content.
    For now, this is a placeholder that should be replaced with ground truth.
    """
    return """This is the expected text from the first page.
Replace this with the actual ground truth text after manual verification."""


@pytest.fixture
def accuracy_threshold() -> float:
    """Minimum accuracy threshold for the test to pass."""
    return 90.0  # 90% accuracy


# ==================== Test 1: JSON Output Comparison ====================


@pytest.mark.integration
@pytest.mark.slow
def test_json_output_comparison(
    sample_pdf_path: Path,
    expected_json_path: Path,
    test_output_dir: Path,
) -> None:
    """Test OCR by comparing full JSON output structure.

    This test:
    1. Processes the first page of the sample PDF using MinerU 2.5 VLM + Gemini 2.5 Flash
    2. Compares the JSON output with expected JSON
    3. Reports differences in structure and content
    """
    # Skip if sample file doesn't exist
    if not sample_pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {sample_pdf_path}")

    # Skip if expected JSON doesn't exist
    if not expected_json_path.exists():
        pytest.skip(
            f"Expected JSON not found: {expected_json_path}\n"
            f"Run test_json_baseline first to generate ground truth."
        )

    # Initialize pipeline with MinerU 2.5 VLM + Gemini 2.5 Flash
    pipeline = Pipeline(
        use_cache=False,
        cache_dir=test_output_dir / "cache",
        output_dir=test_output_dir,
        temp_dir=test_output_dir / "tmp",
        backend="gemini",  # ✅ Use Gemini backend for text correction
        model="gemini-2.5-flash",
        gemini_tier="free",  # Adjust based on your tier
        detector="mineru-vlm",  # ✅ MinerU 2.5 VLM detector (1.2B model)
        sorter="mineru-vlm",    # ✅ MinerU VLM sorter
        # mineru_model defaults to "opendatalab/MinerU2.5-2509-1.2B"
        mineru_backend="transformers",  # or "vllm-engine" for 20-30x speedup
    )

    # Process first page
    result = pipeline.process_pdf(sample_pdf_path, max_pages=1)

    # Load expected JSON
    with expected_json_path.open("r", encoding="utf-8") as f:
        expected_json = json.load(f)

    # Normalize both JSONs for comparison - convert Document to dict
    result_normalized = normalize_json(result.to_dict())
    expected_normalized = normalize_json(expected_json)

    # Save normalized results for debugging
    result_file = test_output_dir / "result_normalized.json"
    expected_file = test_output_dir / "expected_normalized.json"

    result_file.write_text(json.dumps(result_normalized, indent=2, ensure_ascii=False))
    expected_file.write_text(json.dumps(expected_normalized, indent=2, ensure_ascii=False))

    # Compare JSONs
    print(f"\n{'=' * 60}")
    print("JSON Comparison Results:")
    print(f"{'=' * 60}")
    print(f"Result saved to: {result_file}")
    print(f"Expected saved to: {expected_file}")

    # Deep comparison
    differences = _compare_json_deep(expected_normalized, result_normalized)

    if differences:
        print(f"\nFound {len(differences)} differences:")
        for i, diff in enumerate(differences[:10], 1):  # Show first 10 differences
            print(f"  {i}. {diff}")
        if len(differences) > 10:
            print(f"  ... and {len(differences) - 10} more differences")
        print(f"{'=' * 60}")

        # Calculate text-based accuracy as fallback - convert Document to dict
        result_text = extract_text_from_result(result.to_dict())
        expected_text = extract_text_from_result(expected_json)

        result_normalized_text = " ".join(result_text.split())
        expected_normalized_text = " ".join(expected_text.split())

        metrics = calculate_accuracy(result_normalized_text, expected_normalized_text)
        print(f"\nText Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Levenshtein Distance: {metrics['distance']}")
        print(f"{'=' * 60}")
    else:
        print("✅ JSON outputs match perfectly!")
        print(f"{'=' * 60}")

    # Assert based on comparison
    assert not differences or len(differences) < 5, (
        f"JSON outputs differ significantly: {len(differences)} differences found. "
        f"See {result_file} and {expected_file} for details."
    )


def _compare_json_deep(expected: Any, actual: Any, path: str = "root") -> list[str]:
    """Deep comparison of two JSON structures.
    
    Args:
        expected: Expected JSON data
        actual: Actual JSON data
        path: Current path in JSON structure
        
    Returns:
        List of difference descriptions
    """
    differences: list[str] = []

    if type(expected) is not type(actual):
        differences.append(f"{path}: type mismatch ({type(expected).__name__} vs {type(actual).__name__})")
        return differences

    if isinstance(expected, dict):
        # Check for missing or extra keys
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())

        missing_keys = expected_keys - actual_keys
        extra_keys = actual_keys - expected_keys

        if missing_keys:
            differences.append(f"{path}: missing keys {missing_keys}")
        if extra_keys:
            differences.append(f"{path}: extra keys {extra_keys}")

        # Compare common keys
        for key in expected_keys & actual_keys:
            differences.extend(_compare_json_deep(expected[key], actual[key], f"{path}.{key}"))

    elif isinstance(expected, list):
        if len(expected) != len(actual):
            differences.append(f"{path}: length mismatch ({len(expected)} vs {len(actual)})")
            return differences

        for i, (exp_item, act_item) in enumerate(zip(expected, actual, strict=False)):
            differences.extend(_compare_json_deep(exp_item, act_item, f"{path}[{i}]"))

    # Compare primitive values with tolerance for floats
    elif isinstance(expected, float) and isinstance(actual, float):
        if abs(expected - actual) > 0.001:  # Tolerance for float comparison
            differences.append(f"{path}: value mismatch ({expected} vs {actual})")
    elif expected != actual:
        differences.append(f"{path}: value mismatch ({expected} vs {actual})")

    return differences


# ==================== Test 2: Markdown Output Comparison ====================


@pytest.mark.integration
@pytest.mark.slow
def test_markdown_output_comparison(
    sample_pdf_path: Path,
    expected_markdown_path: Path,
    accuracy_threshold: float,
    test_output_dir: Path,
) -> None:
    """Test OCR by comparing Markdown output.

    This test:
    1. Processes the first page of the sample PDF using MinerU 2.5 VLM + Gemini 2.5 Flash
    2. Converts OCR result to Markdown format using region type-based conversion
    3. Compares with expected Markdown using Levenshtein distance
    4. Fails if accuracy is below threshold (90% by default)
    """
    # Skip if sample file doesn't exist
    if not sample_pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {sample_pdf_path}")

    # Skip if expected markdown doesn't exist
    if not expected_markdown_path.exists():
        pytest.skip(
            f"Expected Markdown not found: {expected_markdown_path}\n"
            f"Run test_markdown_baseline first to generate ground truth."
        )

    # Initialize pipeline with MinerU 2.5 VLM + Gemini 2.5 Flash
    pipeline = Pipeline(
        use_cache=False,
        cache_dir=test_output_dir / "cache",
        output_dir=test_output_dir / "output",
        temp_dir=test_output_dir / "tmp",
        backend="gemini",  # ✅ Use Gemini backend
        model="gemini-2.5-flash",
        gemini_tier="free",  # Adjust based on your tier
        detector="mineru-vlm",  # ✅ MinerU 2.5 VLM detector
        sorter="mineru-vlm",    # ✅ MinerU VLM sorter
        mineru_model="opendatalab/PDF-Extract-Kit-1.0",  # MinerU model
        mineru_backend="transformers",  # or "vllm-engine" if available
    )

    # Process first page
    result = pipeline.process_pdf(sample_pdf_path, max_pages=1)

    # Convert to Markdown - convert Document to dict
    markdown_output = document_dict_to_markdown(result.to_dict())

    # Load expected Markdown
    expected_markdown = expected_markdown_path.read_text(encoding="utf-8")

    # Normalize whitespace for comparison (collapse multiple spaces/newlines)
    markdown_normalized = " ".join(markdown_output.split())
    expected_normalized = " ".join(expected_markdown.split())

    # Calculate accuracy
    metrics = calculate_accuracy(markdown_normalized, expected_normalized)

    # Save outputs for debugging
    output_file = test_output_dir / "markdown_output.md"
    output_file.write_text(markdown_output, encoding="utf-8")

    expected_file = test_output_dir / "markdown_expected.md"
    expected_file.write_text(expected_markdown, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("Markdown Accuracy Results:")
    print(f"{'=' * 60}")
    print(f"Output saved to: {output_file}")
    print(f"Expected saved to: {expected_file}")
    print(f"\nAccuracy: {metrics['accuracy']:.2f}%")
    print(f"Levenshtein Distance: {metrics['distance']}")
    print(f"Similarity: {metrics['similarity']:.2%}")
    print(f"Threshold: {accuracy_threshold}%")
    print("\nLengths:")
    print(f"  Output: {metrics['predicted_length']} characters")
    print(f"  Expected: {metrics['expected_length']} characters")
    print(f"{'=' * 60}")

    # Assert based on accuracy threshold
    assert metrics["accuracy"] >= accuracy_threshold, (
        f"Markdown accuracy ({metrics['accuracy']:.2f}%) is below threshold ({accuracy_threshold}%). "
        f"Levenshtein distance: {metrics['distance']}. "
        f"See {output_file} and {expected_file} for details."
    )


# ==================== Baseline Tests ====================


@pytest.mark.integration
@pytest.mark.slow
def test_json_baseline(sample_pdf_path: Path, test_output_dir: Path) -> None:
    """Generate baseline JSON output for comparison testing.

    This test:
    1. Processes the first page of the sample PDF using MinerU 2.5 VLM + Gemini 2.5 Flash
    2. Saves the JSON output as a baseline
    3. User should manually verify and move to fixtures directory

    Run this first to generate ground truth JSON:
        uv run pytest tests/test_ocr_accuracy.py::test_json_baseline -v -s

    Then manually verify the output and save as:
        tests/fixtures/98A-004_page1_expected.json
    """
    # Skip if sample file doesn't exist
    if not sample_pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {sample_pdf_path}")

    # Use tests/output instead of tmp_path for persistent storage
    # test_output_dir passed as fixture
    # test_output_dir already created by fixture

    # Initialize pipeline with MinerU 2.5 VLM + Gemini 2.5 Flash
    pipeline = Pipeline(
        use_cache=False,
        cache_dir=test_output_dir / "cache",
        output_dir=test_output_dir,
        temp_dir=test_output_dir / "tmp",
        backend="gemini",  # ✅ Use Gemini backend for text correction
        model="gemini-2.5-flash",
        gemini_tier="free",  # Adjust based on your tier
        detector="mineru-vlm",  # ✅ MinerU 2.5 VLM detector (1.2B model)
        sorter="mineru-vlm",    # ✅ MinerU VLM sorter
        # mineru_model defaults to "opendatalab/MinerU2.5-2509-1.2B"
        mineru_backend="transformers",  # or "vllm-engine" for 20-30x speedup
    )

    # Process only the first page
    result = pipeline.process_pdf(sample_pdf_path, max_pages=1)

    # Verify basic processing - result is now a Document object
    assert result.processed_pages == 1
    assert len(result.pages) == 1

    # Save baseline JSON - convert Document to dict
    baseline_file = test_output_dir / "baseline_output.json"
    baseline_file.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    # Also save normalized version
    normalized = normalize_json(result.to_dict())
    normalized_file = test_output_dir / "baseline_output_normalized.json"
    normalized_file.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))

    # Read actual page_1.json to get region count
    output_dir = test_output_dir / "gemini-2.5-flash" / "98A-004"
    page_json_file = output_dir / "page_1.json"

    if page_json_file.exists():
        page_data = json.loads(page_json_file.read_text())
        num_regions = len(page_data.get("blocks", []))
    else:
        num_regions = 0

    print(f"\n{'=' * 60}")
    print("Baseline JSON Output Generated:")
    print(f"{'=' * 60}")
    print("Configuration:")
    print("  - Backend: gemini")
    print("  - Model: gemini-2.5-flash")
    print("  - Detector: mineru-vlm")
    print("  - Sorter: mineru-vlm")
    print("\nOutput files:")
    print(f"  Summary: {baseline_file}")
    print(f"  Page data: {page_json_file}")
    print(f"  Normalized: {normalized_file}")
    print("\nResults:")
    print(f"  Pages processed: {result.processed_pages}")
    print(f"  Blocks detected: {num_regions}")
    print("\nNext steps:")
    print("1. Manually verify the JSON output")
    print("2. Create fixtures directory: mkdir -p tests/fixtures")
    print("3. Copy page JSON as ground truth:")
    print(f"   cp {page_json_file} tests/fixtures/98A-004_page1_expected.json")
    print("4. Run: uv run pytest tests/test_ocr_accuracy.py::test_json_output_comparison -v")
    print(f"{'=' * 60}")


@pytest.mark.integration
@pytest.mark.slow
def test_markdown_baseline(sample_pdf_path: Path, test_output_dir: Path) -> None:
    """Generate baseline Markdown output for comparison testing.

    This test:
    1. Processes the first page of the sample PDF using MinerU 2.5 VLM + Gemini 2.5 Flash
    2. Converts the result to Markdown using region type-based conversion
    3. Saves as baseline for manual verification

    Run this to generate ground truth Markdown:
        uv run pytest tests/test_ocr_accuracy.py::test_markdown_baseline -v -s

    Then manually verify the output and save as:
        tests/fixtures/98A-004_page1_expected.md
    """
    # Skip if sample file doesn't exist
    if not sample_pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {sample_pdf_path}")

    # Use tests/output instead of tmp_path for persistent storage
    # test_output_dir passed as fixture
    # test_output_dir already created by fixture

    # Initialize pipeline with MinerU 2.5 VLM + Gemini 2.5 Flash
    pipeline = Pipeline(
        use_cache=False,
        cache_dir=test_output_dir / "cache",
        output_dir=test_output_dir,
        temp_dir=test_output_dir / "tmp",
        backend="gemini",  # ✅ Use Gemini backend
        model="gemini-2.5-flash",
        gemini_tier="free",  # Adjust based on your tier
        detector="mineru-vlm",  # ✅ MinerU 2.5 VLM detector
        sorter="mineru-vlm",    # ✅ MinerU VLM sorter
        mineru_model="opendatalab/PDF-Extract-Kit-1.0",  # MinerU model
        mineru_backend="transformers",  # or "vllm-engine" if available
    )

    # Process only the first page
    result = pipeline.process_pdf(sample_pdf_path, max_pages=1)

    # Verify basic processing - result is now a Document object
    assert result.processed_pages == 1
    assert len(result.pages) == 1

    # Convert to Markdown using the new dict wrapper - convert Document to dict
    markdown_output = document_dict_to_markdown(result.to_dict())

    # Save baseline Markdown
    baseline_file = test_output_dir / "baseline_output.md"
    baseline_file.write_text(markdown_output, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("Baseline Markdown Output Generated:")
    print(f"{'=' * 60}")
    print("Configuration:")
    print("  - Backend: gemini")
    print("  - Model: gemini-2.5-flash")
    print("  - Detector: mineru-vlm")
    print("  - Sorter: mineru-vlm")
    print("  - Conversion: Region type-based")
    print("\nOutput file:")
    print(f"  {baseline_file}")
    print("\nResults:")
    print(f"  Pages processed: {result.processed_pages}")
    print(f"  Blocks detected: {len(result.pages[0].blocks)}")
    print(f"  Markdown length: {len(markdown_output)} characters")
    print("\nNext steps:")
    print("1. Manually verify the Markdown output")
    print("2. Copy as ground truth:")
    print(f"   cp {baseline_file} tests/fixtures/98A-004_page1_expected.md")
    print("3. Run: uv run pytest tests/test_ocr_accuracy.py::test_markdown_output_comparison -v")
    print(f"{'=' * 60}")


# ==================== Unit Tests ====================


def test_levenshtein_distance() -> None:
    """Test the Levenshtein distance implementation."""
    # Test identical strings
    assert levenshtein_distance("hello", "hello") == 0

    # Test completely different strings
    assert levenshtein_distance("abc", "xyz") == 3

    # Test classic example
    assert levenshtein_distance("kitten", "sitting") == 3

    # Test empty strings
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("hello", "") == 5
    assert levenshtein_distance("", "world") == 5

    # Test single character difference
    assert levenshtein_distance("test", "text") == 1


def test_calculate_accuracy() -> None:
    """Test accuracy calculation."""
    # Perfect match
    metrics = calculate_accuracy("hello world", "hello world")
    assert metrics["distance"] == 0
    assert metrics["similarity"] == 1.0
    assert metrics["accuracy"] == 100.0

    # Partial match
    metrics = calculate_accuracy("hello world", "hello word")
    assert metrics["distance"] == 1
    assert 90.0 < metrics["accuracy"] < 95.0

    # Empty strings
    metrics = calculate_accuracy("", "")
    assert metrics["distance"] == 0
    assert metrics["similarity"] == 1.0
    assert metrics["accuracy"] == 100.0


def test_normalize_json() -> None:
    """Test JSON normalization."""
    input_json = {
        "processed_at": "2024-01-01T00:00:00",
        "num_pages": 1,
        "output_directory": "/full/path/to/output/dir",
        "pages": [
            {
                "processed_at": "2024-01-01T00:00:00",
                "image_path": "/full/path/to/image.jpg",
                "blocks": [],
            }
        ],
    }

    normalized = normalize_json(input_json)

    # Check that volatile fields are removed
    assert "processed_at" not in normalized
    assert "processed_at" not in normalized["pages"][0]

    # Check that paths are simplified
    assert normalized["output_directory"] == "dir"
    assert normalized["pages"][0]["image_path"] == "image.jpg"

    # Check that other fields are preserved
    assert normalized["num_pages"] == 1
    assert len(normalized["pages"]) == 1
