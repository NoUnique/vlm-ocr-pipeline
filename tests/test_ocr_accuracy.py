"""Test OCR accuracy using Levenshtein distance metric."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from pipeline import Pipeline


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
    regions = page_result.get("regions", [])
    
    extracted_texts: list[str] = []
    for region in regions:
        if "corrected_text" in region:
            extracted_texts.append(region["corrected_text"])
        elif "text" in region:
            extracted_texts.append(region["text"])
        elif "structured_text" in region:
            extracted_texts.append(region["structured_text"])
    
    return "\n".join(extracted_texts)


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to sample PDF for testing."""
    # Get project root directory (two levels up from tests/)
    project_root = Path(__file__).parent.parent
    return project_root / "samples" / "98A-004.pdf"


@pytest.fixture
def expected_json_path(tmp_path: Path) -> Path:
    """Path to expected JSON output (ground truth).
    
    This file should be created by running the baseline test first,
    then manually verifying and saving it as ground truth.
    """
    # Get fixtures directory relative to this test file
    fixtures_dir = Path(__file__).parent / "fixtures"
    return fixtures_dir / "98A-004_page1_expected.json"


@pytest.fixture
def expected_markdown_path(tmp_path: Path) -> Path:
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
    tmp_path: Path,
) -> None:
    """Test OCR by comparing full JSON output structure.
    
    This test:
    1. Processes the first page of the sample PDF
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
    
    # Initialize pipeline
    pipeline = Pipeline(
        use_cache=False,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "tmp",
        backend="openai",
        model="gemini-2.5-flash",
    )
    
    # Process first page
    result = pipeline.process_pdf(sample_pdf_path, max_pages=1)
    
    # Load expected JSON
    with expected_json_path.open("r", encoding="utf-8") as f:
        expected_json = json.load(f)
    
    # Normalize both JSONs for comparison
    result_normalized = normalize_json(result)
    expected_normalized = normalize_json(expected_json)
    
    # Save normalized results for debugging
    result_file = tmp_path / "result_normalized.json"
    expected_file = tmp_path / "expected_normalized.json"
    
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
        
        # Calculate text-based accuracy as fallback
        result_text = extract_text_from_result(result)
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
@pytest.mark.skip(reason="Markdown conversion logic not implemented yet")
def test_markdown_output_comparison(
    sample_pdf_path: Path,
    expected_markdown_path: Path,
    accuracy_threshold: float,
    tmp_path: Path,
) -> None:
    """Test OCR by comparing Markdown output.
    
    This test will be implemented after markdown conversion logic is added.
    
    Future implementation:
    1. Process the first page of the sample PDF
    2. Convert OCR result to Markdown format
    3. Compare with expected Markdown using Levenshtein distance
    4. Fail if accuracy is below threshold
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
    
    # TODO: Implement after markdown conversion logic is added
    # pipeline = Pipeline(...)
    # result = pipeline.process_pdf(sample_pdf_path, max_pages=1)
    # markdown_output = convert_to_markdown(result)  # To be implemented
    # expected_markdown = expected_markdown_path.read_text(encoding="utf-8")
    # metrics = calculate_accuracy(markdown_output, expected_markdown)
    # assert metrics["accuracy"] >= accuracy_threshold
    
    pytest.fail("Markdown conversion logic not implemented yet")


# ==================== Baseline Tests ====================


@pytest.mark.integration
@pytest.mark.slow
def test_json_baseline(sample_pdf_path: Path, tmp_path: Path) -> None:
    """Generate baseline JSON output for comparison testing.
    
    This test:
    1. Processes the first page of the sample PDF
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
    
    # Initialize pipeline
    pipeline = Pipeline(
        use_cache=False,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "tmp",
        backend="openai",
        model="gemini-2.5-flash",
    )
    
    # Process only the first page
    result = pipeline.process_pdf(sample_pdf_path, max_pages=1)
    
    # Verify basic processing
    assert result["processed_pages"] == 1
    assert len(result["pages"]) == 1
    
    # Save baseline JSON
    baseline_file = tmp_path / "baseline_output.json"
    baseline_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Also save normalized version
    normalized = normalize_json(result)
    normalized_file = tmp_path / "baseline_output_normalized.json"
    normalized_file.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))
    
    print(f"\n{'=' * 60}")
    print("Baseline JSON Output Generated:")
    print(f"{'=' * 60}")
    print(f"Full output: {baseline_file}")
    print(f"Normalized: {normalized_file}")
    print(f"\nPages processed: {result['processed_pages']}")
    print(f"Regions detected: {len(result['pages'][0]['regions'])}")
    print("\nNext steps:")
    print("1. Manually verify the JSON output")
    print("2. Create fixtures directory: mkdir -p tests/fixtures")
    print("3. Copy normalized JSON as ground truth:")
    print(f"   cp {normalized_file} tests/fixtures/98A-004_page1_expected.json")
    print("4. Run: uv run pytest tests/test_ocr_accuracy.py::test_json_output_comparison -v")
    print(f"{'=' * 60}")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skip(reason="Markdown conversion logic not implemented yet")
def test_markdown_baseline(sample_pdf_path: Path, tmp_path: Path) -> None:
    """Generate baseline Markdown output for comparison testing.
    
    This test will be implemented after markdown conversion logic is added.
    
    Future implementation:
    1. Process the first page
    2. Convert to Markdown
    3. Save as baseline for manual verification
    """
    # Skip if sample file doesn't exist
    if not sample_pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {sample_pdf_path}")
    
    # TODO: Implement after markdown conversion logic is added
    pytest.fail("Markdown conversion logic not implemented yet")


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
                "regions": [],
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
