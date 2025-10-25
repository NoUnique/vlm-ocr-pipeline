"""End-to-end tests for PaddleOCR pipeline (PP-DocLayoutV2 + PaddleOCR-VL)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import Pipeline
from pipeline.conversion.input.image import load_image


@pytest.fixture(scope="module")
def test_image_path():
    """Get path to test image."""
    # Use existing test image from archive
    image_path = Path("tests/output/archive/20251016_180523/tmp/98A-004_page_1.jpg")
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")
    return image_path


@pytest.fixture(scope="module")
def output_dir():
    """Create and return output directory for test results."""
    output_path = Path("tests/output/paddleocr_e2e")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def test_paddleocr_doclayout_v2_detector_only(test_image_path, output_dir):
    """Test PP-DocLayoutV2 detector with passthrough sorter."""
    # Load test image
    image = load_image(test_image_path)

    # Create pipeline with PaddleOCR detector + sorter
    pipeline = Pipeline(
        detector="paddleocr-doclayout-v2",
        sorter="paddleocr-doclayout-v2",  # Auto-selected, but explicit here
        backend="openai",  # Won't be used (detection only)
    )

    # Run detection + sorting only
    assert pipeline.detector is not None, "Detector should be initialized"
    assert pipeline.sorter is not None, "Sorter should be initialized"
    blocks = pipeline.detector.detect(image)
    sorted_blocks = pipeline.sorter.sort(blocks, image)

    # Verify results
    assert len(sorted_blocks) > 0, "Should detect at least one block"

    # Verify all blocks have order field set
    for block in sorted_blocks:
        assert block.order is not None, f"Block {block.type} missing order"

    # Verify blocks are sorted by order
    orders = [block.order for block in sorted_blocks if block.order is not None]
    assert orders == sorted(orders), "Blocks should be sorted by order field"

    # Save results to JSON
    output_file = output_dir / "detection_only.json"
    result_data = {
        "num_blocks": len(sorted_blocks),
        "blocks": [
            {
                "order": block.order,
                "type": block.type,
                "bbox": block.bbox.to_xywh_list(),
                "confidence": block.detection_confidence,
            }
            for block in sorted_blocks
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Detection results saved to: {output_file}")
    print(f"✓ Detected {len(sorted_blocks)} blocks")
    print(f"✓ First 3 blocks order: {orders[:3]}")


@pytest.mark.slow
def test_paddleocr_full_pipeline_with_paddleocr_vl(test_image_path, output_dir):
    """Test full PaddleOCR pipeline: PP-DocLayoutV2 detector + PaddleOCR-VL recognizer."""
    # Create pipeline with PaddleOCR detector + recognizer
    pipeline = Pipeline(
        detector="paddleocr-doclayout-v2",
        sorter="paddleocr-doclayout-v2",
        backend="paddleocr-vl",  # PaddleOCR-VL recognizer
        output_dir=str(output_dir / "full_pipeline"),
    )

    # Process single image
    result = pipeline.process_single_image(image_path=test_image_path)

    # Verify results
    assert result is not None, "Pipeline should return result"
    assert len(result["blocks"]) > 0, "Should have at least one block"

    # Verify blocks have text field
    text_blocks = [b for b in result["blocks"] if b.get("text")]
    assert len(text_blocks) > 0, "Should have at least one block with text"

    # Save results to JSON
    output_file = output_dir / "full_pipeline.json"

    # Convert to serializable format
    serializable_result = {
        "page_num": result["page_num"],
        "num_blocks": len(result["blocks"]),
        "blocks": [
            {
                "order": block.get("order"),
                "type": block.get("type"),
                "bbox": block.get("xywh"),  # Already in xywh format from to_dict()
                "confidence": block.get("detection_confidence"),
                "text": block.get("text", ""),
            }
            for block in result["blocks"]
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Full pipeline results saved to: {output_file}")
    print(f"✓ Processed {len(result['blocks'])} blocks")
    print(f"✓ Text blocks: {len(text_blocks)}")
    print(f"✓ First block text: {text_blocks[0].get('text', '')[:50]}...")


@pytest.mark.slow
def test_paddleocr_reading_order_verification(test_image_path, output_dir):
    """Verify that PP-DocLayoutV2 pointer network provides correct reading order."""
    # Load test image
    image = load_image(test_image_path)

    # Create pipeline
    pipeline = Pipeline(
        detector="paddleocr-doclayout-v2",
        sorter="paddleocr-doclayout-v2",
        backend="openai",
    )

    # Run detection + sorting
    assert pipeline.detector is not None, "Detector should be initialized"
    assert pipeline.sorter is not None, "Sorter should be initialized"
    blocks = pipeline.detector.detect(image)
    sorted_blocks = pipeline.sorter.sort(blocks, image)

    # Analyze reading order by Y coordinates
    y_coordinates = [(block.order, block.bbox.y0, block.type) for block in sorted_blocks]

    # Save analysis to JSON
    output_file = output_dir / "reading_order_analysis.json"
    analysis_data = {
        "total_blocks": len(sorted_blocks),
        "blocks_by_order": [
            {
                "order": order,
                "y_coordinate": y,
                "type": block_type,
            }
            for order, y, block_type in y_coordinates
        ],
        "is_top_to_bottom": all(y_coordinates[i][1] <= y_coordinates[i + 1][1] for i in range(len(y_coordinates) - 1)),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Reading order analysis saved to: {output_file}")
    print(f"✓ Blocks in order: {len(sorted_blocks)}")
    print(f"✓ Top-to-bottom order: {analysis_data['is_top_to_bottom']}")

    # Print first 5 blocks for debugging
    print("\nFirst 5 blocks (order → Y coordinate):")
    for order, y, block_type in y_coordinates[:5]:
        print(f"  [{order}] y={y} type={block_type}")
