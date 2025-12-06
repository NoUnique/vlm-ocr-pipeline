"""Tests for Protocol implementation validation.

This module tests that all implementations properly follow their Protocol definitions,
ensuring type safety and interface compliance across the codebase.

Test coverage:
- Detector Protocol validation
- Sorter Protocol validation
- Recognizer Protocol validation

Note: Uses session-scoped fixtures from conftest.py to avoid TORCH_LIBRARY conflicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.types import BBox, Block, Detector, Recognizer, Sorter

# ==================== Session-scoped Fixtures from conftest.py ====================
# doclayout_yolo_detector, mineru_doclayout_yolo_detector are from conftest.py


@pytest.fixture(scope="session")
def pymupdf_sorter():
    """Create PyMuPDF sorter once for session."""
    from pipeline.layout.ordering import create_sorter

    return create_sorter("pymupdf")


@pytest.fixture(scope="session")
def xycut_sorter():
    """Create XY-Cut sorter once for session."""
    from pipeline.layout.ordering import create_sorter

    return create_sorter("mineru-xycut")


# Note: openai_recognizer and gemini_recognizer fixtures are in conftest.py


# ==================== Detector Protocol Tests ====================


@pytest.mark.slow
class TestDetectorProtocol:
    """Test that all detectors properly implement the Detector Protocol."""

    def test_doclayout_yolo_implements_protocol(self, doclayout_yolo_detector):
        """Test that doclayout-yolo implements Detector protocol correctly."""
        assert isinstance(
            doclayout_yolo_detector, Detector
        ), "doclayout-yolo should implement Detector protocol"
        assert hasattr(doclayout_yolo_detector, "detect")
        assert callable(doclayout_yolo_detector.detect)

    def test_mineru_doclayout_yolo_implements_protocol(self, mineru_doclayout_yolo_detector):
        """Test that mineru-doclayout-yolo implements Detector protocol correctly."""
        assert isinstance(
            mineru_doclayout_yolo_detector, Detector
        ), "mineru-doclayout-yolo should implement Detector protocol"
        assert hasattr(mineru_doclayout_yolo_detector, "detect")
        assert callable(mineru_doclayout_yolo_detector.detect)

    def test_detector_detect_signature(self, doclayout_yolo_detector):
        """Test that detect() method has correct signature."""
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call detect()
        blocks = doclayout_yolo_detector.detect(test_image)

        # Verify return type
        assert isinstance(blocks, list), "detect() should return a list"
        assert all(isinstance(b, Block) for b in blocks), "detect() should return list of Block objects"

    def test_detector_block_structure(self, doclayout_yolo_detector):
        """Test that detected blocks have required structure."""
        # Create test image (white square in center)
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[50:150, 50:150] = 255

        # Detect blocks
        blocks = doclayout_yolo_detector.detect(test_image)

        # Check each block has required fields
        for block in blocks:
            assert hasattr(block, "type"), "Block should have 'type' field"
            assert hasattr(block, "bbox"), "Block should have 'bbox' field"
            assert isinstance(block.bbox, BBox), "Block.bbox should be BBox instance"


# ==================== Sorter Protocol Tests ====================


class TestSorterProtocol:
    """Test that all sorters properly implement the Sorter Protocol."""

    def test_pymupdf_implements_protocol(self, pymupdf_sorter):
        """Test that pymupdf implements Sorter protocol correctly."""
        assert isinstance(pymupdf_sorter, Sorter), "pymupdf should implement Sorter protocol"
        assert hasattr(pymupdf_sorter, "sort")
        assert callable(pymupdf_sorter.sort)

    def test_xycut_implements_protocol(self, xycut_sorter):
        """Test that mineru-xycut implements Sorter protocol correctly."""
        assert isinstance(xycut_sorter, Sorter), "mineru-xycut should implement Sorter protocol"
        assert hasattr(xycut_sorter, "sort")
        assert callable(xycut_sorter.sort)

    def test_sorter_sort_signature(self, pymupdf_sorter):
        """Test that sort() method has correct signature."""
        # Create test blocks
        test_blocks = [
            Block(type="text", bbox=BBox(10, 10, 90, 30)),
            Block(type="text", bbox=BBox(10, 40, 90, 60)),
        ]
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call sort()
        sorted_blocks = pymupdf_sorter.sort(test_blocks, test_image)

        # Verify return type
        assert isinstance(sorted_blocks, list), "sort() should return a list"
        assert all(isinstance(b, Block) for b in sorted_blocks), "sort() should return list of Block objects"
        assert len(sorted_blocks) == len(test_blocks), "sort() should return same number of blocks"

    def test_sorter_adds_order_field(self, pymupdf_sorter):
        """Test that sorter adds order field to blocks."""
        # Create test blocks (bottom-up order)
        test_blocks = [
            Block(type="text", bbox=BBox(10, 40, 90, 60)),  # Lower block
            Block(type="text", bbox=BBox(10, 10, 90, 30)),  # Upper block
        ]
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Sort blocks
        sorted_blocks = pymupdf_sorter.sort(test_blocks, test_image)

        # Check order field is added
        for block in sorted_blocks:
            assert block.order is not None, "Sorted blocks should have 'order' field set"
            assert isinstance(block.order, int), "'order' should be an integer"

        # Verify blocks are sorted by order
        orders = [b.order for b in sorted_blocks if b.order is not None]
        assert orders == sorted(orders), "Blocks should be sorted by 'order' field"


# ==================== Recognizer Protocol Tests ====================


class TestRecognizerProtocol:
    """Test that all recognizers properly implement the Recognizer Protocol."""

    def test_openai_implements_protocol(self, openai_recognizer):
        """Test that openai implements Recognizer protocol correctly."""
        assert isinstance(openai_recognizer, Recognizer), "openai should implement Recognizer protocol"
        assert hasattr(openai_recognizer, "process_blocks")
        assert callable(openai_recognizer.process_blocks)
        assert hasattr(openai_recognizer, "correct_text")
        assert callable(openai_recognizer.correct_text)

    def test_gemini_implements_protocol(self, gemini_recognizer):
        """Test that gemini implements Recognizer protocol correctly."""
        assert isinstance(gemini_recognizer, Recognizer), "gemini should implement Recognizer protocol"
        assert hasattr(gemini_recognizer, "process_blocks")
        assert callable(gemini_recognizer.process_blocks)
        assert hasattr(gemini_recognizer, "correct_text")
        assert callable(gemini_recognizer.correct_text)


# ==================== Factory Return Type Tests ====================


@pytest.mark.slow
class TestFactoryReturnTypes:
    """Test that factory functions return correct Protocol types."""

    def test_create_detector_returns_detector_protocol(self, doclayout_yolo_detector):
        """Test that create_detector() returns Detector protocol."""
        assert isinstance(doclayout_yolo_detector, Detector)

    def test_create_sorter_returns_sorter_protocol(self, pymupdf_sorter):
        """Test that create_sorter() returns Sorter protocol."""
        assert isinstance(pymupdf_sorter, Sorter)

    def test_create_recognizer_returns_recognizer_protocol(self, openai_recognizer):
        """Test that create_recognizer() returns Recognizer protocol."""
        assert isinstance(openai_recognizer, Recognizer)


# ==================== Protocol Method Signature Tests ====================


@pytest.mark.slow
class TestProtocolMethodSignatures:
    """Test that Protocol methods have correct signatures across implementations."""

    def test_all_detectors_have_consistent_detect_signature(
        self, doclayout_yolo_detector, mineru_doclayout_yolo_detector
    ):
        """Test that all detectors have the same detect() signature."""
        import inspect

        detectors = [doclayout_yolo_detector, mineru_doclayout_yolo_detector]

        for detector in detectors:
            # Check method exists
            assert hasattr(detector, "detect")
            assert callable(detector.detect)

            # Check it accepts image parameter
            sig = inspect.signature(detector.detect)
            params = list(sig.parameters.keys())
            assert "image" in params, "detect() should have 'image' parameter"

    def test_all_sorters_have_consistent_sort_signature(self, pymupdf_sorter, xycut_sorter):
        """Test that all sorters have the same sort() signature."""
        import inspect

        sorters = [pymupdf_sorter, xycut_sorter]

        for sorter in sorters:
            # Check method exists
            assert hasattr(sorter, "sort")
            assert callable(sorter.sort)

            # Check it accepts blocks and image parameters
            sig = inspect.signature(sorter.sort)
            params = list(sig.parameters.keys())
            assert "blocks" in params, "sort() should have 'blocks' parameter"
            assert "image" in params, "sort() should have 'image' parameter"

    def test_all_recognizers_have_consistent_process_blocks_signature(
        self, openai_recognizer, gemini_recognizer
    ):
        """Test that all recognizers have the same process_blocks() signature."""
        import inspect

        recognizers = [openai_recognizer, gemini_recognizer]

        for recognizer in recognizers:
            # Check method exists
            assert hasattr(recognizer, "process_blocks")
            assert callable(recognizer.process_blocks)

            # Check it accepts image and blocks parameters
            sig = inspect.signature(recognizer.process_blocks)
            params = list(sig.parameters.keys())
            assert "image" in params, "process_blocks() should have 'image' parameter"
            assert "blocks" in params, "process_blocks() should have 'blocks' parameter"
