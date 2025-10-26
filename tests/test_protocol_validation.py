"""Tests for Protocol implementation validation.

This module tests that all implementations properly follow their Protocol definitions,
ensuring type safety and interface compliance across the codebase.

Test coverage:
- Detector Protocol validation
- Sorter Protocol validation
- Recognizer Protocol validation
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.layout.detection import create_detector
from pipeline.layout.ordering import create_sorter
from pipeline.recognition import create_recognizer
from pipeline.types import BBox, Block, Detector, Recognizer, Sorter


class TestDetectorProtocol:
    """Test that all detectors properly implement the Detector Protocol."""

    @pytest.mark.parametrize(
        "detector_name",
        [
            "doclayout-yolo",
            "mineru-doclayout-yolo",
            # "mineru-vlm",  # Requires model download
            # "paddleocr-doclayout-v2",  # Requires PaddleOCR installation
            # "olmocr-vlm",  # Requires model download
        ],
    )
    def test_detector_implements_protocol(self, detector_name: str):
        """Test that detector implements Detector protocol correctly.

        Args:
            detector_name: Name of detector to test
        """
        # Create detector
        detector = create_detector(detector_name)

        # Verify it implements the Protocol
        assert isinstance(detector, Detector), f"{detector_name} should implement Detector protocol"

        # Verify it has the required method
        assert hasattr(detector, "detect"), f"{detector_name} should have detect() method"
        assert callable(detector.detect), f"{detector_name}.detect() should be callable"

    def test_detector_detect_signature(self):
        """Test that detect() method has correct signature."""
        detector = create_detector("doclayout-yolo")

        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call detect()
        blocks = detector.detect(test_image)

        # Verify return type
        assert isinstance(blocks, list), "detect() should return a list"
        assert all(isinstance(b, Block) for b in blocks), "detect() should return list of Block objects"

    def test_detector_block_structure(self):
        """Test that detected blocks have required structure."""
        detector = create_detector("doclayout-yolo")

        # Create test image (white square in center)
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[50:150, 50:150] = 255

        # Detect blocks
        blocks = detector.detect(test_image)

        # Check each block has required fields
        for block in blocks:
            assert hasattr(block, "type"), "Block should have 'type' field"
            assert hasattr(block, "bbox"), "Block should have 'bbox' field"
            assert isinstance(block.bbox, BBox), "Block.bbox should be BBox instance"
            assert block.type in [
                "text",
                "title",
                "table",
                "figure",
                "list",
                "caption",
                "footnote",
                "equation",
                "abstract",
            ], f"Block type should be valid: {block.type}"


class TestSorterProtocol:
    """Test that all sorters properly implement the Sorter Protocol."""

    @pytest.mark.parametrize(
        "sorter_name",
        [
            "pymupdf",
            "mineru-xycut",
            # "mineru-layoutreader",  # Requires model download
            # "mineru-vlm",  # Requires model download
            # "olmocr-vlm",  # Requires model download
            # "paddleocr-doclayout-v2",  # Requires PaddleOCR installation
        ],
    )
    def test_sorter_implements_protocol(self, sorter_name: str):
        """Test that sorter implements Sorter protocol correctly.

        Args:
            sorter_name: Name of sorter to test
        """
        # Create sorter
        sorter = create_sorter(sorter_name)

        # Verify it implements the Protocol
        assert isinstance(sorter, Sorter), f"{sorter_name} should implement Sorter protocol"

        # Verify it has the required method
        assert hasattr(sorter, "sort"), f"{sorter_name} should have sort() method"
        assert callable(sorter.sort), f"{sorter_name}.sort() should be callable"

    def test_sorter_sort_signature(self):
        """Test that sort() method has correct signature."""
        sorter = create_sorter("pymupdf")

        # Create test blocks
        test_blocks = [
            Block(type="text", bbox=BBox(10, 10, 90, 30)),
            Block(type="text", bbox=BBox(10, 40, 90, 60)),
        ]
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Call sort()
        sorted_blocks = sorter.sort(test_blocks, test_image)

        # Verify return type
        assert isinstance(sorted_blocks, list), "sort() should return a list"
        assert all(isinstance(b, Block) for b in sorted_blocks), "sort() should return list of Block objects"
        assert len(sorted_blocks) == len(test_blocks), "sort() should return same number of blocks"

    def test_sorter_adds_order_field(self):
        """Test that sorter adds order field to blocks."""
        sorter = create_sorter("pymupdf")

        # Create test blocks (bottom-up order)
        test_blocks = [
            Block(type="text", bbox=BBox(10, 40, 90, 60)),  # Lower block
            Block(type="text", bbox=BBox(10, 10, 90, 30)),  # Upper block
        ]
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Sort blocks
        sorted_blocks = sorter.sort(test_blocks, test_image)

        # Check order field is added
        for block in sorted_blocks:
            assert block.order is not None, "Sorted blocks should have 'order' field set"
            assert isinstance(block.order, int), "'order' should be an integer"

        # Verify blocks are sorted by order
        orders = [b.order for b in sorted_blocks if b.order is not None]
        assert orders == sorted(orders), "Blocks should be sorted by 'order' field"


class TestRecognizerProtocol:
    """Test that all recognizers properly implement the Recognizer Protocol."""

    @pytest.mark.parametrize(
        "recognizer_name",
        [
            "openai",
            "gemini",
            # "paddleocr-vl",  # Requires PaddleOCR installation
        ],
    )
    def test_recognizer_implements_protocol(self, recognizer_name: str):
        """Test that recognizer implements Recognizer protocol correctly.

        Args:
            recognizer_name: Recognizer name to test
        """
        # Create recognizer
        recognizer = create_recognizer(recognizer_name)

        # Verify it implements the Protocol
        assert isinstance(recognizer, Recognizer), f"{recognizer_name} should implement Recognizer protocol"

        # Verify it has required methods
        assert hasattr(recognizer, "process_blocks"), f"{recognizer_name} should have process_blocks() method"
        assert callable(recognizer.process_blocks), f"{recognizer_name}.process_blocks() should be callable"
        assert hasattr(recognizer, "correct_text"), f"{recognizer_name} should have correct_text() method"
        assert callable(recognizer.correct_text), f"{recognizer_name}.correct_text() should be callable"

    def test_recognizer_process_blocks_signature(self):
        """Test that process_blocks() method has correct signature."""
        recognizer = create_recognizer("openai")

        # Note: We can't actually call process_blocks without API keys
        # Just verify the signature exists and is callable
        assert callable(recognizer.process_blocks)

    def test_recognizer_correct_text_signature(self):
        """Test that correct_text() method has correct signature."""
        recognizer = create_recognizer("openai")

        # Note: We can't actually call correct_text without API keys
        # Just verify the signature exists and is callable
        assert callable(recognizer.correct_text)


class TestProtocolTypeChecking:
    """Test that type checking works correctly with Protocols."""

    def test_detector_typing(self):
        """Test that Detector type checking works."""
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            detector: Detector = create_detector("doclayout-yolo")
            # This should type-check correctly
            assert detector is not None

    def test_sorter_typing(self):
        """Test that Sorter type checking works."""
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            sorter: Sorter = create_sorter("pymupdf")
            # This should type-check correctly
            assert sorter is not None

    def test_recognizer_typing(self):
        """Test that Recognizer type checking works."""
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            recognizer: Recognizer = create_recognizer("openai")
            # This should type-check correctly
            assert recognizer is not None


class TestFactoryReturnTypes:
    """Test that factory functions return correct Protocol types."""

    def test_create_detector_returns_detector_protocol(self):
        """Test that create_detector() returns Detector protocol."""
        detector = create_detector("doclayout-yolo")

        # Should be instance of Detector protocol
        assert isinstance(detector, Detector)

    def test_create_sorter_returns_sorter_protocol(self):
        """Test that create_sorter() returns Sorter protocol."""
        sorter = create_sorter("pymupdf")

        # Should be instance of Sorter protocol
        assert isinstance(sorter, Sorter)

    def test_create_recognizer_returns_recognizer_protocol(self):
        """Test that create_recognizer() returns Recognizer protocol."""
        recognizer = create_recognizer("openai")

        # Should be instance of Recognizer protocol
        assert isinstance(recognizer, Recognizer)


class TestProtocolMethodSignatures:
    """Test that Protocol methods have correct signatures across implementations."""

    def test_all_detectors_have_consistent_detect_signature(self):
        """Test that all detectors have the same detect() signature."""
        detectors = [
            create_detector("doclayout-yolo"),
            create_detector("mineru-doclayout-yolo"),
        ]

        for detector in detectors:
            # Check method exists
            assert hasattr(detector, "detect")
            assert callable(detector.detect)

            # Check it accepts image parameter
            import inspect

            sig = inspect.signature(detector.detect)
            params = list(sig.parameters.keys())
            assert "image" in params, "detect() should have 'image' parameter"

    def test_all_sorters_have_consistent_sort_signature(self):
        """Test that all sorters have the same sort() signature."""
        sorters = [
            create_sorter("pymupdf"),
            create_sorter("mineru-xycut"),
        ]

        for sorter in sorters:
            # Check method exists
            assert hasattr(sorter, "sort")
            assert callable(sorter.sort)

            # Check it accepts blocks and image parameters
            import inspect

            sig = inspect.signature(sorter.sort)
            params = list(sig.parameters.keys())
            assert "blocks" in params, "sort() should have 'blocks' parameter"
            assert "image" in params, "sort() should have 'image' parameter"

    def test_all_recognizers_have_consistent_process_blocks_signature(self):
        """Test that all recognizers have the same process_blocks() signature."""
        recognizers = [
            create_recognizer("openai"),
            create_recognizer("gemini"),
        ]

        for recognizer in recognizers:
            # Check method exists
            assert hasattr(recognizer, "process_blocks")
            assert callable(recognizer.process_blocks)

            # Check it accepts image and blocks parameters
            import inspect

            sig = inspect.signature(recognizer.process_blocks)
            params = list(sig.parameters.keys())
            assert "image" in params, "process_blocks() should have 'image' parameter"
            assert "blocks" in params, "process_blocks() should have 'blocks' parameter"
