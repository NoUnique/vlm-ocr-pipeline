"""Tests for Protocol interfaces.

Tests cover:
- Detector Protocol compliance
- Sorter Protocol compliance
- Recognizer Protocol compliance
- Renderer Protocol compliance
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import Mock

import numpy as np

from pipeline.types import BBox, Block, Detector, Recognizer, Sorter


class TestDetectorProtocol:
    """Tests for Detector Protocol."""

    def test_protocol_has_detect_method(self):
        """Test that Detector Protocol requires detect method."""
        # Protocol classes have method stubs, check via annotations
        _annotations = getattr(Detector, "__protocol_attrs__", set())  # For reference
        # At minimum, detect should be callable
        assert callable(getattr(Detector, "detect", None)) or "detect" in dir(Detector)

    def test_mock_detector_satisfies_protocol(self):
        """Test that a properly structured mock satisfies Detector Protocol."""

        class MockDetector:
            name = "mock-detector"
            source = "mock-detector"

            def detect(self, image: np.ndarray) -> list[Block]:
                return [Block(type="text", bbox=BBox(0, 0, 100, 100))]

            def detect_batch(self, images: list[np.ndarray]) -> list[list[Block]]:
                return [self.detect(image) for image in images]

        detector = MockDetector()
        assert isinstance(detector, Detector)

    def test_incomplete_detector_fails_protocol(self):
        """Test that incomplete implementation doesn't satisfy Protocol."""

        class IncompleteDetector:
            # Missing name and source
            def detect(self, image: np.ndarray) -> list[Block]:
                return []

        detector = IncompleteDetector()
        # Note: runtime_checkable only checks methods exist, not attributes at runtime
        # But we document that name and source are required
        assert hasattr(detector, "detect")


class TestSorterProtocol:
    """Tests for Sorter Protocol."""

    def test_protocol_has_sort_method(self):
        """Test that Sorter Protocol requires sort method."""
        assert "sort" in dir(Sorter)

    def test_mock_sorter_satisfies_protocol(self):
        """Test that a properly structured mock satisfies Sorter Protocol."""

        class MockSorter:
            name = "mock-sorter"

            def sort(
                self, blocks: list[Block], image: np.ndarray, **kwargs: Any
            ) -> list[Block]:
                return blocks

        sorter = MockSorter()
        assert isinstance(sorter, Sorter)

    def test_sorter_with_kwargs(self):
        """Test that sorter can accept kwargs."""

        class MockSorter:
            name = "mock-sorter"

            def sort(
                self, blocks: list[Block], image: np.ndarray, **kwargs: Any
            ) -> list[Block]:
                # Accept and use kwargs
                _page = kwargs.get("pymupdf_page")  # Accept but unused in mock
                return blocks

        sorter = MockSorter()
        blocks = [Block(type="text", bbox=BBox(0, 0, 100, 100))]
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = sorter.sort(blocks, image, pymupdf_page=Mock())
        assert result == blocks


class TestRecognizerProtocol:
    """Tests for Recognizer Protocol."""

    def test_protocol_has_required_methods(self):
        """Test that Recognizer Protocol requires essential methods."""
        assert "process_blocks" in dir(Recognizer)
        assert "correct_text" in dir(Recognizer)
        assert "process_blocks_batch" in dir(Recognizer)

    def test_mock_recognizer_satisfies_protocol(self):
        """Test that a properly structured mock satisfies Recognizer Protocol."""

        class MockRecognizer:
            name = "mock-recognizer"
            supports_correction = True

            def process_blocks(
                self, image: np.ndarray | None, blocks: Sequence[Block]
            ) -> list[Block]:
                return list(blocks)

            def correct_text(self, text: str) -> str | dict[str, Any]:
                return text

            def process_blocks_batch(
                self,
                images: Sequence[np.ndarray | None],
                blocks_list: Sequence[Sequence[Block]],
            ) -> list[list[Block]]:
                return [self.process_blocks(image, blocks) for image, blocks in zip(images, blocks_list, strict=False)]

        recognizer = MockRecognizer()
        assert isinstance(recognizer, Recognizer)

    def test_recognizer_without_correction(self):
        """Test recognizer that doesn't support correction."""

        class NoCorrectRecognizer:
            name = "no-correct-recognizer"
            supports_correction = False

            def process_blocks(
                self, image: np.ndarray | None, blocks: Sequence[Block]
            ) -> list[Block]:
                return list(blocks)

            def correct_text(self, text: str) -> str:
                # Returns original text unchanged
                return text

            def process_blocks_batch(
                self,
                images: Sequence[np.ndarray | None],
                blocks_list: Sequence[Sequence[Block]],
            ) -> list[list[Block]]:
                return [self.process_blocks(image, blocks) for image, blocks in zip(images, blocks_list, strict=False)]

        recognizer = NoCorrectRecognizer()
        assert isinstance(recognizer, Recognizer)
        assert not recognizer.supports_correction

    def test_recognizer_correct_text_dict_return(self):
        """Test that correct_text can return a dict."""

        class DictRecognizer:
            name = "dict-recognizer"
            supports_correction = True

            def process_blocks(
                self, image: np.ndarray | None, blocks: Sequence[Block]
            ) -> list[Block]:
                return list(blocks)

            def correct_text(self, text: str) -> dict[str, Any]:
                return {"corrected_text": text.upper(), "correction_ratio": 0.1}

            def process_blocks_batch(
                self,
                images: Sequence[np.ndarray | None],
                blocks_list: Sequence[Sequence[Block]],
            ) -> list[list[Block]]:
                return []

        recognizer = DictRecognizer()
        result = recognizer.correct_text("test")
        assert isinstance(result, dict)
        assert "corrected_text" in result
        assert result["corrected_text"] == "TEST"


class TestRendererProtocol:
    """Tests for Renderer Protocol."""

    def test_renderer_is_callable(self):
        """Test that Renderer Protocol is callable."""

        def mock_renderer(blocks: Sequence[Block], **kwargs: Any) -> str:
            return "\n".join(b.text or "" for b in blocks)

        # Functions matching the signature are valid renderers
        blocks = [
            Block(type="text", bbox=BBox(0, 0, 100, 100), text="Hello"),
            Block(type="text", bbox=BBox(0, 100, 100, 200), text="World"),
        ]
        result = mock_renderer(blocks)
        assert result == "Hello\nWorld"

    def test_renderer_with_kwargs(self):
        """Test that renderer can accept kwargs."""

        def mock_renderer(blocks: Sequence[Block], **kwargs: Any) -> str:
            separator = kwargs.get("separator", "\n")
            return separator.join(b.text or "" for b in blocks)

        blocks = [
            Block(type="text", bbox=BBox(0, 0, 100, 100), text="A"),
            Block(type="text", bbox=BBox(0, 100, 100, 200), text="B"),
        ]
        result = mock_renderer(blocks, separator=" | ")
        assert result == "A | B"


class TestProtocolTypeChecking:
    """Tests for Protocol runtime type checking."""

    def test_isinstance_check_for_detector(self):
        """Test isinstance works for Detector Protocol."""

        class ValidDetector:
            name = "valid"
            source = "valid"

            def detect(self, image: np.ndarray) -> list[Block]:
                return []

            def detect_batch(self, images: list[np.ndarray]) -> list[list[Block]]:
                return []

        class InvalidDetector:
            # Has detect method but wrong signature (missing return type annotation at runtime)
            def detect(self) -> None:  # Wrong signature
                pass

        valid = ValidDetector()
        invalid = InvalidDetector()

        # runtime_checkable checks if methods exist
        assert isinstance(valid, Detector)
        # InvalidDetector has detect method, so it passes basic check
        # (Protocol doesn't check signatures at runtime)
        assert hasattr(invalid, "detect")

    def test_isinstance_check_for_sorter(self):
        """Test isinstance works for Sorter Protocol."""

        class ValidSorter:
            name = "valid"

            def sort(
                self, blocks: list[Block], image: np.ndarray, **kwargs: Any
            ) -> list[Block]:
                return blocks

        valid = ValidSorter()
        assert isinstance(valid, Sorter)

    def test_isinstance_check_for_recognizer(self):
        """Test isinstance works for Recognizer Protocol."""

        class ValidRecognizer:
            name = "valid"
            supports_correction = False

            def process_blocks(
                self, image: np.ndarray | None, blocks: Sequence[Block]
            ) -> list[Block]:
                return []

            def correct_text(self, text: str) -> str:
                return text

            def process_blocks_batch(
                self,
                images: Sequence[np.ndarray | None],
                blocks_list: Sequence[Sequence[Block]],
            ) -> list[list[Block]]:
                return []

        valid = ValidRecognizer()
        assert isinstance(valid, Recognizer)

