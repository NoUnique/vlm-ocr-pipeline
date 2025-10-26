"""Tests for resource management utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.resources import (
    ManagedResource,
    managed_image_processing,
    managed_numpy_array,
    open_pdf_document,
)


class TestOpenPdfDocument:
    """Tests for PyMuPDF document context manager."""

    def test_opens_and_closes_document(self, tmp_path: Path):
        """Test that PDF document is properly opened and closed."""
        # Create a simple PDF for testing (we'll skip if PyMuPDF not available)
        pytest.importorskip("fitz")

        # Create minimal PDF using PyMuPDF
        import fitz

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()  # type: ignore[attr-defined]
        doc.insert_page(0, text="Test page")  # type: ignore[attr-defined]
        doc.save(str(pdf_path))  # type: ignore[attr-defined]
        doc.close()  # type: ignore[attr-defined]

        # Test context manager
        with open_pdf_document(pdf_path) as doc:
            assert doc.page_count == 1
            page = doc.load_page(0)
            assert page is not None

        # Document should be closed after context exit (no way to directly test,
        # but we can verify no errors occur)

    def test_raises_on_missing_pymupdf(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Test that RuntimeError is raised when PyMuPDF is not available."""
        # Mock PyMuPDF as unavailable
        import pipeline.resources

        monkeypatch.setattr(pipeline.resources, "_HAS_PYMUPDF", False)
        monkeypatch.setattr(pipeline.resources, "fitz", None)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with pytest.raises(RuntimeError, match="PyMuPDF .* is not available"):
            with open_pdf_document(pdf_path):
                pass

    def test_raises_on_missing_file(self):
        """Test that FileNotFoundError is raised for missing PDF."""
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            with open_pdf_document("nonexistent.pdf"):
                pass


class TestManagedNumpyArray:
    """Tests for numpy array context manager."""

    def test_manages_single_array(self):
        """Test managing a single numpy array."""
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        with managed_numpy_array(arr) as (managed,):
            assert managed.shape == (100, 100, 3)
            assert managed.dtype == np.uint8

        # Array cleanup is automatic (no direct test possible)

    def test_manages_multiple_arrays(self):
        """Test managing multiple numpy arrays."""
        arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
        arr2 = np.ones((50, 50), dtype=np.float32)

        with managed_numpy_array(arr1, arr2) as (m1, m2):
            assert m1.shape == (100, 100, 3)
            assert m2.shape == (50, 50)


class TestManagedImageProcessing:
    """Tests for managed image processing context manager."""

    def test_runs_without_errors(self):
        """Test that context manager runs without errors."""
        with managed_image_processing():
            # Create some temporary arrays
            temp = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = temp * 2  # Simulate processing

        # Should complete without errors


class TestManagedResource:
    """Tests for generic managed resource."""

    def test_calls_cleanup_function(self):
        """Test that cleanup function is called."""
        cleanup_called = []

        def cleanup(resource: dict) -> None:
            cleanup_called.append(True)
            resource.clear()

        resource = {"data": [1, 2, 3]}

        with ManagedResource(resource, cleanup, "test-resource") as res:
            assert res == {"data": [1, 2, 3]}

        assert len(cleanup_called) == 1
        assert resource == {}

    def test_handles_cleanup_errors(self):
        """Test that cleanup errors are handled gracefully."""

        def failing_cleanup(resource: object) -> None:
            raise ValueError("Cleanup failed")

        resource = object()

        # Should not raise, just log warning
        with ManagedResource(resource, failing_cleanup, "failing-resource"):
            pass
