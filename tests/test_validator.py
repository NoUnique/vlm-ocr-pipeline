"""Tests for pipeline validator."""

from __future__ import annotations

from pipeline.validator import PipelineValidator


def test_validator_valid_combination():
    """Test validator accepts valid combinations."""
    result = PipelineValidator.validate("doclayout-yolo", "mineru-xycut")
    
    assert result.is_valid is True


def test_validator_invalid_detector():
    """Test validator rejects unknown detector."""
    result = PipelineValidator.validate("unknown-detector", "none")
    
    assert result.is_valid is False
    assert "Unknown detector" in result.message


def test_validator_recommended_combination():
    """Test validator identifies recommended combinations."""
    result = PipelineValidator.validate("mineru-vlm", "mineru-vlm")
    
    assert result.is_valid is True
    assert result.is_recommended is True
    assert "efficient" in result.message.lower()


def test_validator_get_compatible_sorters():
    """Test validator returns compatible sorters."""
    sorters = PipelineValidator.get_compatible_sorters("doclayout-yolo")
    
    assert "pymupdf" in sorters
    assert "mineru-layoutreader" in sorters
    assert "mineru-xycut" in sorters
    assert "olmocr-vlm" in sorters


def test_validator_mineru_vlm_compatible_sorters():
    """Test validator returns all sorters for mineru-vlm detector."""
    sorters = PipelineValidator.get_compatible_sorters("mineru-vlm")
    
    assert "pymupdf" in sorters
    assert "mineru-vlm" in sorters
    assert "olmocr-vlm" in sorters
    assert "mineru-xycut" in sorters


def test_validator_get_recommended_combinations():
    """Test validator returns recommended combinations."""
    recommendations = PipelineValidator.get_recommended_combinations()
    
    assert len(recommendations) > 0
    assert any(det == "mineru-vlm" and sort == "mineru-vlm" 
               for det, sort, _ in recommendations)

