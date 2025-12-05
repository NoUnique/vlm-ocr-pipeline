"""Tests for BaseStage and related classes.

Tests cover:
- BaseStage abstract class behavior
- StageResult dataclass
- StageError exception
- Timing and error handling
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from pipeline.stages.base import BaseStage, StageError, StageResult


# Test implementations
class ConcreteStage(BaseStage[str, str]):
    """A concrete stage for testing."""

    name = "test-stage"

    def _process_impl(self, input_data: str, **context: Any) -> str:
        """Process input and return uppercase."""
        return input_data.upper()


class FailingStage(BaseStage[str, str]):
    """A stage that always fails."""

    name = "failing-stage"

    def _process_impl(self, input_data: str, **context: Any) -> str:
        """Raise an error."""
        raise ValueError(f"Processing failed for: {input_data}")


class SlowStage(BaseStage[str, str]):
    """A stage that takes time."""

    name = "slow-stage"

    def __init__(self, delay: float = 0.1):
        self.delay = delay

    def _process_impl(self, input_data: str, **context: Any) -> str:
        """Process with delay."""
        time.sleep(self.delay)
        return input_data


class TestStageError:
    """Tests for StageError exception."""

    def test_stage_error_creation(self):
        """Test creating a StageError."""
        error = StageError("detection", "Image is too dark")
        assert error.stage_name == "detection"
        assert error.cause is None
        assert "[detection] Image is too dark" in str(error)

    def test_stage_error_with_cause(self):
        """Test creating a StageError with a cause."""
        original_error = ValueError("Invalid input")
        error = StageError("recognition", "Failed to extract text", cause=original_error)
        assert error.stage_name == "recognition"
        assert error.cause is original_error
        assert "[recognition] Failed to extract text" in str(error)


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_stage_result_creation(self):
        """Test creating a StageResult."""
        result = StageResult(
            data="Hello World",
            stage_name="test-stage",
            processing_time_ms=150.5,
        )
        assert result.data == "Hello World"
        assert result.stage_name == "test-stage"
        assert result.processing_time_ms == 150.5
        assert result.metadata == {}

    def test_stage_result_with_metadata(self):
        """Test creating a StageResult with metadata."""
        result = StageResult(
            data=["block1", "block2"],
            stage_name="detection",
            processing_time_ms=200.0,
            metadata={"num_blocks": 2, "confidence_avg": 0.95},
        )
        assert result.metadata["num_blocks"] == 2
        assert result.metadata["confidence_avg"] == 0.95

    def test_stage_result_processing_time_sec(self):
        """Test processing_time_sec property."""
        result = StageResult(
            data="test",
            stage_name="test-stage",
            processing_time_ms=1500.0,
        )
        assert result.processing_time_sec == 1.5


class TestBaseStageAbstract:
    """Tests for BaseStage abstract behavior."""

    def test_cannot_instantiate_base_stage(self):
        """Test that BaseStage cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseStage()  # type: ignore

    def test_concrete_stage_can_be_instantiated(self):
        """Test that a concrete stage can be instantiated."""
        stage = ConcreteStage()
        assert stage.name == "test-stage"


class TestBaseStageProcess:
    """Tests for process method."""

    def test_process_basic(self):
        """Test basic processing."""
        stage = ConcreteStage()
        result = stage.process("hello")
        assert result == "HELLO"

    def test_process_with_context(self):
        """Test processing with context."""

        class ContextStage(BaseStage[str, str]):
            name = "context-stage"

            def _process_impl(self, input_data: str, **context: Any) -> str:
                prefix = context.get("prefix", "")
                return f"{prefix}{input_data}"

        stage = ContextStage()
        result = stage.process("world", prefix="hello ")
        assert result == "hello world"

    def test_process_error_handling(self):
        """Test that errors are wrapped in StageError."""
        stage = FailingStage()

        with pytest.raises(StageError) as exc_info:
            stage.process("test")

        assert exc_info.value.stage_name == "failing-stage"
        assert "Processing failed for: test" in str(exc_info.value)
        assert isinstance(exc_info.value.cause, ValueError)


class TestBaseStageProcessWithResult:
    """Tests for process_with_result method."""

    def test_process_with_result_basic(self):
        """Test process_with_result returns StageResult."""
        stage = ConcreteStage()
        result = stage.process_with_result("hello")

        assert isinstance(result, StageResult)
        assert result.data == "HELLO"
        assert result.stage_name == "test-stage"
        assert result.processing_time_ms > 0

    def test_process_with_result_timing(self):
        """Test that processing time is measured."""
        stage = SlowStage(delay=0.05)
        result = stage.process_with_result("test")

        assert result.processing_time_ms >= 50  # At least 50ms
        assert result.processing_time_sec >= 0.05

    def test_process_with_result_error(self):
        """Test error handling in process_with_result."""
        stage = FailingStage()

        with pytest.raises(StageError):
            stage.process_with_result("test")


class TestBaseStageProcessBatch:
    """Tests for process_batch method."""

    def test_process_batch(self):
        """Test batch processing."""
        stage = ConcreteStage()
        inputs = ["hello", "world", "test"]
        results = stage.process_batch(inputs)

        assert results == ["HELLO", "WORLD", "TEST"]

    def test_process_batch_empty(self):
        """Test batch processing with empty list."""
        stage = ConcreteStage()
        results = stage.process_batch([])
        assert results == []

    def test_process_batch_with_context(self):
        """Test batch processing with context."""

        class ContextStage(BaseStage[str, str]):
            name = "context-stage"

            def _process_impl(self, input_data: str, **context: Any) -> str:
                suffix = context.get("suffix", "")
                return f"{input_data}{suffix}"

        stage = ContextStage()
        inputs = ["a", "b", "c"]
        results = stage.process_batch(inputs, suffix="!")

        assert results == ["a!", "b!", "c!"]


class TestBaseStageCleanup:
    """Tests for cleanup method."""

    def test_cleanup_default(self):
        """Test that default cleanup does nothing (no error)."""
        stage = ConcreteStage()
        stage.cleanup()  # Should not raise

    def test_cleanup_custom(self):
        """Test custom cleanup implementation."""

        class CleanupStage(BaseStage[str, str]):
            name = "cleanup-stage"

            def __init__(self):
                self.cleaned = False

            def _process_impl(self, input_data: str, **context: Any) -> str:
                return input_data

            def cleanup(self) -> None:
                self.cleaned = True

        stage = CleanupStage()
        assert not stage.cleaned
        stage.cleanup()
        assert stage.cleaned


class TestBaseStageRepr:
    """Tests for __repr__ method."""

    def test_repr(self):
        """Test string representation."""
        stage = ConcreteStage()
        assert repr(stage) == "ConcreteStage(name='test-stage')"


class TestBaseStageLogging:
    """Tests for logging behavior."""

    def test_process_logs_on_success(self, caplog):
        """Test that successful processing logs debug info."""
        import logging

        with caplog.at_level(logging.DEBUG):
            stage = ConcreteStage()
            stage.process("test")

        assert "test-stage completed" in caplog.text

    def test_process_logs_on_error(self, caplog):
        """Test that failed processing logs error info."""
        import logging

        with caplog.at_level(logging.ERROR):
            stage = FailingStage()
            with pytest.raises(StageError):
                stage.process("test")

        assert "failing-stage failed" in caplog.text

