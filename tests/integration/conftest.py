"""Pytest fixtures specific to integration tests.

Integration tests may require actual models to be loaded.
These fixtures handle model creation with proper cleanup.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pipeline.config import PipelineConfig
    from pipeline.factory import ComponentFactory


@pytest.fixture
def integration_config() -> PipelineConfig:
    """Create a PipelineConfig for integration testing.

    Returns:
        Validated PipelineConfig instance
    """
    from pipeline.config import PipelineConfig

    config = PipelineConfig(
        detector="paddleocr-doclayout-v2",
        recognizer="paddleocr-vl",
        use_cache=False,
    )
    config.validate()
    return config


@pytest.fixture
def component_factory(integration_config: PipelineConfig) -> ComponentFactory:
    """Create a ComponentFactory for integration testing.

    Returns:
        ComponentFactory instance
    """
    from pipeline.factory import ComponentFactory

    return ComponentFactory(integration_config)


@pytest.fixture
def integration_temp_dir(tmp_path: Path) -> Path:
    """Create a temp directory for integration tests.

    Returns:
        Path to temporary directory
    """
    test_dir = tmp_path / "integration_test"
    test_dir.mkdir()
    (test_dir / "cache").mkdir()
    (test_dir / "output").mkdir()
    (test_dir / "temp").mkdir()
    return test_dir

