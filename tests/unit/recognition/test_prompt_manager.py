"""Tests for PromptManager.

Tests the prompt management system which handles:
- Directory hierarchy matching (exact, wildcard, org, default)
- YAML file loading with error handling
- Prompt retrieval with fallbacks
- Block type mapping
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import yaml

from pipeline.prompt import PromptManager


class TestPromptManagerInit:
    """Tests for PromptManager initialization."""

    def test_init_with_default_model(self):
        """Test initialization with unknown model uses default directory."""
        manager = PromptManager(model="mystery-model")

        assert manager.prompts_dir == Path("settings/prompts/default")
        assert "text_extraction" in manager.prompts
        assert manager.model == "mystery-model"
        assert manager.backend == "openai"

    def test_init_with_custom_backend(self):
        """Test initialization with custom backend."""
        manager = PromptManager(model="test-model", backend="gemini")

        assert manager.backend == "gemini"

    def test_model_without_org_prefix(self):
        """Test handling models without org prefix."""
        manager = PromptManager(model="gpt-4")

        # Should fall back to default
        assert "default" in str(manager.prompts_dir)

    def test_ultimate_fallback_when_nothing_found(self):
        """Test ultimate fallback when no directories found."""
        manager = PromptManager(model="test-model-nonexistent")

        assert str(manager.prompts_dir) == "settings/prompts/default"


class TestLoadPrompts:
    """Tests for _load_prompts method."""

    @patch("pipeline.prompt.Path.exists")
    def test_load_prompts_directory_not_found(self, mock_exists: Mock):
        """Test loading prompts when directory doesn't exist."""
        mock_exists.return_value = False

        manager = PromptManager(model="test-model")

        assert len(manager.prompts) == 0

    @patch("builtins.open", new_callable=mock_open, read_data="system: Test prompt\nuser: Test user prompt")
    @patch("pipeline.prompt.Path.exists")
    def test_load_prompts_success(self, mock_exists: Mock, mock_file: Mock):
        """Test successful prompt loading."""
        mock_exists.return_value = True

        manager = PromptManager(model="test-model")

        assert len(manager.prompts) > 0

    @patch("builtins.open", side_effect=yaml.YAMLError("Invalid YAML"))
    @patch("pipeline.prompt.Path.exists")
    def test_load_prompts_yaml_error(self, mock_exists: Mock, mock_file: Mock):
        """Test handling YAML parsing errors."""
        mock_exists.return_value = True

        manager = PromptManager(model="test-model")

        # Should handle error gracefully
        assert isinstance(manager.prompts, dict)

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    @patch("pipeline.prompt.Path.exists")
    def test_load_prompts_os_error(self, mock_exists: Mock, mock_file: Mock):
        """Test handling file system errors."""
        mock_exists.return_value = True

        manager = PromptManager(model="test-model")

        # Should handle error gracefully
        assert isinstance(manager.prompts, dict)

    @patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"))
    @patch("pipeline.prompt.Path.exists")
    def test_load_prompts_unicode_error(self, mock_exists: Mock, mock_file: Mock):
        """Test handling unicode decode errors."""
        mock_exists.return_value = True

        manager = PromptManager(model="test-model")

        # Should handle error gracefully
        assert isinstance(manager.prompts, dict)


class TestGetPrompt:
    """Tests for get_prompt method."""

    def test_get_prompt_with_kwargs_formatting(self):
        """Test prompt retrieval with kwarg formatting."""
        manager = PromptManager(model="mystery-model")

        prompt = manager.get_prompt("text_correction", "user", text="Example content")

        assert "Example content" in prompt
        assert isinstance(prompt, str)

    def test_get_prompt_nested_structure(self):
        """Test prompt retrieval from nested structure."""
        manager = PromptManager(model="mystery-model")

        # Set up nested structure
        manager.prompts = {
            "content_analysis": {"table_analysis": {"user": "Analyze this table", "system": "You are an analyst"}}
        }

        prompt = manager.get_prompt("content_analysis", "table_analysis", "user")

        assert prompt == "Analyze this table"

    def test_get_prompt_with_fallback_to_user_key(self):
        """Test fallback to 'user' key when specific key not found."""
        manager = PromptManager(model="mystery-model")
        manager.prompts = {"content_analysis": {"table_analysis": {"user": "Default prompt"}}}

        prompt = manager.get_prompt("content_analysis", "table_analysis", "nonexistent_key")

        assert prompt == "Default prompt"

    def test_get_prompt_string_value(self):
        """Test getting prompt when value is a string."""
        manager = PromptManager(model="mystery-model")
        manager.prompts = {"text_extraction": {"system": "Extract text"}}

        prompt = manager.get_prompt("text_extraction", "system")

        assert prompt == "Extract text"

    def test_get_prompt_with_format_kwargs(self):
        """Test prompt formatting with multiple kwargs."""
        manager = PromptManager(model="mystery-model")
        manager.prompts = {"test": {"user": "Process {item1} and {item2}"}}

        prompt = manager.get_prompt("test", "user", item1="first", item2="second")

        assert "first" in prompt
        assert "second" in prompt

    def test_get_prompt_missing_category_uses_fallback(self):
        """Test fallback when category not found."""
        manager = PromptManager(model="mystery-model")
        manager.prompts.clear()

        prompt = manager.get_prompt("unknown_category", "system")

        assert prompt == "Process this content according to unknown_category guidelines."

    def test_get_prompt_handles_key_error(self):
        """Test handling KeyError in prompt retrieval."""
        manager = PromptManager(model="mystery-model")
        manager.prompts = {"test": "not a dict"}

        prompt = manager.get_prompt("test", "nonexistent")

        # Should fall back to fallback prompt
        assert isinstance(prompt, str)

    def test_get_prompt_handles_type_error(self):
        """Test handling TypeError in prompt retrieval."""
        manager = PromptManager(model="mystery-model")
        manager.prompts = {"test": None}

        prompt = manager.get_prompt("test", "system")

        # Should fall back to fallback prompt
        assert isinstance(prompt, str)


class TestGetFallbackPrompt:
    """Tests for _get_fallback_prompt method."""

    def test_fallback_text_extraction_system(self):
        """Test fallback for text extraction system prompt."""
        manager = PromptManager(model="test-model")

        prompt = manager._get_fallback_prompt("text_extraction", "system")

        assert "OCR system" in prompt
        assert "Extract" in prompt

    def test_fallback_text_extraction_user(self):
        """Test fallback for text extraction user prompt."""
        manager = PromptManager(model="test-model")

        prompt = manager._get_fallback_prompt("text_extraction", "user")

        assert "Extract" in prompt
        assert "text" in prompt

    def test_fallback_content_analysis_table(self):
        """Test fallback for table analysis."""
        manager = PromptManager(model="test-model")

        prompt = manager._get_fallback_prompt("content_analysis", "table_analysis")

        assert "table" in prompt.lower()
        assert "JSON" in prompt

    def test_fallback_content_analysis_figure(self):
        """Test fallback for figure analysis."""
        manager = PromptManager(model="test-model")

        prompt = manager._get_fallback_prompt("content_analysis", "figure_analysis")

        assert "image" in prompt.lower() or "Analyze" in prompt
        assert "JSON" in prompt

    def test_fallback_text_correction_with_kwargs(self):
        """Test fallback text correction with formatting."""
        manager = PromptManager(model="test-model")

        prompt = manager._get_fallback_prompt("text_correction", "user", text="Sample text")

        assert "Sample text" in prompt
        assert "Correct" in prompt

    def test_fallback_unknown_category(self):
        """Test fallback for completely unknown category."""
        manager = PromptManager(model="test-model")

        prompt = manager._get_fallback_prompt("unknown_category", "unknown_type")

        assert "Process this content" in prompt
        assert "unknown_category" in prompt

    def test_fallback_handles_format_error(self):
        """Test fallback handles ValueError in formatting."""
        manager = PromptManager(model="test-model")

        # Text correction expects {text} but we don't provide it
        prompt = manager._get_fallback_prompt("text_correction", "user")

        # Should return ultimate fallback
        assert isinstance(prompt, str)


class TestGetPromptForBlockType:
    """Tests for get_prompt_for_block_type method."""

    def test_get_prompt_for_table(self):
        """Test getting prompt for table block type."""
        manager = PromptManager(model="mystery-model")

        prompt = manager.get_prompt_for_block_type("table")

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_prompt_for_figure(self):
        """Test getting prompt for figure block type."""
        manager = PromptManager(model="mystery-model")

        prompt = manager.get_prompt_for_block_type("figure")

        assert isinstance(prompt, str)

    def test_get_prompt_for_formula(self):
        """Test getting prompt for formula block type."""
        manager = PromptManager(model="mystery-model")

        prompt = manager.get_prompt_for_block_type("formula")

        assert isinstance(prompt, str)

    def test_get_prompt_for_title(self):
        """Test getting prompt for title block type."""
        manager = PromptManager(model="mystery-model")

        prompt = manager.get_prompt_for_block_type("title")

        assert isinstance(prompt, str)

    def test_get_prompt_for_list(self):
        """Test getting prompt for list block type."""
        manager = PromptManager(model="mystery-model")

        prompt = manager.get_prompt_for_block_type("list")

        assert isinstance(prompt, str)

    def test_get_prompt_for_plain_text(self):
        """Test getting prompt for plain text block type."""
        manager = PromptManager(model="mystery-model")

        prompt = manager.get_prompt_for_block_type("plain text")

        assert isinstance(prompt, str)

    def test_get_prompt_for_unknown_block_type(self):
        """Test getting prompt for unknown block type defaults to figure_analysis."""
        manager = PromptManager(model="mystery-model")

        prompt = manager.get_prompt_for_block_type("unknown_type")

        assert isinstance(prompt, str)


class TestGetGeminiPromptForBlockType:
    """Tests for deprecated get_gemini_prompt_for_block_type method."""

    def test_deprecated_method_calls_new_method(self):
        """Test that deprecated method delegates to new method."""
        manager = PromptManager(model="mystery-model")

        old_prompt = manager.get_gemini_prompt_for_block_type("table")
        new_prompt = manager.get_prompt_for_block_type("table")

        assert old_prompt == new_prompt


class TestReloadPrompts:
    """Tests for reload_prompts method."""

    @patch.object(PromptManager, "_load_prompts")
    def test_reload_prompts(self, mock_load: Mock):
        """Test that reload_prompts calls _load_prompts."""
        mock_load.return_value = {"test": "value"}

        manager = PromptManager(model="test-model")

        manager.reload_prompts()

        # _load_prompts should be called again
        assert mock_load.call_count >= 2  # Once in __init__, once in reload

    def test_reload_prompts_updates_prompts(self):
        """Test that reload updates the prompts dictionary."""
        manager = PromptManager(model="mystery-model")

        # Clear prompts
        manager.prompts.clear()
        assert len(manager.prompts) == 0

        # Reload should restore prompts
        manager.reload_prompts()

        assert len(manager.prompts) > 0
