from pathlib import Path

from pipeline.prompt import PromptManager


def test_prompt_manager_uses_default_dir_for_unknown_model():
    manager = PromptManager(model="mystery-model")

    assert manager.prompts_dir == Path("settings/prompts/default")
    assert "text_extraction" in manager.prompts


def test_prompt_manager_get_prompt_formats_kwargs():
    manager = PromptManager(model="mystery-model")

    prompt = manager.get_prompt("text_correction", "user", text="Example content")

    assert "Example content" in prompt
    assert isinstance(prompt, str)


def test_prompt_manager_returns_fallback_for_missing_category():
    manager = PromptManager(model="mystery-model")
    manager.prompts.clear()

    prompt = manager.get_prompt("unknown_category", "system")

    assert prompt == "Process this content according to unknown_category guidelines."
