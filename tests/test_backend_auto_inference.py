"""Tests for automatic backend inference from recognizer names."""

from __future__ import annotations

import pytest

from pipeline.backend_validator import resolve_recognizer_backend


class TestRecognizerBackendAutoInference:
    """Test automatic backend inference from recognizer model names."""

    def test_gemini_models_auto_select_gemini_backend(self) -> None:
        """Gemini models should auto-select gemini backend."""
        # Test various Gemini model names
        test_cases = [
            "gemini-2.5-flash",
            "gemini-2.0-pro",
            "gemini-1.5-flash",
            "gemini-pro",
        ]

        for model_name in test_cases:
            backend, error = resolve_recognizer_backend(model_name, None)
            assert backend == "gemini", f"Expected 'gemini' backend for {model_name}, got {backend}"
            assert error is None, f"Expected no error for {model_name}, got {error}"

    def test_gpt_models_auto_select_openai_backend(self) -> None:
        """GPT/ChatGPT models should auto-select openai backend."""
        # Test various GPT model names
        test_cases = [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "chatgpt",
        ]

        for model_name in test_cases:
            backend, error = resolve_recognizer_backend(model_name, None)
            assert backend == "openai", f"Expected 'openai' backend for {model_name}, got {backend}"
            assert error is None, f"Expected no error for {model_name}, got {error}"

    def test_huggingface_format_auto_selects_openai_backend(self) -> None:
        """Models in {org}/{model} format should auto-select openai backend for OpenRouter."""
        # Test HuggingFace-style model names (used with OpenRouter)
        test_cases = [
            "meta-llama/Llama-3-8b",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "google/gemma-7b-it",
            "01-ai/Yi-34B-Chat",
        ]

        for model_name in test_cases:
            backend, error = resolve_recognizer_backend(model_name, None)
            assert backend == "openai", f"Expected 'openai' backend for HuggingFace format {model_name}, got {backend}"
            assert error is None, f"Expected no error for {model_name}, got {error}"

    def test_registered_models_use_default_backend(self) -> None:
        """Registered models (paddleocr-vl, deepseek-ocr) should use their default backends."""
        # Test registered models from models.yaml
        test_cases = [
            ("paddleocr-vl", "pytorch"),  # default_backend: pytorch (native)
            ("deepseek-ocr", "hf"),  # default_backend: hf
        ]

        for model_name, expected_backend in test_cases:
            backend, error = resolve_recognizer_backend(model_name, None)
            assert backend == expected_backend, f"Expected '{expected_backend}' backend for {model_name}, got {backend}"
            assert error is None, f"Expected no error for {model_name}, got {error}"

    def test_user_specified_backend_takes_precedence(self) -> None:
        """User-specified backend should override auto-inference (if valid)."""
        # Test that explicit backend specification works
        backend, error = resolve_recognizer_backend("paddleocr-vl", "vllm")
        assert backend == "vllm", "User-specified backend should be used"
        assert error is None

    def test_invalid_backend_falls_back_to_default(self) -> None:
        """Invalid backend should fall back to default with error message."""
        backend, error = resolve_recognizer_backend("paddleocr-vl", "invalid-backend")
        assert backend == "pytorch", "Should fall back to default backend"
        assert error is not None, "Should return error message"
        assert "not supported" in error.lower(), "Error should mention unsupported backend"

    def test_unknown_recognizer_without_slash_returns_error(self) -> None:
        """Unknown recognizer without slash should return error."""
        backend, error = resolve_recognizer_backend("unknown-model", None)
        assert backend is None, "Unknown model should return None backend"
        assert error is not None, "Unknown model should return error"
        assert "unknown recognizer" in error.lower(), "Error should mention unknown recognizer"

    def test_unknown_recognizer_with_slash_uses_openai(self) -> None:
        """Unknown recognizer with slash (HuggingFace format) should use openai backend."""
        backend, error = resolve_recognizer_backend("custom-org/custom-model", None)
        assert backend == "openai", "Unknown HuggingFace format should use openai"
        assert error is None, "Should not return error for HuggingFace format"


class TestBackendInferenceIntegration:
    """Integration tests for backend inference in realistic scenarios."""

    def test_gemini_backend_inference_without_flag(self) -> None:
        """Test that gemini models work without --recognizer-backend flag."""
        # This simulates: python main.py --recognizer gemini-2.5-flash
        backend, error = resolve_recognizer_backend("gemini-2.5-flash", None)
        assert backend == "gemini"
        assert error is None

    def test_openrouter_model_inference(self) -> None:
        """Test that OpenRouter models work without --recognizer-backend flag."""
        # This simulates: python main.py --recognizer meta-llama/Llama-3-8b
        backend, error = resolve_recognizer_backend("meta-llama/Llama-3-8b", None)
        assert backend == "openai"
        assert error is None

    def test_local_model_with_explicit_backend(self) -> None:
        """Test local model with explicit backend selection."""
        # This simulates: python main.py --recognizer paddleocr-vl --recognizer-backend vllm
        backend, error = resolve_recognizer_backend("paddleocr-vl", "vllm")
        assert backend == "vllm"
        assert error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
