# Recognizers

Text recognition backends for extracting text from detected blocks.

## Supported Recognizers

### Cloud VLM APIs

- **OpenAI**: GPT-4 Vision, GPT-4o
- **Gemini**: Gemini 2.5 Flash, Gemini 2.0 Flash
- **OpenRouter**: Access to multiple VLMs

### Local Models

- **PaddleOCR-VL**: PaddleOCR-VL-0.9B (NaViT + ERNIE-4.5-0.3B)
  - 0.9B parameters
  - 109 languages support
  - No API costs
  - Requires GPU (recommended)

!!! note
    Detailed recognizer API documentation coming soon. See [Basic Usage](../getting-started/basic-usage.md) for usage examples.
