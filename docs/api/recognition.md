# Recognition API

Text recognition and correction using VLMs.

## TextRecognizer

```python
from pipeline.recognition import TextRecognizer

recognizer = TextRecognizer(
    backend="gemini",
    model="gemini-2.5-flash",
    use_cache=True
)

# Process blocks
processed_blocks = recognizer.process_blocks(image, blocks)

# Correct text
corrected = recognizer.correct_text(raw_text)
```

!!! note "Full API Reference"
    Detailed API reference coming soon. See [Recognizers](../architecture/recognizers.md) for available recognizers.
