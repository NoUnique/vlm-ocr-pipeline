# Recognition API

Text recognition and correction using Vision Language Models.

## Overview

The recognition module provides:

- **TextRecognizer**: Main class for text extraction from blocks
- **Multiple backends**: OpenAI, Gemini, PaddleOCR-VL, DeepSeek-OCR
- **Caching**: Content-based caching to avoid reprocessing
- **Rate limiting**: Automatic throttling for API backends

## Quick Start

```python
from pipeline.recognition import TextRecognizer
import numpy as np

# Create recognizer with Gemini backend
recognizer = TextRecognizer(
    backend="gemini",
    model="gemini-2.5-flash",
    use_cache=True,
)

# Process blocks
image = np.zeros((1000, 800, 3), dtype=np.uint8)
processed_blocks = recognizer.process_blocks(image, blocks)

# Correct text
corrected = recognizer.correct_text(raw_text)
```

## TextRecognizer

### Constructor

```python
class TextRecognizer:
    def __init__(
        self,
        backend: str = "gemini",
        model: str = "gemini-2.5-flash",
        use_cache: bool = True,
        cache_dir: str | Path = ".cache",
        gemini_tier: str = "free",
        recognizer_backend: str | None = None,
        prompts_dir: str | Path | None = None,
        **kwargs,
    ):
        """Initialize text recognizer.

        Args:
            backend: API backend ("gemini", "openai", "paddleocr-vl", "deepseek-ocr")
            model: Model name
            use_cache: Enable content-based caching
            cache_dir: Cache directory path
            gemini_tier: Gemini API tier for rate limiting
            recognizer_backend: Inference backend for local models
            prompts_dir: Custom prompts directory
        """
```

### Methods

#### `process_blocks`

```python
def process_blocks(
    self,
    image: np.ndarray,
    blocks: Sequence[Block],
) -> list[Block]:
    """Extract text from blocks.

    Args:
        image: Full page image as numpy array
        blocks: List of blocks to process

    Returns:
        List of blocks with text field populated
    """
```

#### `correct_text`

```python
def correct_text(self, text: str) -> str | dict[str, Any]:
    """Correct extracted text using VLM.

    Args:
        text: Raw extracted text

    Returns:
        Corrected text string, or dict with:
        - corrected_text: str
        - correction_ratio: float (0.0 = no change)
    """
```

## Available Backends

### Cloud VLM APIs

| Backend | Models | Rate Limits | Cost |
|---------|--------|-------------|------|
| `gemini` | gemini-2.5-flash, gemini-2.0-flash | 15 RPM (free) | Free tier available |
| `openai` | gpt-4o, gpt-4-turbo | Varies by tier | Pay per token |
| `openrouter` | Multiple VLMs | Varies by model | Pay per token |

### Local Models

| Backend | Model | Parameters | Languages |
|---------|-------|------------|-----------|
| `paddleocr-vl` | PaddleOCR-VL-0.9B | 0.9B | 109 languages |
| `deepseek-ocr` | DeepSeek-OCR | - | Contextual compression |

## Backend Configuration

### Gemini Backend

```python
recognizer = TextRecognizer(
    backend="gemini",
    model="gemini-2.5-flash",
    gemini_tier="free",  # free, tier1, tier2, tier3
)
```

**Environment Variable**: `GEMINI_API_KEY`

**Rate Limits (Free Tier)**:
- 15 requests per minute
- 1,500,000 tokens per minute
- 1,500 requests per day

### OpenAI Backend

```python
recognizer = TextRecognizer(
    backend="openai",
    model="gpt-4o",
)
```

**Environment Variable**: `OPENAI_API_KEY`

### PaddleOCR-VL Backend

```python
recognizer = TextRecognizer(
    backend="paddleocr-vl",
    recognizer_backend="pytorch",  # pytorch, vllm, sglang
)
```

**Requirements**: GPU recommended, PaddleX installation

### DeepSeek-OCR Backend

```python
recognizer = TextRecognizer(
    backend="deepseek-ocr",
    recognizer_backend="hf",  # hf, vllm
)
```

## Recognizer Protocol

All recognizers implement the `Recognizer` protocol:

```python
from typing import Protocol, Any, Sequence
import numpy as np
from pipeline.types import Block

class Recognizer(Protocol):
    """Protocol for text recognizers."""

    def process_blocks(
        self,
        image: np.ndarray,
        blocks: Sequence[Block],
    ) -> list[Block]:
        """Extract text from blocks."""
        ...

    def correct_text(self, text: str) -> str | dict[str, Any]:
        """Correct extracted text."""
        ...
```

## Caching

The recognizer uses content-based caching to avoid reprocessing:

```python
# Cache key = hash(block_image + block_type + prompt)
recognizer = TextRecognizer(
    backend="gemini",
    use_cache=True,
    cache_dir=".cache",
)
```

## Implementing Custom Recognizers

```python
from pipeline.types import Block
from typing import Sequence, Any
import numpy as np

class MyRecognizer:
    """Custom recognizer implementation."""

    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def process_blocks(
        self,
        image: np.ndarray,
        blocks: Sequence[Block],
    ) -> list[Block]:
        """Extract text from blocks."""
        result_blocks = []

        for block in blocks:
            cropped = block.bbox.crop(image)
            text = self.model.recognize(cropped)
            block.text = text
            result_blocks.append(block)

        return result_blocks

    def correct_text(self, text: str) -> dict[str, Any]:
        """Correct extracted text."""
        corrected = self.model.correct(text)
        return {
            "corrected_text": corrected,
            "correction_ratio": calculate_ratio(text, corrected),
        }
```

## CLI Usage

```bash
# Default recognizer (gemini)
python main.py --input doc.pdf

# Specific recognizer
python main.py --input doc.pdf --recognizer gpt-4o

# Local model with backend
python main.py --input doc.pdf --recognizer paddleocr-vl --recognizer-backend vllm

# Check rate limit status
python main.py --rate-limit-status --recognizer gemini-2.5-flash --gemini-tier free
```

## See Also

- [Recognizers Architecture](../architecture/recognizers.md) - Detailed backend comparison
- [Basic Usage](../getting-started/basic-usage.md) - Usage examples
- [Types API](types.md) - Block class reference
