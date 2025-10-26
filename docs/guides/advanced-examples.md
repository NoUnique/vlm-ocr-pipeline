# Advanced Examples

Complex use cases and advanced configurations for VLM OCR Pipeline.

## Multi-Column Academic Papers

Process research papers with complex multi-column layouts:

```bash
python main.py --input paper.pdf \
    --detector paddleocr-doclayout-v2 \
    --sorter pymupdf \
    --backend gemini \
    --dpi 300 \
    --output results/papers/
```

**Why this configuration?**

- **PP-DocLayoutV2**: High-quality detector for academic layouts
- **PyMuPDF sorter**: Best for multi-column detection
- **DPI 300**: Higher quality for small text
- **Gemini**: Cost-effective with good accuracy

## Large Batch Processing

Process hundreds of PDFs efficiently:

```bash
# Use local model to avoid API costs
python main.py --input documents/ \
    --detector paddleocr-doclayout-v2 \
    --recognizer paddleocr-vl \
    --cache-dir .cache/ \
    --output batch-results/
```

**Best practices**:

- Use PaddleOCR-VL to avoid API costs
- Enable caching to skip duplicates
- Process in batches if memory constrained
- Monitor disk space for large outputs

## Mixed Content Documents

Documents with tables, figures, and equations:

```bash
python main.py --input textbook.pdf \
    --detector mineru-vlm \
    --sorter mineru-vlm \
    --backend openai \
    --model gpt-4o \
    --output textbooks/
```

**Why VLM detection?**

- Better understanding of complex layouts
- Accurate table structure detection
- Equation recognition
- Figure caption association

## Cost-Optimized Processing

Minimize API costs while maintaining quality:

```bash
# Use Gemini free tier with caching
python main.py --input doc.pdf \
    --detector doclayout-yolo \
    --sorter mineru-xycut \
    --backend gemini \
    --gemini-tier free \
    --cache-dir .cache/ \
    --max-pages 10  # Test first

# Check rate limits
python main.py --rate-limit-status --backend gemini --gemini-tier free
```

**Tips**:

1. Test on few pages first (`--max-pages`)
2. Use caching to avoid reprocessing
3. Monitor free tier limits
4. Switch to local model if limits reached

## Programmatic Usage

Use the pipeline in your Python code:

```python
from pathlib import Path
from pipeline import Pipeline

# Initialize pipeline
pipeline = Pipeline(
    detector_name="doclayout-yolo",
    sorter_name="mineru-xycut",
    backend="gemini",
    model="gemini-2.5-flash",
    cache_dir=Path(".cache"),
    output_dir=Path("output"),
    use_cache=True
)

# Process single PDF
result = pipeline.process_single_pdf(
    pdf_path=Path("document.pdf"),
    max_pages=5
)

# Access results
for page in result.pages:
    print(f"Page {page.page_num}")
    print(f"Text: {page.corrected_text}")
    print(f"Blocks: {len(page.blocks)}")
    print(f"Correction ratio: {page.correction_ratio}")
```

## Custom Detector Integration

Integrate your own detector:

```python
# my_detector.py
import numpy as np
from pipeline.types import Block, BBox

class CustomDetector:
    """Custom layout detector."""

    def detect(self, image: np.ndarray) -> list[Block]:
        """Detect layout blocks."""
        # Your detection logic here
        blocks = []

        # Example: detect blocks
        for detection in your_model.predict(image):
            block = Block(
                type=detection.label,  # "text", "table", etc.
                bbox=BBox(
                    x0=int(detection.box[0]),
                    y0=int(detection.box[1]),
                    x1=int(detection.box[2]),
                    y1=int(detection.box[3])
                ),
                detection_confidence=float(detection.confidence),
                source="custom-detector"
            )
            blocks.append(block)

        return blocks

# Register in factory
# pipeline/layout/detection/__init__.py
def create_detector(name: str, **kwargs) -> Detector:
    if name == "custom":
        from .custom_detector import CustomDetector
        return CustomDetector(**kwargs)
    # ...

# Use it
python main.py --input doc.pdf --detector custom --backend gemini
```

## Custom Prompts

Override default prompts for specific use cases:

```yaml
# settings/prompts/gemini/custom_text_extraction.yaml
system: |
  You are a specialized OCR system for medical documents.
  Pay special attention to:
  - Drug names and dosages
  - Medical terminology
  - Patient information
  - Dates and times

user: |
  Extract text from this medical document image.
  Preserve all formatting, especially:
  - Tables with patient data
  - Lists of medications
  - Diagnostic results

  Output in Markdown format.

fallback: |
  [OCR failed - manual review required]
```

Then use `PromptManager` in code:

```python
from pipeline.prompt import PromptManager

pm = PromptManager(model="gemini-2.5-flash")
custom_prompt = pm.get_prompt("custom_text_extraction", "user")
```

## Multi-Language Documents

Process documents in multiple languages:

```bash
# PaddleOCR-VL supports 109 languages
python main.py --input multilang.pdf \
    --detector paddleocr-doclayout-v2 \
    --recognizer paddleocr-vl \
    --output multilang-results/
```

**Supported languages** (PaddleOCR-VL):

- European: English, Spanish, French, German, Italian, Portuguese, etc.
- Asian: Chinese (Simplified/Traditional), Japanese, Korean, Thai, Vietnamese
- Middle Eastern: Arabic, Hebrew, Persian
- And 100+ more

## Rate Limit Monitoring

Monitor and adapt to rate limits in real-time:

```python
from pipeline.recognition.api.ratelimit import rate_limiter

# Check status before processing
status = rate_limiter.get_status()
print(f"Current RPM: {status['current']['rpm']} / {status['limits']['rpm']}")
print(f"RPD: {status['current']['rpd']} / {status['limits']['rpd']}")

# Process with automatic throttling
if rate_limiter.wait_if_needed(estimated_tokens=1000):
    # Process page
    result = pipeline.process_page(image, page_num=1)
else:
    print("Daily limit exceeded")
```

## Selective Page Processing

Process specific pages of interest:

```bash
# Process only pages with tables
python main.py --input report.pdf \
    --pages 5,12,18,25 \
    --detector paddleocr-doclayout-v2 \
    --backend gemini

# Process chapter intros (every 10 pages)
python main.py --input book.pdf \
    --pages 1,11,21,31,41,51 \
    --backend gemini
```

## Error Recovery

Handle processing errors gracefully:

```python
from pipeline import Pipeline
from pipeline.exceptions import ProcessingError, APIError

pipeline = Pipeline(backend="gemini")

try:
    result = pipeline.process_single_pdf("document.pdf")
except APIError as e:
    print(f"API error: {e}")
    # Retry with different backend
    pipeline.backend = "openai"
    result = pipeline.process_single_pdf("document.pdf")
except ProcessingError as e:
    print(f"Processing error: {e}")
    # Continue with next document
    pass
```

## Performance Benchmarking

Compare different configurations:

```python
import time
from pathlib import Path

configurations = [
    ("doclayout-yolo", "mineru-xycut", "gemini"),
    ("paddleocr-doclayout-v2", "pymupdf", "gemini"),
    ("mineru-vlm", "mineru-vlm", "openai"),
]

test_pdf = Path("test.pdf")
results = {}

for detector, sorter, backend in configurations:
    pipeline = Pipeline(
        detector_name=detector,
        sorter_name=sorter,
        backend=backend
    )

    start = time.time()
    result = pipeline.process_single_pdf(test_pdf, max_pages=3)
    elapsed = time.time() - start

    results[f"{detector}/{sorter}/{backend}"] = {
        "time": elapsed,
        "pages": len(result.pages),
        "avg_correction_ratio": sum(p.correction_ratio for p in result.pages) / len(result.pages)
    }

# Print comparison
for config, metrics in results.items():
    print(f"{config}:")
    print(f"  Time: {metrics['time']:.2f}s")
    print(f"  Avg correction: {metrics['avg_correction_ratio']:.2%}")
```

## Integration with External Tools

### Export to Different Formats

```python
from pipeline import Pipeline
import json

pipeline = Pipeline(backend="gemini")
result = pipeline.process_single_pdf("document.pdf")

# Export to JSON
with open("output.json", "w") as f:
    json.dump([page.to_dict() for page in result.pages], f, indent=2)

# Export to plain text
with open("output.txt", "w") as f:
    for page in result.pages:
        f.write(f"=== Page {page.page_num} ===\n")
        f.write(page.corrected_text)
        f.write("\n\n")
```

### Post-Processing

```python
def post_process_markdown(text: str) -> str:
    """Custom post-processing for markdown output."""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix common OCR errors
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")

    # Normalize quotes
    text = text.replace(""", '"').replace(""", '"')

    return text

# Apply to results
for page in result.pages:
    page.corrected_text = post_process_markdown(page.corrected_text)
```

## Best Practices Summary

!!! tip "Performance"
    - Use DocLayout-YOLO for speed
    - Use XY-Cut sorter for simple layouts
    - Enable caching for repeated processing
    - Test on few pages before full batch

!!! tip "Quality"
    - Use PP-DocLayoutV2 for complex layouts
    - Use PyMuPDF for multi-column documents
    - Use VLM detectors for mixed content
    - Increase DPI for small text

!!! tip "Cost Optimization"
    - Use Gemini free tier when possible
    - Cache recognition results
    - Use PaddleOCR-VL for large batches
    - Monitor rate limits

!!! warning "Common Issues"
    - Check page height for PyPDF conversions
    - Validate detector/sorter combinations
    - Handle rate limits gracefully
    - Clean up temporary files

## Next Steps

- [API Reference](../api/pipeline.md) - Detailed API documentation
- [Architecture Overview](../architecture/overview.md) - System design
- [Contributing Guide](contributing.md) - Contribute your own examples
