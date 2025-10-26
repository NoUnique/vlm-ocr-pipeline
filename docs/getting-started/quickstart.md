# Quick Start

Get up and running with VLM OCR Pipeline in 5 minutes!

## Prerequisites

- Python 3.11+ installed
- Gemini API key (free tier available)

## Step 1: Install

```bash
# Clone the repository
git clone https://github.com/NoUnique/vlm-ocr-pipeline.git
cd vlm-ocr-pipeline

# Set up environment
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Fix YOLO compatibility
python setup.py
```

## Step 2: Configure API

```bash
# Set Gemini API key
export GEMINI_API_KEY="your_api_key_here"
```

!!! tip "Get a Free Gemini API Key"
    Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to get a free API key.

## Step 3: Run Your First Pipeline

```bash
# Process a single PDF
python main.py --input document.pdf --backend gemini
```

That's it! The pipeline will:

1. 📄 Load your PDF and render each page as an image
2. 🔍 Detect layout blocks (text, tables, figures, etc.)
3. 📊 Analyze reading order
4. 📝 Extract text using Gemini Vision
5. 🔧 Correct and improve text quality
6. 💾 Save results to `output/gemini-2.5-flash/document/`

## Understanding the Output

After processing, you'll find:

```
output/
└── gemini-2.5-flash/
    └── document/
        ├── page_1.json          # Detailed page data
        ├── page_1.md            # Markdown output
        └── document_summary.json  # Processing metadata
```

### Example Output

**page_1.md**:
```markdown
# Introduction

This document describes...

## Table of Contents

1. Getting Started
2. Advanced Features
3. API Reference
```

**page_1.json**:
```json
{
  "page_num": 1,
  "text": "# Introduction\n\nThis document describes...",
  "corrected_text": "# Introduction\n\nThis document describes...",
  "correction_ratio": 0.05,
  "blocks": [
    {
      "type": "title",
      "bbox": [100, 50, 500, 120],
      "text": "Introduction",
      "order": 0
    }
  ]
}
```

## Common Use Cases

### Single Image

```bash
python main.py --input photo.jpg --backend gemini
```

### Batch Processing

```bash
# Process all PDFs in a directory
python main.py --input documents/ --backend gemini
```

### Limit Pages (for testing)

```bash
# Process only first 5 pages
python main.py --input document.pdf --backend gemini --max-pages 5

# Process specific page range
python main.py --input document.pdf --backend gemini --page-range 10-20

# Process specific pages
python main.py --input document.pdf --backend gemini --pages 1,5,10
```

### Use OpenAI Instead

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# Run with OpenAI backend
python main.py --input document.pdf --backend openai --model gpt-4o
```

### Local Processing (No API)

```bash
# Use PaddleOCR-VL (local model, no API calls)
python main.py --input document.pdf \
    --detector paddleocr-doclayout-v2 \
    --recognizer paddleocr-vl
```

## Rate Limiting

The pipeline automatically handles rate limits for Gemini API:

```bash
# Check current rate limit status
python main.py --rate-limit-status --backend gemini --gemini-tier free
```

**Free Tier Limits**:
- 15 requests per minute
- 1,500,000 tokens per minute
- 1,500 requests per day

The pipeline will automatically wait when limits are reached.

## Troubleshooting

### "GEMINI_API_KEY not set"

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### "Rate limit exceeded"

The pipeline will automatically wait. Alternatively:

```bash
# Use OpenAI instead
python main.py --input doc.pdf --backend openai

# Or use local model (no API)
python main.py --input doc.pdf --recognizer paddleocr-vl
```

### "CUDA out of memory"

If using PaddleOCR-VL locally:

```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

## Next Steps

Now that you've run your first pipeline:

- [Basic Usage Guide](basic-usage.md) - Learn about all available options
- [Architecture Overview](../architecture/overview.md) - Understand how the pipeline works
- [Advanced Examples](../guides/advanced-examples.md) - Complex use cases and customizations

## Tips for Best Results

!!! tip "Optimize Processing"
    - Use `--max-pages` to test on a few pages first
    - Check rate limit status regularly for Gemini
    - Use caching to avoid reprocessing identical content

!!! warning "API Costs"
    - Gemini free tier has daily limits
    - OpenAI charges per token
    - Consider using PaddleOCR-VL for large batch processing

!!! info "Performance"
    - DocLayout-YOLO is fastest for detection
    - PaddleOCR-VL provides good quality without API costs
    - Gemini 2.5 Flash is fast and cost-effective
