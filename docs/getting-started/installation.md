# Installation

This guide will help you set up VLM OCR Pipeline on your system.

## Prerequisites

- **Python 3.11+**: Recommended for best compatibility
- **uv**: Fast Python package manager (optional but recommended)
- **Git**: For cloning the repository and managing submodules

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NoUnique/vlm-ocr-pipeline.git
cd vlm-ocr-pipeline
```

### 2. Set Up Python Environment

=== "Using uv (Recommended)"

    ```bash
    # Install uv if you haven't
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create virtual environment
    uv venv --python 3.11 .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install dependencies
    uv pip install -r requirements.txt
    ```

=== "Using pip"

    ```bash
    # Create virtual environment
    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

### 3. Fix DocLayout-YOLO Compatibility

```bash
python setup.py
```

!!! note "What does setup.py do?"
    The `setup.py` script fixes compatibility issues between DocLayout-YOLO and the ultralytics package by modifying the YOLO model files.

## API Configuration

Choose the backend you want to use and configure the corresponding API keys.

### Gemini API (Recommended for Free Tier)

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the generated key
5. Set environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Free Tier Limits** (as of 2024):
- 15 requests per minute (RPM)
- 1,500,000 tokens per minute (TPM)
- 1,500 requests per day (RPD)

### OpenAI API

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an API key
3. Set environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### OpenRouter API (Alternative)

OpenRouter provides access to multiple VLMs through a single API:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

## Optional Components

### PaddleOCR-VL (Local Recognition)

For local text recognition without API calls:

```bash
# PaddleX is already included as a submodule
cd external/PaddleX
git checkout v3.3.1
pip install -e .
```

**Requirements**:
- GPU with CUDA support (recommended)
- ~4GB VRAM for PaddleOCR-VL-0.9B

### External Frameworks (Git Submodules)

The project includes several external frameworks as submodules:

```bash
# Initialize all submodules
git submodule update --init --recursive

# Or initialize specific submodules
git submodule update --init external/MinerU      # MinerU detectors/sorters
git submodule update --init external/olmocr      # olmOCR VLM sorter
git submodule update --init external/PaddleOCR   # PP-DocLayoutV2 detector
git submodule update --init external/PaddleX     # PaddleOCR-VL recognizer
```

## Verification

Verify your installation:

```bash
# Check Python version
python --version  # Should be 3.11+

# Check dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Run a simple test
python main.py --help
```

## Troubleshooting

### CUDA/GPU Issues

If you encounter CUDA-related errors:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only PyTorch (if no GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Ultralytics/YOLO Errors

If you see errors about `ultralytics` or YOLO models:

```bash
# Re-run the setup script
python setup.py

# Or manually reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics==8.2.0
```

### Missing Dependencies

If you encounter import errors:

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

## Next Steps

Now that you have VLM OCR Pipeline installed:

- [Quick Start Guide](quickstart.md) - Run your first OCR pipeline
- [Basic Usage](basic-usage.md) - Learn the core features
- [Architecture Overview](../architecture/overview.md) - Understand how it works
