# Makefile for VLM OCR Pipeline

.PHONY: help test test-all test-types test-detectors test-sorters test-recognizers test-stages test-conversion test-cache test-factory test-validators lint format typecheck pre-commit clean

# Default target
help:
	@echo "VLM OCR Pipeline - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-all          - Run all tests with verbose output"
	@echo "  make test-types        - Test type system (BBox, Block, etc.)"
	@echo "  make test-detectors    - Test layout detectors"
	@echo "  make test-sorters      - Test reading order sorters"
	@echo "  make test-recognizers  - Test text recognizers"
	@echo "  make test-stages       - Test pipeline stages"
	@echo "  make test-conversion   - Test PDF/image conversion"
	@echo "  make test-cache        - Test recognition cache"
	@echo "  make test-factory      - Test factory patterns"
	@echo "  make test-validators   - Test validators"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              - Run ruff linter"
	@echo "  make format            - Format code with ruff"
	@echo "  make typecheck         - Run pyright type checker"
	@echo "  make pre-commit        - Run all pre-commit checks"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove cache and temp files"

# Run all tests
test:
	@echo "Running all tests..."
	uv run pytest tests/ -v

test-all:
	@echo "Running all tests with verbose output..."
	uv run pytest tests/ -vv

# Module-specific tests
test-types:
	@echo "Testing type system..."
	uv run pytest tests/test_types.py tests/test_adapters.py -v

test-detectors:
	@echo "Testing detectors..."
	uv run pytest tests/test_detectors.py -v

test-sorters:
	@echo "Testing sorters..."
	uv run pytest tests/test_sorters.py -v

test-recognizers:
	@echo "Testing recognizers..."
	uv run pytest tests/test_recognizers.py tests/test_recognition_cache.py tests/test_async_api.py -v

test-stages:
	@echo "Testing pipeline stages..."
	uv run pytest tests/test_stages.py -v

test-conversion:
	@echo "Testing document conversion..."
	uv run pytest tests/test_pdf_conversion.py tests/test_image_loading.py tests/test_markdown_conversion.py -v

test-cache:
	@echo "Testing recognition cache..."
	uv run pytest tests/test_recognition_cache.py -v

test-factory:
	@echo "Testing factory patterns..."
	uv run pytest tests/test_factory.py -v

test-validators:
	@echo "Testing validators..."
	uv run pytest tests/test_validator.py tests/test_protocol_validation.py tests/test_backend_auto_inference.py -v

# Code quality
lint:
	@echo "Running ruff linter..."
	uv run ruff check .

format:
	@echo "Formatting code..."
	uv run ruff format .

typecheck:
	@echo "Running pyright type checker..."
	npx pyright

pre-commit:
	@echo "Running pre-commit checks..."
	./scripts/pre-commit-check.sh

# Cleanup
clean:
	@echo "Cleaning cache and temp files..."
	rm -rf .cache .tmp .logs __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"
