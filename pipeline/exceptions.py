"""Custom exception classes for the VLM OCR Pipeline.

This module defines a hierarchy of custom exceptions to provide better
error handling and more specific error messages throughout the pipeline.

Exception Hierarchy:
    PipelineError (base)
    ├── ConfigurationError
    │   ├── InvalidConfigError
    │   └── MissingConfigError
    ├── APIError
    │   ├── APIClientError
    │   ├── APIAuthenticationError
    │   ├── APIRateLimitError
    │   └── APITimeoutError
    ├── ProcessingError
    │   ├── PageProcessingError
    │   ├── DetectionError
    │   ├── RecognitionError
    │   └── RenderingError
    ├── FileError
    │   ├── FileLoadError
    │   ├── FileSaveError
    │   └── FileFormatError
    └── DependencyError

Usage:
    try:
        # Some operation
        pass
    except APIRateLimitError as e:
        # Handle rate limit specifically
        logger.warning("Rate limit exceeded: %s", e)
    except APIError as e:
        # Handle all API errors
        logger.error("API error: %s", e)
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all pipeline errors.

    All custom exceptions in the pipeline should inherit from this class.
    This allows catching all pipeline-specific errors with a single handler.
    """


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigurationError(PipelineError):
    """Base exception for configuration-related errors.

    Raised when there are issues with configuration files, settings,
    or initialization parameters.
    """


class InvalidConfigError(ConfigurationError):
    """Raised when configuration values are invalid or malformed.

    Examples:
        - Invalid tier name
        - Malformed YAML/JSON
        - Invalid parameter values
    """


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing.

    Examples:
        - Missing API key
        - Required config file not found
        - Missing required parameters
    """


# ============================================================================
# API Errors
# ============================================================================


class APIError(PipelineError):
    """Base exception for API-related errors.

    Raised when interacting with external APIs (OpenAI, Gemini, etc.).
    """


class APIClientError(APIError):
    """Raised when API client initialization or setup fails.

    Examples:
        - Invalid API endpoint
        - Client library initialization failure
        - Network configuration issues
    """


class APIAuthenticationError(APIError):
    """Raised when API authentication fails.

    Examples:
        - Invalid API key
        - Expired credentials
        - Insufficient permissions
    """


class APIRateLimitError(APIError):
    """Raised when API rate limits are exceeded.

    Examples:
        - RPM (Requests Per Minute) limit reached
        - TPM (Tokens Per Minute) limit reached
        - RPD (Requests Per Day) limit reached
    """


class APITimeoutError(APIError):
    """Raised when API requests timeout.

    Examples:
        - Network timeout
        - Server not responding
        - Request took too long
    """


# ============================================================================
# Processing Errors
# ============================================================================


class ProcessingError(PipelineError):
    """Base exception for document processing errors.

    Raised when there are issues during document processing stages.
    """


class PageProcessingError(ProcessingError):
    """Raised when processing a specific page fails.

    Examples:
        - Page rendering failure
        - Invalid page number
        - Corrupted page data
    """


class DetectionError(ProcessingError):
    """Raised when layout detection fails.

    Examples:
        - Detector initialization failure
        - Detection model error
        - Invalid detection results
    """


class RecognitionError(ProcessingError):
    """Raised when text recognition fails.

    Examples:
        - Text extraction failure
        - OCR model error
        - Invalid recognition results
    """


class RenderingError(ProcessingError):
    """Raised when rendering blocks to output format fails.

    Examples:
        - Markdown conversion error
        - Template rendering failure
        - Invalid block structure
    """


# ============================================================================
# File Errors
# ============================================================================


class FileError(PipelineError):
    """Base exception for file operation errors.

    Raised when there are issues with file I/O operations.
    """


class FileLoadError(FileError):
    """Raised when loading a file fails.

    Examples:
        - File not found
        - Permission denied
        - Unsupported file format
        - Corrupted file
    """


class FileSaveError(FileError):
    """Raised when saving a file fails.

    Examples:
        - Permission denied
        - Disk full
        - Invalid path
        - Write operation failed
    """


class FileFormatError(FileError):
    """Raised when file format is invalid or unsupported.

    Examples:
        - Invalid PDF
        - Unsupported image format
        - Malformed JSON/YAML
        - Invalid file extension
    """


# ============================================================================
# Dependency Errors
# ============================================================================


class DependencyError(PipelineError):
    """Raised when required dependencies are missing or incompatible.

    Examples:
        - Missing optional dependency (PyMuPDF, MinerU, etc.)
        - Incompatible version
        - Import failure
        - Library not installed
    """
