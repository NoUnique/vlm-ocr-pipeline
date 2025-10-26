"""Tests for recognition cache module.

Tests the RecognitionCache class which handles:
- Image hashing for cache keys
- Reading and writing cached results
- Cache invalidation and error handling
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from pipeline.recognition.cache import RecognitionCache


class TestRecognitionCacheInit:
    """Tests for RecognitionCache initialization."""

    def test_init_creates_cache_dir(self, tmp_path: Path):
        """Test that initialization creates cache directory."""
        cache_dir = tmp_path / "cache"
        assert not cache_dir.exists()

        cache = RecognitionCache(cache_dir=cache_dir, use_cache=True)

        assert cache_dir.exists()
        assert cache.cache_dir == cache_dir
        assert cache.use_cache is True

    def test_init_with_existing_dir(self, tmp_path: Path):
        """Test initialization with existing directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache = RecognitionCache(cache_dir=cache_dir, use_cache=True)

        assert cache_dir.exists()
        assert cache.cache_dir == cache_dir

    def test_init_with_use_cache_false(self, tmp_path: Path):
        """Test initialization with use_cache=False."""
        cache_dir = tmp_path / "cache"
        cache = RecognitionCache(cache_dir=cache_dir, use_cache=False)

        assert cache.cache_dir == cache_dir
        assert cache.use_cache is False
        assert cache_dir.exists()  # Still creates directory

    def test_init_creates_nested_dirs(self, tmp_path: Path):
        """Test initialization creates nested directories."""
        cache_dir = tmp_path / "deep" / "nested" / "cache"
        assert not cache_dir.exists()

        cache = RecognitionCache(cache_dir=cache_dir, use_cache=True)

        assert cache_dir.exists()
        assert cache.cache_dir == cache_dir


class TestCalculateImageHash:
    """Tests for calculate_image_hash method."""

    def test_calculate_hash_for_simple_image(self, tmp_path: Path):
        """Test calculating hash for a simple image."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        image_hash = cache.calculate_image_hash(image)

        assert isinstance(image_hash, str)
        assert len(image_hash) == 32  # MD5 hash length
        assert image_hash.isalnum()  # Hex string

    def test_same_image_same_hash(self, tmp_path: Path):
        """Test that identical images produce identical hashes."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        hash1 = cache.calculate_image_hash(image)
        hash2 = cache.calculate_image_hash(image)

        assert hash1 == hash2

    def test_different_images_different_hashes(self, tmp_path: Path):
        """Test that different images produce different hashes."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        hash1 = cache.calculate_image_hash(image1)
        hash2 = cache.calculate_image_hash(image2)

        assert hash1 != hash2

    def test_hash_independent_of_image_size(self, tmp_path: Path):
        """Test that hash is independent of original image size."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        # Create same content but different sizes
        image1 = np.zeros((50, 50, 3), dtype=np.uint8)
        image2 = np.zeros((200, 200, 3), dtype=np.uint8)

        hash1 = cache.calculate_image_hash(image1)
        hash2 = cache.calculate_image_hash(image2)

        # Same content should produce same hash after resize
        assert hash1 == hash2

    @patch("cv2.imencode")
    def test_hash_encoding_failure(self, mock_imencode: Mock, tmp_path: Path):
        """Test that encoding failure raises ValueError."""
        mock_imencode.return_value = (False, None)
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Failed to encode image for hashing"):
            cache.calculate_image_hash(image)

    def test_hash_with_color_image(self, tmp_path: Path):
        """Test hashing with colored image."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        image_hash = cache.calculate_image_hash(image)

        assert isinstance(image_hash, str)
        assert len(image_hash) == 32


class TestGetCachedResult:
    """Tests for get_cached_result method."""

    def test_get_cached_result_hit(self, tmp_path: Path):
        """Test getting cached result when cache exists."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"
        expected_result = {"text": "Hello World", "confidence": 0.95}

        # Create cache file
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(expected_result, f)

        result = cache.get_cached_result(image_hash, cache_type)

        assert result == expected_result

    def test_get_cached_result_miss(self, tmp_path: Path):
        """Test getting cached result when cache doesn't exist."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "nonexistent"
        cache_type = "gemini_ocr"

        result = cache.get_cached_result(image_hash, cache_type)

        assert result is None

    def test_get_cached_result_with_use_cache_false(self, tmp_path: Path):
        """Test that get_cached_result returns None when use_cache=False."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=False)
        image_hash = "abc123"
        cache_type = "gemini_ocr"

        # Create cache file (should be ignored)
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump({"text": "Hello"}, f)

        result = cache.get_cached_result(image_hash, cache_type)

        assert result is None

    def test_get_cached_result_invalid_json(self, tmp_path: Path):
        """Test getting cached result with corrupted JSON file."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"

        # Create invalid JSON file
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        with cache_file.open("w", encoding="utf-8") as f:
            f.write("invalid json content {")

        result = cache.get_cached_result(image_hash, cache_type)

        assert result is None

    def test_get_cached_result_unicode_content(self, tmp_path: Path):
        """Test getting cached result with unicode content."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"
        expected_result = {"text": "안녕하세요 世界", "confidence": 0.95}

        # Create cache file with unicode
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(expected_result, f, ensure_ascii=False)

        result = cache.get_cached_result(image_hash, cache_type)

        assert result == expected_result

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_get_cached_result_read_error(self, mock_open: Mock, tmp_path: Path):
        """Test getting cached result when file read fails."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"

        # Create cache file
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        cache_file.touch()

        result = cache.get_cached_result(image_hash, cache_type)

        assert result is None


class TestSaveToCache:
    """Tests for save_to_cache method."""

    def test_save_to_cache_success(self, tmp_path: Path):
        """Test successfully saving result to cache."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"
        result = {"text": "Hello World", "confidence": 0.95}

        cache.save_to_cache(image_hash, cache_type, result)

        # Verify cache file exists
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        assert cache_file.exists()

        # Verify content
        with cache_file.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data == result

    def test_save_to_cache_with_use_cache_false(self, tmp_path: Path):
        """Test that save_to_cache does nothing when use_cache=False."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=False)
        image_hash = "abc123"
        cache_type = "gemini_ocr"
        result = {"text": "Hello World"}

        cache.save_to_cache(image_hash, cache_type, result)

        # Verify cache file was NOT created
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        assert not cache_file.exists()

    def test_save_to_cache_excludes_coords(self, tmp_path: Path):
        """Test that coords field is excluded from cache."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"
        result = {
            "text": "Hello World",
            "confidence": 0.95,
            "coords": [[0, 0], [100, 100]],
        }

        cache.save_to_cache(image_hash, cache_type, result)

        # Verify coords is excluded
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        with cache_file.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert "coords" not in saved_data
        assert saved_data == {"text": "Hello World", "confidence": 0.95}

    def test_save_to_cache_unicode_content(self, tmp_path: Path):
        """Test saving cache with unicode content."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"
        result = {"text": "안녕하세요 世界", "confidence": 0.95}

        cache.save_to_cache(image_hash, cache_type, result)

        # Verify unicode is preserved
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        with cache_file.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data["text"] == "안녕하세요 世界"

    def test_save_to_cache_overwrites_existing(self, tmp_path: Path):
        """Test that saving overwrites existing cache file."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"

        # Save first result
        result1 = {"text": "Old text"}
        cache.save_to_cache(image_hash, cache_type, result1)

        # Save second result (should overwrite)
        result2 = {"text": "New text"}
        cache.save_to_cache(image_hash, cache_type, result2)

        # Verify only new result exists
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        with cache_file.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data == result2

    def test_save_to_cache_with_indent(self, tmp_path: Path):
        """Test that saved JSON is formatted with indentation."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"
        result = {"text": "Hello", "confidence": 0.95}

        cache.save_to_cache(image_hash, cache_type, result)

        # Verify JSON is indented
        cache_file = tmp_path / f"{cache_type}_{image_hash}.json"
        content = cache_file.read_text(encoding="utf-8")

        assert "\n" in content  # Has newlines
        assert "  " in content  # Has indentation

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_save_to_cache_write_error(self, mock_open: Mock, tmp_path: Path):
        """Test that save_to_cache handles write errors gracefully."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image_hash = "abc123"
        cache_type = "gemini_ocr"
        result = {"text": "Hello"}

        # Should not raise exception
        cache.save_to_cache(image_hash, cache_type, result)


class TestCacheIntegration:
    """Integration tests for RecognitionCache."""

    def test_full_cache_workflow(self, tmp_path: Path):
        """Test complete workflow: hash -> save -> retrieve."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)

        # Create image and calculate hash
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_hash = cache.calculate_image_hash(image)

        # Save result
        cache_type = "gemini_ocr"
        result = {"text": "Detected text", "confidence": 0.98}
        cache.save_to_cache(image_hash, cache_type, result)

        # Retrieve result
        cached_result = cache.get_cached_result(image_hash, cache_type)

        assert cached_result == result

    def test_multiple_cache_types_for_same_image(self, tmp_path: Path):
        """Test storing multiple cache types for the same image."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_hash = cache.calculate_image_hash(image)

        # Save different cache types
        cache.save_to_cache(image_hash, "gemini_ocr", {"text": "OCR result"})
        cache.save_to_cache(image_hash, "table", {"table": "Table result"})

        # Retrieve both
        ocr_result = cache.get_cached_result(image_hash, "gemini_ocr")
        table_result = cache.get_cached_result(image_hash, "table")

        assert ocr_result == {"text": "OCR result"}
        assert table_result == {"table": "Table result"}

    def test_cache_persistence_across_instances(self, tmp_path: Path):
        """Test that cache persists across different cache instances."""
        # First instance saves
        cache1 = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_hash = cache1.calculate_image_hash(image)
        cache1.save_to_cache(image_hash, "gemini_ocr", {"text": "Cached"})

        # Second instance retrieves
        cache2 = RecognitionCache(cache_dir=tmp_path, use_cache=True)
        result = cache2.get_cached_result(image_hash, "gemini_ocr")

        assert result == {"text": "Cached"}

    def test_disabled_cache_workflow(self, tmp_path: Path):
        """Test that disabled cache doesn't save or retrieve."""
        cache = RecognitionCache(cache_dir=tmp_path, use_cache=False)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image_hash = cache.calculate_image_hash(image)

        # Try to save (should be ignored)
        cache.save_to_cache(image_hash, "gemini_ocr", {"text": "Test"})

        # Try to retrieve (should return None)
        result = cache.get_cached_result(image_hash, "gemini_ocr")

        assert result is None

        # Verify no cache files created
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 0
