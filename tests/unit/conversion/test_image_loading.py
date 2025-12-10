"""Tests for image loading utilities.

Tests the image loading functions which handle:
- Loading images in various formats (JPEG, PNG)
- Error handling for invalid files
- PNG alpha channel conversion
- Extension validation with warnings
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pipeline.conversion.input.image import load_image, load_jpeg, load_png


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_valid_image(self, tmp_path: Path):
        """Test loading a valid image file."""
        # Create a test image
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), test_image)

        # Load the image
        loaded_image = load_image(image_path)

        assert loaded_image is not None
        assert isinstance(loaded_image, np.ndarray)
        assert loaded_image.shape == (100, 100, 3)
        assert loaded_image.dtype == np.uint8

    def test_load_image_maintains_bgr_format(self, tmp_path: Path):
        """Test that loaded image maintains BGR format."""
        # Create a solid blue image (BGR: [255, 0, 0])
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[:, :, 0] = 255  # Blue channel
        image_path = tmp_path / "blue.jpg"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_image(image_path)

        # Verify blue channel is dominant
        assert loaded_image[25, 25, 0] > 200  # Blue channel should be high
        assert loaded_image[25, 25, 1] < 50  # Green channel should be low
        assert loaded_image[25, 25, 2] < 50  # Red channel should be low

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Test loading a non-existent file raises ValueError."""
        nonexistent_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(ValueError, match="Could not load image"):
            load_image(nonexistent_path)

    def test_load_corrupted_file(self, tmp_path: Path):
        """Test loading a corrupted file raises ValueError."""
        # Create a file with invalid image data
        corrupted_path = tmp_path / "corrupted.jpg"
        with corrupted_path.open("wb") as f:
            f.write(b"This is not a valid image file")

        with pytest.raises(ValueError, match="Could not load image"):
            load_image(corrupted_path)

    def test_load_different_image_sizes(self, tmp_path: Path):
        """Test loading images of different sizes."""
        sizes = [(50, 50, 3), (100, 200, 3), (300, 150, 3)]

        for size in sizes:
            test_image = np.random.randint(0, 256, size, dtype=np.uint8)
            image_path = tmp_path / f"test_{size[0]}x{size[1]}.png"
            cv2.imwrite(str(image_path), test_image)

            loaded_image = load_image(image_path)

            assert loaded_image.shape == size

    def test_load_png_image(self, tmp_path: Path):
        """Test loading a PNG image through load_image."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.png"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_image(image_path)

        assert loaded_image is not None
        assert loaded_image.shape == (100, 100, 3)


class TestLoadJPEG:
    """Tests for load_jpeg function."""

    def test_load_jpeg_with_jpg_extension(self, tmp_path: Path):
        """Test loading JPEG with .jpg extension."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_jpeg(image_path)

        assert loaded_image is not None
        assert loaded_image.shape == (100, 100, 3)

    def test_load_jpeg_with_jpeg_extension(self, tmp_path: Path):
        """Test loading JPEG with .jpeg extension."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpeg"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_jpeg(image_path)

        assert loaded_image is not None
        assert loaded_image.shape == (100, 100, 3)

    def test_load_jpeg_case_insensitive(self, tmp_path: Path):
        """Test that JPEG loading is case insensitive."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.JPG"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_jpeg(image_path)

        assert loaded_image is not None
        assert loaded_image.shape == (100, 100, 3)

    def test_load_jpeg_wrong_extension_warns(self, tmp_path: Path, caplog):
        """Test that loading with wrong extension logs warning."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # Save as JPEG but with .png extension
        image_path = tmp_path / "test.png"
        cv2.imwrite(str(image_path), test_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        with caplog.at_level("WARNING"):
            loaded_image = load_jpeg(image_path)

        assert "not JPEG" in caplog.text
        assert loaded_image is not None

    def test_load_jpeg_nonexistent_file(self, tmp_path: Path):
        """Test loading non-existent JPEG raises ValueError."""
        nonexistent_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(ValueError, match="Could not load image"):
            load_jpeg(nonexistent_path)

    def test_load_jpeg_different_quality(self, tmp_path: Path):
        """Test loading JPEG images with different quality settings."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        for quality in [50, 75, 95]:
            image_path = tmp_path / f"test_q{quality}.jpg"
            cv2.imwrite(str(image_path), test_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

            loaded_image = load_jpeg(image_path)

            assert loaded_image is not None
            assert loaded_image.shape == (100, 100, 3)


class TestLoadPNG:
    """Tests for load_png function."""

    def test_load_png_rgb(self, tmp_path: Path):
        """Test loading RGB PNG image."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.png"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_png(image_path)

        assert loaded_image is not None
        assert loaded_image.shape == (100, 100, 3)
        assert loaded_image.dtype == np.uint8

    def test_load_png_with_alpha_channel(self, tmp_path: Path):
        """Test loading PNG with alpha channel converts to BGR."""
        # Create RGBA image
        test_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        test_image[:, :, 3] = 128  # Set alpha channel
        image_path = tmp_path / "test_alpha.png"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_png(image_path)

        # Should be converted to BGR (3 channels)
        assert loaded_image is not None
        assert loaded_image.shape == (100, 100, 3)
        assert loaded_image.ndim == 3

    def test_load_png_alpha_conversion_logged(self, tmp_path: Path, caplog):
        """Test that alpha channel conversion is logged."""
        # Create RGBA image
        test_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        image_path = tmp_path / "test_alpha.png"
        cv2.imwrite(str(image_path), test_image)

        with caplog.at_level("INFO"):
            load_png(image_path)

        assert "alpha channel" in caplog.text.lower()

    def test_load_png_wrong_extension_warns(self, tmp_path: Path, caplog):
        """Test that loading with wrong extension logs warning."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        # Save as PNG but with .jpg extension
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), test_image)

        with caplog.at_level("WARNING"):
            loaded_image = load_png(image_path)

        assert "not PNG" in caplog.text
        assert loaded_image is not None

    def test_load_png_nonexistent_file(self, tmp_path: Path):
        """Test loading non-existent PNG raises ValueError."""
        nonexistent_path = tmp_path / "nonexistent.png"

        with pytest.raises(ValueError, match="Could not load PNG image"):
            load_png(nonexistent_path)

    def test_load_png_case_insensitive(self, tmp_path: Path):
        """Test that PNG loading is case insensitive."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.PNG"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_png(image_path)

        assert loaded_image is not None
        assert loaded_image.shape == (100, 100, 3)

    def test_load_png_grayscale(self, tmp_path: Path):
        """Test loading grayscale PNG image."""
        # Create grayscale image
        test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        image_path = tmp_path / "test_gray.png"
        cv2.imwrite(str(image_path), test_image)

        loaded_image = load_png(image_path)

        assert loaded_image is not None
        # Grayscale image loaded as 2D
        assert loaded_image.ndim in [2, 3]

    def test_load_png_different_compression(self, tmp_path: Path):
        """Test loading PNG images with different compression levels."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        for compression in [0, 5, 9]:
            image_path = tmp_path / f"test_c{compression}.png"
            cv2.imwrite(str(image_path), test_image, [cv2.IMWRITE_PNG_COMPRESSION, compression])

            loaded_image = load_png(image_path)

            assert loaded_image is not None
            assert loaded_image.shape == (100, 100, 3)


class TestImageLoadingIntegration:
    """Integration tests for image loading functions."""

    def test_load_image_formats_consistency(self, tmp_path: Path):
        """Test that all loading functions return consistent format."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Save and load as JPEG
        jpeg_path = tmp_path / "test.jpg"
        cv2.imwrite(str(jpeg_path), test_image)
        jpeg_loaded = load_jpeg(jpeg_path)

        # Save and load as PNG
        png_path = tmp_path / "test.png"
        cv2.imwrite(str(png_path), test_image)
        png_loaded = load_png(png_path)

        # Both should have same shape and dtype
        assert jpeg_loaded.shape == png_loaded.shape
        assert jpeg_loaded.dtype == png_loaded.dtype

    def test_load_functions_use_load_image(self, tmp_path: Path):
        """Test that load_jpeg uses load_image internally."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), test_image)

        with patch("pipeline.conversion.input.image.load_image") as mock_load:
            mock_load.return_value = test_image
            result = load_jpeg(image_path)

            mock_load.assert_called_once_with(image_path)
            assert np.array_equal(result, test_image)

    def test_multiple_format_loading(self, tmp_path: Path):
        """Test loading multiple image formats sequentially."""
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Create various format files
        formats = [
            ("test.jpg", load_jpeg),
            ("test.jpeg", load_jpeg),
            ("test.png", load_png),
            ("test.bmp", load_image),
        ]

        loaded_images = []
        for filename, load_func in formats:
            file_path = tmp_path / filename
            cv2.imwrite(str(file_path), test_image)
            loaded_images.append(load_func(file_path))

        # All should have loaded successfully
        assert all(image is not None for image in loaded_images)
        assert all(image.shape == (100, 100, 3) for image in loaded_images)

    def test_load_after_modify(self, tmp_path: Path):
        """Test loading image after modifying it on disk."""
        # Create initial image
        test_image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), test_image1)

        loaded1 = load_image(image_path)
        assert loaded1[50, 50, 0] < 10  # Should be mostly black

        # Modify image
        test_image2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(image_path), test_image2)

        loaded2 = load_image(image_path)
        assert loaded2[50, 50, 0] > 200  # Should be mostly white
