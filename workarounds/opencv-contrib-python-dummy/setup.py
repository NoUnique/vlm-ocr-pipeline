"""Dummy opencv-contrib-python package to satisfy PaddleX dependency check.

PaddleX hardcodes a check for 'opencv-contrib-python' package name,
but we use 'opencv-python-headless' to avoid GUI dependencies (libGL.so.1).

This dummy package provides the metadata that PaddleX looks for,
while the actual cv2 module comes from opencv-python-headless.

Installation:
    uv pip install -e workarounds/opencv-contrib-python-dummy

How it works:
    1. PaddleX checks: importlib.metadata.version("opencv-contrib-python")
    2. This dummy package satisfies that check
    3. Actual cv2 import uses opencv-python-headless

Why needed:
    - PaddleX requires opencv-contrib-python for PDFReaderBackend
    - opencv-contrib-python has libGL.so.1 dependency (GUI libraries)
    - Server environments don't have GUI libraries installed
    - opencv-python-headless works fine but has different package name
"""

from setuptools import setup

setup(
    name="opencv-contrib-python",
    version="4.7.0.72",  # Match opencv-python-headless version
    description="Dummy package to satisfy PaddleX opencv-contrib-python dependency",
    long_description=__doc__,
    author="VLM OCR Pipeline",
    py_modules=[],  # No actual Python modules
    install_requires=[],  # Don't install anything
)
