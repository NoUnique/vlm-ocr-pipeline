#!/usr/bin/env python3
"""
Setup script for the VLM OCR Pipeline project.
Handles DocLayout-YOLO compatibility fixes and environment setup.
"""

import importlib
import logging
import site
import sys
import textwrap
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_doclayout_yolo_model_file() -> Path:
    """Find the DocLayout-YOLO model.py file that needs patching"""

    # Get all site-packages directories
    site_packages_dirs = site.getsitepackages()
    if hasattr(site, "getusersitepackages"):
        site_packages_dirs.append(site.getusersitepackages())

    # Add virtual environment site-packages if exists
    venv_path = Path(".venv")
    if venv_path.exists():
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site_packages = venv_path / "lib" / python_version / "site-packages"
        if venv_site_packages.exists():
            site_packages_dirs.insert(0, str(venv_site_packages))

    for site_dir in site_packages_dirs:
        model_file = Path(site_dir) / "doclayout_yolo" / "models" / "yolov10" / "model.py"
        if model_file.exists():
            return model_file

    raise FileNotFoundError("DocLayout-YOLO model.py file not found in any site-packages directory")


def fix_doclayout_yolo_compatibility():
    """Fix DocLayout-YOLO compatibility issues"""
    try:
        model_file = find_doclayout_yolo_model_file()
        logger.info("Found DocLayout-YOLO model file: %s", model_file)

        # Read the current content
        content = model_file.read_text()

        # Check if already patched
        if "class YOLOv10(Model, PyTorchModelHubMixin):" in content:
            logger.info("DocLayout-YOLO already patched")
            return True

        # Apply the fix
        original_line = (
            'class YOLOv10(Model, PyTorchModelHubMixin, repo_url="https://github.com/opendatalab/DocLayout-YOLO", '
            'pipeline_tag="object-detection", license="agpl-3.0"):'
        )
        fixed_line = "class YOLOv10(Model, PyTorchModelHubMixin):"

        if original_line in content:
            new_content = content.replace(original_line, fixed_line)
            model_file.write_text(new_content)
            logger.info("DocLayout-YOLO compatibility fix applied successfully")
            return True
        else:
            logger.warning("Could not find the expected line to patch in DocLayout-YOLO")
            return False

    except Exception as e:
        logger.error("Failed to fix DocLayout-YOLO compatibility: %s", e)
        return False


def fix_pytorch26_compatibility():
    """Fix PyTorch 2.6 weights_only compatibility issue.

    PyTorch 2.6 changed the default value of weights_only from False to True,
    which breaks loading of models with custom classes like YOLOv10DetectionModel.
    This patches doclayout_yolo to use weights_only=False.
    """
    try:
        # Find doclayout_yolo tasks.py file
        site_packages_dirs = site.getsitepackages()
        if hasattr(site, "getusersitepackages"):
            site_packages_dirs.append(site.getusersitepackages())

        # Add virtual environment site-packages if exists
        venv_path = Path(".venv")
        if venv_path.exists():
            python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            venv_site_packages = venv_path / "lib" / python_version / "site-packages"
            if venv_site_packages.exists():
                site_packages_dirs.insert(0, str(venv_site_packages))

        tasks_file = None
        for site_dir in site_packages_dirs:
            candidate = Path(site_dir) / "doclayout_yolo" / "nn" / "tasks.py"
            if candidate.exists():
                tasks_file = candidate
                break

        if not tasks_file:
            logger.warning("doclayout_yolo/nn/tasks.py not found - skipping PyTorch 2.6 patch")
            return True  # Not an error if not installed

        logger.info("Found doclayout_yolo tasks.py: %s", tasks_file)

        # Read content
        content = tasks_file.read_text()

        # Check if already patched
        if "weights_only=False" in content:
            logger.info("PyTorch 2.6 compatibility already patched")
            return True

        # Patch torch.load() calls to use weights_only=False
        # Line 753: ckpt = torch.load(file, map_location="cpu")
        original_line = 'ckpt = torch.load(file, map_location="cpu")'
        patched_line = 'ckpt = torch.load(file, map_location="cpu", weights_only=False)'

        if original_line in content:
            new_content = content.replace(original_line, patched_line)
            tasks_file.write_text(new_content)
            logger.info("PyTorch 2.6 compatibility patch applied successfully")
            return True
        else:
            logger.warning("Could not find expected torch.load() line to patch")
            return False

    except Exception as e:
        logger.error("Failed to apply PyTorch 2.6 compatibility patch: %s", e)
        return False


def verify_doclayout_yolo():
    """Verify that DocLayout-YOLO can be imported successfully"""
    try:
        importlib.import_module("doclayout_yolo")

        logger.info("DocLayout-YOLO import verification successful")
        return True
    except Exception as e:
        logger.error("DocLayout-YOLO import verification failed: %s", e)
        return False


def setup_environment():
    """Setup the environment and fix compatibility issues"""
    logger.info("Starting environment setup...")

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning(".env file not found. Creating template...")
        env_template = textwrap.dedent("""
            # VLM Backend API Keys
            GEMINI_API_KEY=your_gemini_api_key_here
            OPENAI_API_KEY=your_openai_api_key_here
            OPENROUTER_API_KEY=your_openrouter_api_key_here

            # Optional: Custom OpenAI base URL (for OpenRouter or other compatible services)
            # OPENAI_BASE_URL=https://openrouter.ai/api/v1
        """).strip()
        env_file.write_text(f"{env_template}\n")
        logger.info("Created .env template file. Please update with your API keys.")

    # Fix DocLayout-YOLO compatibility
    logger.info("Fixing DocLayout-YOLO compatibility...")
    if not fix_doclayout_yolo_compatibility():
        logger.error("❌ DocLayout-YOLO compatibility fix failed")
        return False

    # Fix PyTorch 2.6 compatibility
    logger.info("Fixing PyTorch 2.6 compatibility...")
    if not fix_pytorch26_compatibility():
        logger.error("❌ PyTorch 2.6 compatibility fix failed")
        return False

    # Verify DocLayout-YOLO
    if verify_doclayout_yolo():
        logger.info("✅ DocLayout-YOLO setup completed successfully")
    else:
        logger.error("❌ DocLayout-YOLO verification failed")
        return False

    logger.info("✅ Environment setup completed successfully")
    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
