#!/usr/bin/env python3
"""
Setup script for the VLM OCR Pipeline project.
Handles DocLayout-YOLO compatibility fixes and environment setup.
"""

import sys
import site
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_doclayout_yolo_model_file() -> Path:
    """Find the DocLayout-YOLO model.py file that needs patching"""
    
    # Get all site-packages directories
    site_packages_dirs = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        site_packages_dirs.append(site.getusersitepackages())
    
    # Add virtual environment site-packages if exists
    venv_path = Path('.venv')
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
        logger.info(f"Found DocLayout-YOLO model file: {model_file}")
        
        # Read the current content
        content = model_file.read_text()
        
        # Check if already patched
        if "class YOLOv10(Model, PyTorchModelHubMixin):" in content:
            logger.info("DocLayout-YOLO already patched")
            return True
        
        # Apply the fix
        original_line = 'class YOLOv10(Model, PyTorchModelHubMixin, repo_url="https://github.com/opendatalab/DocLayout-YOLO", pipeline_tag="object-detection", license="agpl-3.0"):'
        fixed_line = 'class YOLOv10(Model, PyTorchModelHubMixin):'
        
        if original_line in content:
            new_content = content.replace(original_line, fixed_line)
            model_file.write_text(new_content)
            logger.info("DocLayout-YOLO compatibility fix applied successfully")
            return True
        else:
            logger.warning("Could not find the expected line to patch in DocLayout-YOLO")
            return False
            
    except Exception as e:
        logger.error(f"Failed to fix DocLayout-YOLO compatibility: {e}")
        return False


def verify_doclayout_yolo():
    """Verify that DocLayout-YOLO can be imported successfully"""
    try:
        import doclayout_yolo
        logger.info("DocLayout-YOLO import verification successful")
        return True
    except Exception as e:
        logger.error(f"DocLayout-YOLO import verification failed: {e}")
        return False


def setup_environment():
    """Setup the environment and fix compatibility issues"""
    logger.info("Starting environment setup...")
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        logger.warning(".env file not found. Creating template...")
        env_template = """# VLM Backend API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Custom OpenAI base URL (for OpenRouter or other compatible services)
# OPENAI_BASE_URL=https://openrouter.ai/api/v1

# Optional: Google Vision API (if using vision extraction method)
GOOGLE_APPLICATION_CREDENTIALS=.credentials/vision_service_account.json
"""
        env_file.write_text(env_template)
        logger.info("Created .env template file. Please update with your API keys.")
    
    # Fix DocLayout-YOLO compatibility
    logger.info("Fixing DocLayout-YOLO compatibility...")
    if fix_doclayout_yolo_compatibility():
        if verify_doclayout_yolo():
            logger.info("✅ DocLayout-YOLO setup completed successfully")
        else:
            logger.error("❌ DocLayout-YOLO verification failed")
            return False
    else:
        logger.error("❌ DocLayout-YOLO compatibility fix failed")
        return False
    
    logger.info("✅ Environment setup completed successfully")
    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1) 