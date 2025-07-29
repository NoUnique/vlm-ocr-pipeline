"""
Google Cloud Vision API client for OCR text extraction.
Handles authentication, image processing, and text detection.
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from google.cloud import vision
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)


class VisionClient:
    """Google Cloud Vision API client for text extraction"""
    
    def __init__(self, credentials_path: Optional[Path] = None):
        """
        Initialize Vision API client
        
        Args:
            credentials_path: Path to service account JSON file
        """
        self.credentials_path = credentials_path or Path.cwd() / ".credentials" / "vision_service_account.json"
        self.client = self._setup_vision_api()
    
    def _setup_vision_api(self) -> Optional[vision.ImageAnnotatorClient]:
        """Setup Google Vision API client"""
        try:
            if not self.credentials_path.exists():
                logger.warning(f"Vision API credentials not found at {self.credentials_path}")
                return None
            
            creds = Credentials.from_service_account_file(str(self.credentials_path))
            client = vision.ImageAnnotatorClient(credentials=creds)
            logger.info("Google Vision API client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Google Vision API client: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Vision API client is available"""
        return self.client is not None
    
    def extract_text(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text from region using Google Vision API
        
        Args:
            region_img: Image region as numpy array
            region_info: Region metadata including type and coordinates
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not self.is_available():
            logger.warning("Vision API client not initialized")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': '[VISION_API_NOT_INITIALIZED]',
                'confidence': 0.0,
                'error': 'vision_not_initialized',
                'error_message': 'Vision API client not initialized'
            }
        
        try:
            image_bytes = self._optimize_image_for_api(region_img)
            
            image = vision.Image(content=image_bytes)
            context = vision.ImageContext(language_hints=['en', 'ko', 'ja'])
            response = self.client.text_detection(image=image, image_context=context)
            
            text = ''
            if response.text_annotations:
                text = response.text_annotations[0].description
            
            result = {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': text,
                'confidence': region_info.get('confidence', 1.0)
            }
            
            del image_bytes, response
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Vision API text extraction error: {e}")
            return {
                'type': region_info['type'],
                'coords': region_info['coords'],
                'text': '[VISION_API_FAILED]',
                'confidence': 0.0,
                'error': 'vision_api_error',
                'error_message': str(e)
            }
    
    def _optimize_image_for_api(self, image: np.ndarray) -> bytes:
        """
        Optimize image for API transmission
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Optimized image as bytes
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize if too large (Vision API has size limits)
        max_dimension = 2048
        h, w = image_rgb.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Encode as JPEG with high quality
        _, buffer = cv2.imencode('.jpg', image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
    
    def reload_client(self) -> bool:
        """
        Reload the Vision API client (useful after credential updates)
        
        Returns:
            True if client was successfully reloaded
        """
        self.client = self._setup_vision_api()
        return self.is_available() 