"""
Unified VLM OCR Pipeline for document processing and text extraction.
Integrates layout detection, OCR, and Vision Language Model-powered text correction.
"""

import hashlib
import json
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Load environment variables if not already loaded
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import cv2
import numpy as np
from pdf2image import convert_from_path, pdfinfo_from_path

from models import DocLayoutYOLO
from .prompt import PromptManager
from .vision import VisionClient
from .gemini import GeminiClient
from .openai import OpenAIClient
from .ratelimit import rate_limiter

logger = logging.getLogger(__name__)


class Pipeline:
    """Unified VLM OCR processing pipeline with integrated text correction"""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.5,
        use_cache: bool = True,
        cache_dir: Union[str, Path] = ".cache",
        output_dir: Union[str, Path] = "output",
        temp_dir: Union[str, Path] = ".tmp",
        text_extraction_method: str = "gemini",
        prompts_dir: Union[str, Path] = "settings/prompts",
        backend: str = "openai",
        model: str = "gemini-2.5-flash",
        gemini_tier: str = "free"
    ):
        """
        Initialize VLM OCR processing pipeline
        
        Args:
            model_path: DocLayout-YOLO model path
            confidence_threshold: Detection confidence threshold
            use_cache: Whether to use caching
            cache_dir: Cache directory path
            output_dir: Output directory path
            temp_dir: Temporary files directory path
            text_extraction_method: Method for text extraction ("gemini" or "vision")
            prompts_dir: Directory containing prompt templates
            backend: Backend API to use ("openai" or "gemini")
            model: Model to use for text processing
            gemini_tier: Gemini API tier for rate limiting (only used with gemini backend)
        """
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
        self.text_extraction_method = text_extraction_method.lower()
        self.backend = backend.lower()
        self.model = model
        self.gemini_tier = gemini_tier
        
        # Initialize rate limiter (only for Gemini backend)
        if self.backend == "gemini":
            rate_limiter.set_tier_and_model(gemini_tier, model)
        
        # Convert paths to Path objects
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.prompts_dir = Path(prompts_dir)
        
        # Create directories
        self._setup_directories()
        
        # Initialize components
        self.prompt_manager = PromptManager(self.prompts_dir)
        self.doc_layout_model = self._setup_layout_model()
        self.vision_client = VisionClient()
        
        # Initialize backend clients
        if self.backend == "gemini":
            self.gemini_client = GeminiClient(gemini_model=model)
            self.ai_client = self.gemini_client
        else:  # OpenAI backend
            self.openai_client = OpenAIClient(model=model)
            self.ai_client = self.openai_client
            # Still initialize Gemini client for fallback if needed
            self.gemini_client = GeminiClient(gemini_model="gemini-2.5-flash")

    def _setup_directories(self) -> None:
        """Create necessary directories"""
        for directory in [self.cache_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_layout_model(self) -> DocLayoutYOLO:
        """Setup DocLayout-YOLO model"""
        model = DocLayoutYOLO(model_path=self.model_path)
        logger.info("DocLayout-YOLO model loaded successfully")
        return model

    def _calculate_image_hash(self, image: np.ndarray) -> str:
        """Calculate hash for image caching"""
        small_img = cv2.resize(image, (32, 32))
        _, buffer = cv2.imencode('.jpg', small_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        image_hash = hashlib.md5(buffer).hexdigest()
        del small_img, buffer
        return image_hash

    def _get_cached_result(self, image_hash: str, cache_type: str) -> Optional[Dict[str, Any]]:
        """Get cached result if exists"""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_type}_{image_hash}.json"
        
        if cache_file.exists():
            try:
                with cache_file.open('r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                logger.debug(f"Cache hit for {cache_type}: {image_hash}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        return None

    def _save_to_cache(self, image_hash: str, cache_type: str, result: Dict[str, Any]) -> None:
        """Save result to cache"""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_type}_{image_hash}.json"
        
        try:
            # Remove coords before caching (will be added back when retrieved)
            cache_data = {k: v for k, v in result.items() if k != 'coords'}
            
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Cached result for {cache_type}: {image_hash}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def _crop_region(self, image: np.ndarray, region: Dict[str, Any]) -> np.ndarray:
        """Crop region from image"""
        coords = region['coords']
        x, y, w, h = coords  # coords format is [x, y, width, height]
        
        # Convert to x1, y1, x2, y2
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        
        # Add small padding
        padding = 5
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        # Ensure valid dimensions
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid region coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return np.zeros((1, 1, 3), dtype=np.uint8)  # Return minimal image
        
        return image[y1:y2, x1:x2]

    def _extract_text_from_region(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from region using selected method with fallback"""
        
        # Use Gemini API by default or if specified
        if self.text_extraction_method == "gemini":
            logger.debug(f"Using {self.backend.upper()} API for text extraction")
            return self._extract_text_with_ai(region_img, region_info)
        
        # Use Vision API if specified
        elif self.text_extraction_method == "vision":
            logger.debug("Using Google Vision API for text extraction")
            result = self._extract_text_with_vision(region_img, region_info)
            
            # Fallback to AI client if Vision API fails
            if result.get('error') and self.ai_client.is_available():
                logger.warning(f"Vision API failed, falling back to {self.backend.upper()} API")
                return self._extract_text_with_ai(region_img, region_info)
            
            return result
        
        # Invalid method, default to AI client
        else:
            logger.warning(f"Invalid text extraction method: {self.text_extraction_method}, using {self.backend.upper()}")
            return self._extract_text_with_ai(region_img, region_info)

    def _extract_text_with_ai(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from region using the configured AI backend"""
        image_hash = self._calculate_image_hash(region_img)
        cache_type = f'{self.backend}_ocr'
        
        cached_result = self._get_cached_result(image_hash, cache_type)
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        # Get prompt from PromptManager
        prompt = self.prompt_manager.get_prompt('text_extraction', 'user')
        
        # Use AI client to extract text
        result = self.ai_client.extract_text(region_img, region_info, prompt)
        
        # Save to cache if successful
        if 'error' not in result:
            self._save_to_cache(image_hash, cache_type, result)
        
        return result

    def _extract_text_with_gemini(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from region using Gemini API"""
        image_hash = self._calculate_image_hash(region_img)
        
        cached_result = self._get_cached_result(image_hash, 'gemini_ocr')
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        # Get prompt from PromptManager
        prompt = self.prompt_manager.get_prompt('text_extraction', 'user')
        
        # Use GeminiClient to extract text
        result = self.gemini_client.extract_text(region_img, region_info, prompt)
        
        # Save to cache if successful
        if 'error' not in result:
            self._save_to_cache(image_hash, 'gemini_ocr', result)
        
        return result

    def _extract_text_with_vision(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from region using Google Vision API"""
        image_hash = self._calculate_image_hash(region_img)
        
        cached_result = self._get_cached_result(image_hash, 'vision_ocr')
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        # Use VisionClient to extract text
        result = self.vision_client.extract_text(region_img, region_info)
        
        # Save to cache if successful
        if 'error' not in result:
            self._save_to_cache(image_hash, 'vision_ocr', result)
        
        return result

    def _process_special_region_with_ai(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process special regions (tables, figures) with configured AI backend"""
        image_hash = self._calculate_image_hash(region_img)
        cache_type = f"{region_info['type']}_{self.backend}"
        
        cached_result = self._get_cached_result(image_hash, cache_type)
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        if not self.ai_client.is_available():
            logger.warning(f"{self.backend.upper()} API client not initialized, falling back to text extraction")
            return self._extract_text_from_region(region_img, region_info)
        
        # Get prompt from PromptManager
        if self.backend == "gemini":
            prompt = self.prompt_manager.get_gemini_prompt_for_region_type(region_info['type'])
        else:
            prompt = self.prompt_manager.get_prompt_for_region_type(region_info['type'])
        
        # Use AI client to process special region
        result = self.ai_client.process_special_region(region_img, region_info, prompt)
        
        # Save to cache if successful
        if 'error' not in result:
            self._save_to_cache(image_hash, cache_type, result)
        
        return result

    def _process_special_region_with_gemini(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process special regions (tables, figures) with Gemini API"""
        image_hash = self._calculate_image_hash(region_img)
        cache_type = f"{region_info['type']}_gemini"
        
        cached_result = self._get_cached_result(image_hash, cache_type)
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        if not self.gemini_client.is_available():
            logger.warning("Gemini API client not initialized, falling back to text extraction")
            return self._extract_text_from_region(region_img, region_info)
        
        # Get prompt from PromptManager
        prompt = self.prompt_manager.get_gemini_prompt_for_region_type(region_info['type'])
        
        # Use GeminiClient to process special region
        result = self.gemini_client.process_special_region(region_img, region_info, prompt)
        
        # Save to cache if successful
        if 'error' not in result:
            self._save_to_cache(image_hash, cache_type, result)
        
        return result

    def correct_text(self, text: str) -> str:
        """Correct OCR text using configured AI backend"""
        if not self.ai_client.is_available() or not text:
            return text
        
        # Get prompts from PromptManager
        system_prompt = self.prompt_manager.get_prompt('text_correction', 'system')
        user_prompt = self.prompt_manager.get_prompt('text_correction', 'user', text=text)
        
        # Use AI client to correct text
        result = self.ai_client.correct_text(text, system_prompt, user_prompt)
        
        # Handle different return types
        if isinstance(result, dict):
            return result.get("corrected_text", text)
        else:
            return str(result)

    def _process_regions(self, image_np: np.ndarray, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all detected regions"""
        processed_regions = []
        processed_coords = set()
        
        for region in regions:
            region_key = f"{region['coords'][0]}_{region['coords'][1]}_{region['coords'][2]}_{region['coords'][3]}"
            
            if region_key in processed_coords:
                continue
            
            region_img = self._crop_region(image_np, region)
            
            if region['type'] in ['table', 'figure']:
                processed_region = self._process_special_region_with_ai(region_img, region)
            else:
                processed_region = self._extract_text_from_region(region_img, region)
            
            processed_regions.append(processed_region)
            processed_coords.add(region_key)
            
            del region_img
            gc.collect()
        
        return processed_regions

    def process_image(self, image_path: Union[str, Path], max_pages: Optional[int] = None, page_range: Optional[Tuple[int, int]] = None, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process single image or PDF"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if image_path.suffix.lower() == '.pdf':
            return self.process_pdf(image_path, max_pages, page_range, pages)
        else:
            return self.process_single_image(image_path)

    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image file"""
        logger.info(f"Processing image: {image_path}")
        
        # Load image
        image_np = cv2.imread(str(image_path))
        if image_np is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect layout
        regions = self.doc_layout_model.predict(image_np, conf=self.confidence_threshold)
        
        # Process regions
        processed_regions = self._process_regions(image_np, regions)
        
        # Correct text for text regions
        for region in processed_regions:
            if region['type'] in ['plain text', 'title', 'list'] and 'text' in region:
                region['corrected_text'] = self.correct_text(region['text'])
        
        result = {
            'image_path': str(image_path),
            'regions': processed_regions,
            'processed_at': datetime.now().isoformat()
        }
        
        return result

    def process_pdf(self, pdf_path: Path, max_pages: Optional[int] = None, page_range: Optional[Tuple[int, int]] = None, pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process PDF file with page limiting options"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Get PDF info
        pdf_info = pdfinfo_from_path(pdf_path)
        total_pages = pdf_info['Pages']
        
        # Determine which pages to process
        pages_to_process = self._determine_pages_to_process(total_pages, max_pages, page_range, pages)
        
        logger.info(f"Processing {len(pages_to_process)} pages: {pages_to_process}")
        
        processed_pages = []
        processing_stopped = False
        
        for page_num in pages_to_process:
            logger.info(f"Processing page {page_num}/{total_pages}")
            
            try:
                # Convert PDF page to image
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=200)
                page_image = np.array(images[0])
                
                # Save temporary image
                temp_image_path = self.temp_dir / f"{pdf_path.stem}_page_{page_num}.jpg"
                cv2.imwrite(str(temp_image_path), cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR))
                logger.info(f"Processing image: {temp_image_path}")
                
                # Detect layout
                regions = self.doc_layout_model.predict(page_image, conf=self.confidence_threshold)
                
                # Process regions
                processed_regions = self._process_regions(page_image, regions)
                
                # Check for rate limit errors
                if self._check_for_rate_limit_errors({'regions': processed_regions}):
                    logger.warning(f"Rate limit detected on page {page_num}. Stopping processing.")
                    processing_stopped = True
                    break
                
                # Correct text for text regions
                for region in processed_regions:
                    if region['type'] in ['plain text', 'title', 'list'] and 'text' in region:
                        corrected_result = self.correct_text(region['text'])
                        region['corrected_text'] = corrected_result
                        
                        # Check for rate limit in text correction
                        if isinstance(corrected_result, str) and any(error in corrected_result for error in ['RATE_LIMIT_EXCEEDED', 'DAILY_LIMIT_EXCEEDED']):
                            logger.warning(f"Rate limit detected during text correction on page {page_num}. Stopping processing.")
                            processing_stopped = True
                            break
                
                if processing_stopped:
                    break
                
                page_result = {
                    'page_number': page_num,
                    'regions': processed_regions,
                    'processed_at': datetime.now().isoformat()
                }
                
                processed_pages.append(page_result)
                
                # Save individual page result
                page_output_dir = self.output_dir / pdf_path.stem
                page_output_dir.mkdir(parents=True, exist_ok=True)
                
                page_output_file = page_output_dir / f"page_{page_num}.json"
                with page_output_file.open('w', encoding='utf-8') as f:
                    json.dump(page_result, f, ensure_ascii=False, indent=2)
                logger.info(f"Results saved to {page_output_file}")
                
                del images, page_image
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                error_page_result = {
                    'page_number': page_num,
                    'error': str(e),
                    'processed_at': datetime.now().isoformat()
                }
                processed_pages.append(error_page_result)
        
        # Create summary
        summary = {
            'pdf_path': str(pdf_path),
            'total_pages': total_pages,
            'processed_pages': len(processed_pages),
            'pages_data': processed_pages,
            'processing_stopped': processing_stopped,
            'processed_at': datetime.now().isoformat()
        }
        
        # Save summary
        summary_output_dir = self.output_dir / pdf_path.stem
        summary_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename based on completion status
        if processing_stopped:
            summary_filename = 'summary_incomplete.json'
        elif self._check_for_any_errors(summary):
            summary_filename = 'summary_partial.json'
        else:
            summary_filename = 'summary.json'
        
        summary_output_file = summary_output_dir / summary_filename
        with summary_output_file.open('w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {summary_output_file}")
        
        logger.info(f"PDF processing complete: {pdf_path} -> {summary_output_dir}")
        
        return summary

    def _determine_pages_to_process(self, total_pages: int, max_pages: Optional[int] = None, page_range: Optional[Tuple[int, int]] = None, pages: Optional[List[int]] = None) -> List[int]:
        """Determine which pages to process based on limiting options"""
        if pages is not None:
            # Specific pages specified
            valid_pages = [p for p in pages if 1 <= p <= total_pages]
            if len(valid_pages) != len(pages):
                invalid_pages = [p for p in pages if p not in valid_pages]
                logger.warning(f"Invalid page numbers (outside 1-{total_pages}): {invalid_pages}")
            return sorted(valid_pages)
        
        elif page_range is not None:
            # Page range specified
            start, end = page_range
            start = max(1, start)
            end = min(total_pages, end)
            return list(range(start, end + 1))
        
        elif max_pages is not None:
            # Max pages specified
            return list(range(1, min(max_pages + 1, total_pages + 1)))
        
        else:
            # Process all pages
            return list(range(1, total_pages + 1))

    def _check_for_rate_limit_errors(self, page_result: Dict[str, Any]) -> bool:
        """Check if page result contains rate limit errors"""
        try:
            # Check in regions
            regions = page_result.get('regions', [])
            if isinstance(regions, list):
                for region in regions:
                    if isinstance(region, dict) and region.get('error') in ['gemini_rate_limit', 'rate_limit_daily']:
                        return True
            
            # Check in corrected_text (can be string or dict)
            corrected_text = page_result.get('corrected_text', '')
            if isinstance(corrected_text, dict) and corrected_text.get('error') in ['gemini_rate_limit', 'rate_limit_daily']:
                return True
            elif isinstance(corrected_text, str) and any(error_indicator in corrected_text for error_indicator in [
                'RATE_LIMIT_EXCEEDED', 'TEXT_CORRECTION_RATE_LIMIT_EXCEEDED', 'DAILY_LIMIT_EXCEEDED', 'TEXT_CORRECTION_DAILY_LIMIT_EXCEEDED'
            ]):
                return True
        except (AttributeError, TypeError) as e:
            logger.debug(f"Error checking rate limit errors: {e}")
        
        return False

    def _check_for_any_errors(self, summary: Dict[str, Any]) -> bool:
        """Check if summary contains any processing errors"""
        try:
            pages_data = summary.get('pages_data', [])
            for page_result in pages_data:
                # Check for page-level errors
                if page_result.get('error'):
                    return True
                
                # Check for region-level errors
                regions = page_result.get('regions', [])
                for region in regions:
                    if isinstance(region, dict) and region.get('error'):
                        return True
                
                # Check for text correction errors
                corrected_text = page_result.get('corrected_text', '')
                if isinstance(corrected_text, dict) and corrected_text.get('error'):
                    return True
                elif isinstance(corrected_text, str) and any(error_indicator in corrected_text for error_indicator in [
                    '[RATE_LIMIT_EXCEEDED]', '[TEXT_CORRECTION_RATE_LIMIT_EXCEEDED]', '[TEXT_CORRECTION_SERVICE_UNAVAILABLE]', 
                    '[TEXT_CORRECTION_FAILED]', '[GEMINI_EXTRACTION_FAILED]', '[VISION_API_FAILED]', '[VISION_API_NOT_INITIALIZED]',
                    '[DAILY_LIMIT_EXCEEDED]', '[TEXT_CORRECTION_DAILY_LIMIT_EXCEEDED]'
                ]):
                    return True
                    
                # Check for processing_stopped flag
                if summary.get('processing_stopped', False):
                    return True
                    
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug(f"Error checking for processing errors: {e}")
            
        return False

    def process_directory(self, input_dir: Path, output_dir: str, max_pages: Optional[int] = None, page_range: Optional[Tuple[int, int]] = None, specific_pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process all PDFs in a directory"""
        input_dir = Path(input_dir)
        output_base = Path(output_dir)
        
        if not input_dir.exists() or not input_dir.is_dir():
            return {"error": f"Directory not found: {input_dir}"}
        
        # Find all PDF files in directory
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            return {"error": f"No PDF files found in directory: {input_dir}"}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {}
        total_files = len(pdf_files)
        processed_files = 0
        
        for pdf_file in pdf_files:
            logger.info(f"Processing PDF {processed_files + 1}/{total_files}: {pdf_file.name}")
            
            try:
                # Set output directory for this PDF
                self.output_dir = output_base / pdf_file.stem
                
                # Process the PDF
                result = self.process_pdf(
                    pdf_file, 
                    max_pages=max_pages,
                    page_range=page_range,
                    pages=specific_pages
                )
                
                results[str(pdf_file)] = result
                processed_files += 1
                
                # Check for processing errors that should stop batch processing
                if result.get('processing_stopped', False):
                    logger.warning(f"Processing stopped for {pdf_file.name} due to rate limits. Continuing with next file.")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                results[str(pdf_file)] = {
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                }
        
        summary = {
            "input_directory": str(input_dir),
            "output_directory": str(output_base),
            "total_files": total_files,
            "processed_files": processed_files,
            "results": results,
            "processed_at": datetime.now().isoformat()
        }
        
        # Save directory summary
        summary_file = output_base / "directory_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with summary_file.open('w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Directory processing complete. Summary saved to: {summary_file}")
        
        return summary

    def _save_results(self, result: Dict[str, Any], output_path: Path) -> None:
        """Save processing results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_path}") 