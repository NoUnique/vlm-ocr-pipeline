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
            backend: Backend API to use ("openai" or "gemini")
            model: Model to use for text processing
            gemini_tier: Gemini API tier for rate limiting (only used with gemini backend)
        """
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.use_cache = use_cache
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
        
        # Create directories
        self._setup_directories()
        
        # Initialize components
        self.prompt_manager = PromptManager(model=self.model, backend=self.backend)
        self.doc_layout_model = self._setup_layout_model()
        # Initialize backend clients
        if self.backend == "gemini":
            self.gemini_client = GeminiClient(gemini_model=model)
            self.ai_client = self.gemini_client
        else:  # OpenAI backend
            self.openai_client = OpenAIClient(model=model)
            self.ai_client = self.openai_client
            # Still initialize Gemini client for fallback if needed
            self.gemini_client = GeminiClient(gemini_model="gemini-2.5-flash")

        logger.info("AI backend initialized: %s (model=%s)", self.backend.upper(), self.model)

    def _setup_directories(self) -> None:
        """Create necessary directories"""
        for directory in [self.cache_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_pdf_output_dir(self, pdf_path: Path) -> Path:
        """Return the output directory for a given PDF as <output>/<model>/<file_stem>"""
        return self.output_dir / self.model / pdf_path.stem

    def _setup_layout_model(self) -> DocLayoutYOLO:
        """Setup DocLayout-YOLO model"""
        model = DocLayoutYOLO(model_path=self.model_path)
        logger.info("DocLayout-YOLO model loaded successfully")
        return model

    def _compose_page_raw_text(self, processed_regions: List[Dict[str, Any]]) -> str:
        """Compose page-level raw text from processed regions in reading order.

        Reading order: top-to-bottom (y), then left-to-right (x). Includes text-like
        regions only and preserves internal newlines within each region's text.
        """
        if not isinstance(processed_regions, list):
            return ""
        text_like_types = {"plain text", "title", "list"}
        sortable_regions: List[Tuple[int, int, str]] = []
        for region in processed_regions:
            if not isinstance(region, dict):
                continue
            region_type = region.get("type")
            if region_type not in text_like_types:
                continue
            coords = region.get("coords") or [0, 0, 0, 0]
            try:
                x, y = int(coords[0]), int(coords[1])
            except Exception:
                x, y = 0, 0
            text_value = region.get("text")
            if isinstance(text_value, str) and text_value.strip():
                # Keep internal newlines; trim outer whitespace only
                sortable_regions.append((y, x, text_value.strip()))
        # Sort by y then x
        sortable_regions.sort(key=lambda t: (t[0], t[1]))
        # Join with a blank line between regions to separate blocks
        return "\n\n".join(t[2] for t in sortable_regions)

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
                logger.debug("Cache hit for %s: %s", cache_type, image_hash)
                return cached_data
            except Exception as e:
                logger.warning("Failed to load cache file %s: %s", cache_file, e)
        
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
            logger.debug("Cached result for %s: %s", cache_type, image_hash)
        except Exception as e:
            logger.warning("Failed to save cache file %s: %s", cache_file, e)

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
            logger.warning("Invalid region coordinates: x1=%s, y1=%s, x2=%s, y2=%s", x1, y1, x2, y2)
            return np.zeros((1, 1, 3), dtype=np.uint8)  # Return minimal image
        
        return image[y1:y2, x1:x2]

    def _extract_text_from_region(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from region using the configured AI backend"""
        logger.debug("Using %s API for text extraction", self.backend.upper())
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

    def _process_special_region_with_ai(self, region_img: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process special regions (tables, figures) with configured AI backend"""
        image_hash = self._calculate_image_hash(region_img)
        cache_type = f"{region_info['type']}_{self.backend}"
        
        cached_result = self._get_cached_result(image_hash, cache_type)
        if cached_result is not None:
            cached_result['coords'] = region_info['coords']
            return cached_result
        
        if not self.ai_client.is_available():
            logger.warning("%s API client not initialized, falling back to text extraction", self.backend.upper())
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
        logger.info("Processing image: %s", image_path)
        
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
        logger.info("Processing PDF: %s", pdf_path)
        
        # Get PDF info
        pdf_info = pdfinfo_from_path(pdf_path)
        total_pages = pdf_info['Pages']
        
        # Determine which pages to process
        pages_to_process = self._determine_pages_to_process(total_pages, max_pages, page_range, pages)
        
        logger.info("Processing %d pages: %s", len(pages_to_process), pages_to_process)
        
        processed_pages = []
        processing_stopped = False
        
        for page_num in pages_to_process:
            logger.info("Processing page %d/%d", page_num, total_pages)
            
            try:
                # Convert PDF page to image
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=200)
                page_image = np.array(images[0])
                
                # Save temporary image
                temp_image_path = self.temp_dir / f"{pdf_path.stem}_page_{page_num}.jpg"
                cv2.imwrite(str(temp_image_path), cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR))
                logger.info("Processing image: %s", temp_image_path)
                
                # Detect layout
                regions = self.doc_layout_model.predict(page_image, conf=self.confidence_threshold)
                
                # Process regions
                processed_regions = self._process_regions(page_image, regions)
                
                # Check for rate limit errors
                if self._check_for_rate_limit_errors({'regions': processed_regions}):
                    logger.warning("Rate limit detected on page %d. Stopping processing.", page_num)
                    processing_stopped = True
                    break
                
                # Page-level text aggregation and single correction
                raw_text = self._compose_page_raw_text(processed_regions)
                
                correction_result = self.correct_text(raw_text)
                if isinstance(correction_result, dict):
                    corrected_text = correction_result.get('corrected_text', raw_text)
                    correction_confidence = float(correction_result.get('confidence', 1.0))
                else:
                    corrected_text = str(correction_result)
                    correction_confidence = 1.0
                
                # Check for rate limit signals in page-level correction
                if isinstance(correction_result, str) and any(error in correction_result for error in ['RATE_LIMIT_EXCEEDED', 'DAILY_LIMIT_EXCEEDED']):
                    logger.warning("Rate limit detected during page text correction on page %d. Stopping processing.", page_num)
                    processing_stopped = True
                    break
                 
                # Compose legacy-style page result
                page_height, page_width = page_image.shape[0], page_image.shape[1]
                page_result = {
                    'image_path': str(self.temp_dir / f"{pdf_path.stem}_page_{page_num}.jpg"),
                    'width': int(page_width),
                    'height': int(page_height),
                    'regions': regions,  # detection results
                    'processed_regions': processed_regions,  # extracted/processed content
                    'raw_text': raw_text,
                    'corrected_text': corrected_text,
                    'correction_confidence': correction_confidence,
                    'processed_at': datetime.now().isoformat(),
                    'page_number': page_num
                }
                 
                processed_pages.append(page_result)
                 
                # Save individual page result
                page_output_dir = self._get_pdf_output_dir(pdf_path)
                page_output_dir.mkdir(parents=True, exist_ok=True)
                 
                page_output_file = page_output_dir / f"page_{page_num}.json"
                with page_output_file.open('w', encoding='utf-8') as f:
                    json.dump(page_result, f, ensure_ascii=False, indent=2)
                logger.info("Results saved to %s", page_output_file)
                 
                del images, page_image
                gc.collect()
                
            except Exception as e:
                logger.error("Error processing page %d: %s", page_num, e)
                error_page_result = {
                    'page_number': page_num,
                    'error': str(e),
                    'processed_at': datetime.now().isoformat()
                }
                processed_pages.append(error_page_result)
        
        # Prepare output directory for summary
        summary_output_dir = self._get_pdf_output_dir(pdf_path)
        summary_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine page statuses for legacy-style summary
        pages_summary = []
        status_counts = {"complete": 0, "partial": 0, "incomplete": 0}
        for pr in processed_pages:
            page_no = int(pr.get('page_number', 0))
            if pr.get('error'):
                status = 'partial'
            else:
                status = 'complete'
            status_counts[status] += 1
            pages_summary.append({
                'page': page_no,
                'status': status,
                'file_suffix': '' if status == 'complete' else status
            })

        # Temp summary to reuse existing error check logic for filename decision
        temp_summary_for_errors = {
            'pages_data': processed_pages,
            'processing_stopped': processing_stopped
        }
        has_errors = self._check_for_any_errors(temp_summary_for_errors)

        # Create legacy-style summary
        summary = {
            'pdf_name': pdf_path.stem,
            'pdf_path': str(pdf_path),
            'num_pages': total_pages,
            'processed_pages': len(processed_pages),
            'output_directory': str(summary_output_dir),
            'processed_at': datetime.now().isoformat(),
            'status_summary': {k: v for k, v in status_counts.items() if v > 0},
            'pages': pages_summary
        }
         
        # Save summary
        # Determine output filename based on completion status
        if processing_stopped:
            summary_filename = 'summary_incomplete.json'
        elif has_errors:
            summary_filename = 'summary_partial.json'
        else:
            summary_filename = 'summary.json'
        
        summary_output_file = summary_output_dir / summary_filename
        with summary_output_file.open('w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", summary_output_file)

        logger.info("PDF processing complete: %s -> %s", pdf_path, summary_output_dir)
        
        return summary

    def _determine_pages_to_process(self, total_pages: int, max_pages: Optional[int] = None, page_range: Optional[Tuple[int, int]] = None, pages: Optional[List[int]] = None) -> List[int]:
        """Determine which pages to process based on limiting options"""
        if pages is not None:
            # Specific pages specified
            valid_pages = [p for p in pages if 1 <= p <= total_pages]
            if len(valid_pages) != len(pages):
                invalid_pages = [p for p in pages if p not in valid_pages]
                logger.warning("Invalid page numbers (outside 1-%d): %s", total_pages, invalid_pages)
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
            logger.debug("Error checking rate limit errors: %s", e)
        
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
            logger.debug("Error checking for processing errors: %s", e)
            
        return False

    def process_directory(self, input_dir: Path, output_dir: str, max_pages: Optional[int] = None, page_range: Optional[Tuple[int, int]] = None, specific_pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """Process all PDFs in a directory"""
        input_dir = Path(input_dir)
        output_base = Path(output_dir)
        model_base_dir = output_base / self.model
        
        if not input_dir.exists() or not input_dir.is_dir():
            return {"error": f"Directory not found: {input_dir}"}
        
        # Find all PDF files in directory
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            return {"error": f"No PDF files found in directory: {input_dir}"}
        
        logger.info("Found %d PDF files to process", len(pdf_files))
        
        results = {}
        total_files = len(pdf_files)
        processed_files = 0
        
        for pdf_file in pdf_files:
            logger.info("Processing PDF %d/%d: %s", processed_files + 1, total_files, pdf_file.name)
            
            try:
                # Process the PDF (outputs will be placed under <output>/<model>/<file>)
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
                    logger.warning("Processing stopped for %s due to rate limits. Continuing with next file.", pdf_file.name)
                
            except Exception as e:
                logger.error("Error processing %s: %s", pdf_file, e)
                results[str(pdf_file)] = {
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                }
        
        # Ensure model base directory exists
        model_base_dir.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "input_directory": str(input_dir),
            "output_directory": str(model_base_dir),
            "total_files": total_files,
            "processed_files": processed_files,
            "results": results,
            "processed_at": datetime.now().isoformat()
        }
        
        # Save directory summary under model-specific directory
        summary_file = model_base_dir / "directory_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        with summary_file.open('w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info("Directory processing complete. Summary saved to: %s", summary_file)
        
        return summary

    def _save_results(self, result: Dict[str, Any], output_path: Path) -> None:
        """Save processing results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info("Results saved to: %s", output_path)
