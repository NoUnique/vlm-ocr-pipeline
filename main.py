#!/usr/bin/env python3
"""
Main entry point for the VLM OCR Pipeline
Provides command-line interface for processing images and PDFs using Vision Language Models
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pipeline import Pipeline

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration with timestamped log files"""
    # Create .logs directory if it doesn't exist
    logs_dir = Path('.logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = logs_dir / f'{timestamp}_ocr_pipeline.log'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename, encoding='utf-8')
        ]
    )

def parse_page_range(page_range_str: str) -> Optional[tuple]:
    """Parse page range string like '1-10' into tuple (1, 10)"""
    try:
        if '-' not in page_range_str:
            raise ValueError("Page range must contain '-'")
        
        start_str, end_str = page_range_str.split('-', 1)
        start_page = int(start_str.strip())
        end_page = int(end_str.strip())
        
        if start_page < 1 or end_page < 1 or start_page > end_page:
            raise ValueError("Invalid page range")
        
        return (start_page, end_page)
    except Exception as e:
        logging.getLogger(__name__).error(f"Invalid page range format '{page_range_str}': {e}")
        return None

def parse_specific_pages(pages_str: str) -> Optional[list]:
    """Parse comma-separated pages like '1,3,5,10' into list [1, 3, 5, 10]"""
    try:
        pages = []
        for page_str in pages_str.split(','):
            page_num = int(page_str.strip())
            if page_num < 1:
                raise ValueError(f"Page number must be positive: {page_num}")
            pages.append(page_num)
        
        if not pages:
            raise ValueError("No valid page numbers found")
        
        return sorted(list(set(pages)))  # Remove duplicates and sort
    except Exception as e:
        logging.getLogger(__name__).error(f"Invalid pages format '{pages_str}': {e}")
        return None

def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="VLM OCR Pipeline - Process images and PDFs with layout detection and VLM-powered text correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input document.pdf
  python main.py --input document.pdf --backend gemini
  python main.py --input document.pdf --model openai/gpt-4o
  python main.py --input document.pdf --backend openai --model gemini-2.5-flash
  python main.py --input /path/to/images/
  python main.py --input document.pdf --output /custom/output/
  python main.py --input document.pdf --text-extraction vision
  python main.py --input document.pdf --max-pages 5
  python main.py --input document.pdf --page-range 10-20
  python main.py --input document.pdf --pages 1,5,10,15
  python main.py --input document.pdf --no-cache --confidence 0.7
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file or directory path (PDF, image, or directory containing PDFs)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory path (default: ./output)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to custom DocLayout-YOLO model (optional)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching (default: caching enabled)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='.cache',
        help='Cache directory path (default: ./.cache)'
    )
    
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='.tmp',
        help='Temporary files directory path (default: ./.tmp)'
    )
    
    parser.add_argument(
        '--text-extraction',
        choices=['gemini', 'vision'],
        default='gemini',
        help='Text extraction method: "gemini" (Gemini API) or "vision" (Google Vision API) (default: gemini)'
    )
    
    parser.add_argument(
        '--backend',
        choices=['openai', 'gemini'],
        default='openai',
        help='Backend API to use: "openai" (OpenAI/OpenRouter) or "gemini" (Google Gemini) (default: openai)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash',
        help='Model to use (default: gemini-2.5-flash). For OpenRouter: use format like "openai/gpt-4"'
    )
    
    parser.add_argument(
        '--prompts-dir',
        type=str,
        help='Directory containing prompt templates (auto-detected based on backend/model if not specified)'
    )
    
    parser.add_argument(
        '--gemini-tier',
        type=str,
        choices=['free', 'tier1', 'tier2', 'tier3'],
        default='free',
        help='Gemini API tier for rate limiting (only used with --backend gemini) (default: free)'
    )
    
    # Page limiting options
    page_group = parser.add_mutually_exclusive_group()
    page_group.add_argument(
        '--max-pages',
        type=int,
        help='Maximum number of pages to process from the beginning (e.g., --max-pages 5)'
    )
    page_group.add_argument(
        '--page-range',
        type=str,
        help='Page range to process in format "start-end" (e.g., --page-range 1-10)'
    )
    page_group.add_argument(
        '--pages',
        type=str,
        help='Specific pages to process, comma-separated (e.g., --pages 1,3,5,10)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--rate-limit-status',
        action='store_true',
        help='Show current rate limit status and exit'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Handle rate limit status request (only for Gemini backend)
    if args.rate_limit_status:
        if args.backend == 'gemini':
            from pipeline.ratelimit import rate_limiter
            rate_limiter.set_tier_and_model(args.gemini_tier, args.model)
            status = rate_limiter.get_status()
            
            print(f"\n📊 Gemini API Rate Limit Status")
            print(f"{'='*50}")
            print(f"Tier: {status['tier']}")
            print(f"Current Model: {status['model']}")
            print(f"\n🔢 Current Model Usage:")
            print(f"  Requests per minute: {status['current']['rpm']}/{status['limits']['rpm'] or 'unlimited'}")
            print(f"  Tokens per minute: {status['current']['tpm']:,}/{status['limits']['tpm']:,}" if status['limits']['tpm'] else f"  Tokens per minute: {status['current']['tpm']:,}/unlimited")
            print(f"  Requests per day: {status['current']['rpd']}/{status['limits']['rpd'] or 'unlimited'}")
            print(f"\n📈 Current Model Utilization:")
            print(f"  RPM: {status['utilization']['rpm_percent']:.1f}%")
            print(f"  TPM: {status['utilization']['tpm_percent']:.1f}%")
            print(f"  RPD: {status['utilization']['rpd_percent']:.1f}%")
            
            # Show all models summary
            if status.get('all_models') and len(status['all_models']) > 1:
                print(f"\n📋 All Models Summary:")
                for model_name, model_usage in status['all_models'].items():
                    is_current = "← CURRENT" if model_name == status['model'] else ""
                    print(f"  {model_name}: RPM:{model_usage['rpm']} TPM:{model_usage['tpm']:,} RPD:{model_usage['rpd']} {is_current}")
        else:
            print(f"\n⚠️  Rate limit status is only available for Gemini backend")
            print(f"Current backend: {args.backend}")
        return
    
    # Validate input is provided when not checking rate limit status
    if not args.input:
        parser.error("the following arguments are required: --input/-i")
    
    # Auto-detect prompts directory if not specified
    if not args.prompts_dir:
        # Determine model family for prompt selection
        if args.backend == 'gemini':
            model_family = 'gemini'
        else:
            # For OpenAI backend, try to detect model family
            model_lower = args.model.lower()
            if 'gpt' in model_lower or 'openai' in model_lower:
                model_family = 'openai'
            elif 'internvl' in model_lower:
                model_family = 'internvl'
            elif 'qwen' in model_lower:
                model_family = 'qwen'
            elif 'phi' in model_lower:
                model_family = 'phi4'
            elif 'gemini' in model_lower:
                model_family = 'gemini'
            else:
                model_family = 'openai'  # Default fallback
        
        args.prompts_dir = f'settings/prompts/{model_family}'
        logger.info(f"Auto-detected prompts directory: {args.prompts_dir}")
    
    logger.info("Starting VLM OCR Pipeline")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Model: {args.model}")
    if args.backend == 'gemini':
        logger.info(f"Gemini Tier: {args.gemini_tier}")
    
    try:
        pipeline = Pipeline(
            model_path=args.model_path,
            confidence_threshold=args.confidence,
            use_cache=not args.no_cache,
            cache_dir=args.cache_dir,
            output_dir=args.output,
            temp_dir=args.temp_dir,
            text_extraction_method=args.text_extraction,
            prompts_dir=args.prompts_dir,
            backend=args.backend,
            model=args.model,
            gemini_tier=args.gemini_tier
        )
        
        input_path = Path(args.input)
        
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return 1
        
        # Parse page limiting options
        max_pages = args.max_pages
        page_range = None
        specific_pages = None
        
        if args.page_range:
            page_range = parse_page_range(args.page_range)
            if page_range is None:
                return 1
        
        if args.pages:
            specific_pages = parse_specific_pages(args.pages)
            if specific_pages is None:
                return 1
        
        # Log page limiting options if set
        if max_pages:
            logger.info(f"Limiting to maximum {max_pages} pages")
        elif page_range:
            logger.info(f"Processing page range: {page_range[0]}-{page_range[1]}")
        elif specific_pages:
            logger.info(f"Processing specific pages: {specific_pages}")
        
        if input_path.is_file():
            file_ext = input_path.suffix.lower()
            
            if file_ext == '.pdf':
                logger.info(f"Processing PDF file: {input_path}")
                result = pipeline.process_pdf(
                    input_path, 
                    max_pages=max_pages,
                    page_range=page_range,
                    pages=specific_pages
                )
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                logger.info(f"Processing image file: {input_path}")
                result = pipeline.process_image(input_path)
                
                model_output_dir = Path(args.output) / pipeline.model
                output_path = model_output_dir / f"{input_path.stem}.json"
                pipeline._save_results(result, output_path)
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return 1
                
        elif input_path.is_dir():
            logger.info(f"Processing directory: {input_path}")
            
            # Note: Page limiting options only apply to individual PDFs
            if max_pages or page_range or specific_pages:
                logger.warning("Page limiting options are applied to each PDF individually in directory mode")
            
            result = pipeline.process_directory(
                input_path, 
                args.output,
                max_pages=max_pages,
                page_range=page_range,
                specific_pages=specific_pages
            )
        else:
            logger.error(f"Invalid input path: {input_path}")
            return 1
        
        if 'error' in result:
            logger.error(f"Processing failed: {result['error']}")
            return 1
        
        logger.info("VLM OCR Pipeline completed successfully")
        
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                logger.info(f"Results saved to: {result.get('output_directory', args.output)}")
            else:
                logger.info(f"Results saved to: {Path(args.output) / pipeline.model}")
        elif input_path.is_dir():
            logger.info(f"Results saved to: {result.get('output_directory', args.output)}")
        else:
            logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 