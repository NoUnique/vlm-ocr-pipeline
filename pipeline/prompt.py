"""
Prompt management system for the VLM OCR Pipeline.
Handles loading, caching, and retrieving prompts from YAML files with VLM-specific fallback support.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt loading and retrieval with YAML-based configuration and fallbacks"""
    
    def __init__(self, prompts_dir: Path):
        """
        Initialize PromptManager
        
        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML files"""
        prompts = {}
        
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            return prompts
        
        prompt_files = {
            'text_extraction': 'text_extraction.yaml',
            'content_analysis': 'content_analysis.yaml', 
            'text_correction': 'text_correction.yaml'
        }
        
        for prompt_type, filename in prompt_files.items():
            prompt_file = self.prompts_dir / filename
            
            if prompt_file.exists():
                try:
                    with prompt_file.open('r', encoding='utf-8') as f:
                        prompts[prompt_type] = yaml.safe_load(f)
                    logger.info(f"Loaded prompts from {prompt_file}")
                except Exception as e:
                    logger.warning(f"Failed to load prompts from {prompt_file}: {e}")
            else:
                logger.warning(f"Prompt file not found: {prompt_file}")
        
        return prompts

    def get_prompt(self, category: str, prompt_type: str, prompt_key: str = None, **kwargs) -> str:
        """Get prompt from loaded prompts with fallback"""
        try:
            prompt_data = self.prompts.get(category, {})
            
            # Handle nested prompt structure (e.g., content_analysis.table_analysis.user)
            if isinstance(prompt_data, dict) and prompt_type in prompt_data:
                prompt_section = prompt_data[prompt_type]
                
                # If prompt_key is specified, get specific key from the section
                if prompt_key and isinstance(prompt_section, dict):
                    if prompt_key in prompt_section:
                        prompt_template = prompt_section[prompt_key]
                    else:
                        # Try 'user' as fallback if specified key not found
                        prompt_template = prompt_section.get('user', str(prompt_section))
                else:
                    # Direct access or string value
                    prompt_template = prompt_section
                
                # Format template with kwargs if provided
                if isinstance(prompt_template, str) and kwargs:
                    return prompt_template.format(**kwargs)
                return str(prompt_template)
                
        except Exception as e:
            logger.warning(f"Error getting prompt {category}.{prompt_type}: {e}")
        
        # Fallback to hardcoded prompts
        return self._get_fallback_prompt(category, prompt_type, **kwargs)

    def _get_fallback_prompt(self, category: str, prompt_type: str, **kwargs) -> str:
        """Fallback prompts when YAML files are not available"""
        fallback_prompts = {
            'text_extraction': {
                'system': "You are an expert OCR system. Extract all visible text accurately, maintaining original language and formatting.",
                'user': "Extract all text from this image accurately. Maintain original language and formatting. Return only the extracted text.",
                'fallback': "Extract all visible text from this image accurately."
            },
            'content_analysis': {
                'table_analysis': '''Analyze this table and respond in JSON format:
                {"markdown_table": "| Col1 | Col2 |\\n|------|------|\\n| Data1 | Data2 |", "summary": "Description", "educational_value": "Significance", "related_topics": ["Topic1", "Topic2"]}''',
                'figure_analysis': '''Analyze this image and respond in JSON format:
                {"description": "Detailed description", "educational_value": "Significance", "related_topics": ["Topic1"], "exam_relevance": "Exam usage"}'''
            },
            'text_correction': {
                'user': "Correct OCR errors in this text while preserving original language and special tags:\n{text}"
            }
        }
        
        try:
            if category in fallback_prompts:
                if prompt_type in fallback_prompts[category]:
                    prompt = fallback_prompts[category][prompt_type]
                    return prompt.format(**kwargs) if kwargs else prompt
        except Exception as e:
            logger.error(f"Error in fallback prompt: {e}")
        
        return f"Process this content according to {category} guidelines."

    def get_prompt_for_region_type(self, region_type: str) -> str:
        """Get appropriate prompt for a region type (backend-agnostic)"""
        region_type_mapping = {
            'table': 'table_analysis',
            'figure': 'figure_analysis',
            'formula': 'figure_analysis',  # Use figure analysis for formulas
            'title': 'figure_analysis',
            'list': 'figure_analysis',
            'plain text': 'figure_analysis'
        }
        
        analysis_type = region_type_mapping.get(region_type, 'figure_analysis')
        return self.get_prompt('content_analysis', analysis_type, 'user')

    def get_gemini_prompt_for_region_type(self, region_type: str) -> str:
        """Get Gemini-specific prompt for a region type (deprecated, use get_prompt_for_region_type)"""
        return self.get_prompt_for_region_type(region_type)
    
    def reload_prompts(self) -> None:
        """Reload prompts from disk (useful for development)"""
        self.prompts = self._load_prompts()
        logger.info("Prompts reloaded from disk") 