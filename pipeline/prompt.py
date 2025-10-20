"""
Prompt management system for the VLM OCR Pipeline.
Handles loading, caching, and retrieving prompts from YAML files with VLM-specific fallback support.
"""

import glob
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt loading and retrieval with YAML-based configuration and fallbacks"""

    def __init__(self, model: str, backend: str = "openai"):
        """
        Initialize PromptManager

        Args:
            model: Model name (e.g., "google/gemini-2.5-flash", "openai/gpt-4o")
            backend: Backend type ("openai" or "gemini")
        """
        self.model = model
        self.backend = backend
        self.prompts_dir = Path(self._find_best_prompts_dir(model, backend))
        self.prompts = self._load_prompts()

    def _find_best_prompts_dir(self, model: str, backend: str = "openai") -> str:
        """Find the best matching prompts directory using hierarchical matching.

        Priority:
        1. Exact model match: prompts/{org}/{full_model_name}/
        2. Wildcard prefix match: prompts/{org}/{model_prefix}*/
        3. Organization match: prompts/{org}/
        4. Default fallback: prompts/default/

        Args:
            model: Model name (e.g., "google/gemini-2.5-flash", "openai/gpt-4o")
            backend: Backend type (for legacy compatibility)

        Returns:
            Path to the best matching prompts directory
        """
        base_prompts_dir = Path("settings/prompts")

        # Parse model name
        if "/" in model:
            org, model_name = model.split("/", 1)
        else:
            # Handle models without org prefix
            org = None
            model_name = model

        candidates = []

        if org:
            # 1. Exact match: prompts/org/full_model_name/
            exact_path = base_prompts_dir / org / model_name
            if exact_path.is_dir():
                candidates.append((1, str(exact_path)))

            # 2. Wildcard prefix match: prompts/org/prefix*/
            org_dir = base_prompts_dir / org
            if org_dir.is_dir():
                for prefix_len in range(len(model_name), 0, -1):
                    prefix = model_name[:prefix_len]
                    pattern = str(org_dir / f"{prefix}*")
                    matches = glob.glob(pattern)
                    for match in matches:
                        match_path = Path(match)
                        if match_path.is_dir() and match_path.name != model_name:
                            # Calculate specificity by prefix length
                            specificity = len(prefix)
                            candidates.append((2, str(match_path), specificity))

            # 3. Organization match: prompts/org/
            org_path = base_prompts_dir / org
            if org_path.is_dir():
                candidates.append((3, str(org_path)))

        # 4. Default fallback
        default_path = base_prompts_dir / "default"
        if default_path.is_dir():
            candidates.append((4, str(default_path)))

        # Sort by priority (lower number = higher priority)
        # For wildcards, also sort by specificity (higher = better)
        if candidates:

            def sort_key(item):
                priority, path, *rest = item
                if rest:  # Wildcard match with specificity
                    specificity = rest[0]
                    return (priority, -specificity)  # Negative for descending order
                return (priority,)

            candidates.sort(key=sort_key)
            head = candidates[0]
            return head[1]

        # Ultimate fallback if nothing found
        return str(base_prompts_dir / "default")

    def _load_prompts(self) -> dict[str, Any]:
        """Load prompts from YAML files"""
        prompts = {}

        if not self.prompts_dir.exists():
            logger.warning("Prompts directory not found: %s", self.prompts_dir)
            return prompts

        prompt_files = {
            "text_extraction": "text_extraction.yaml",
            "content_analysis": "content_analysis.yaml",
            "text_correction": "text_correction.yaml",
        }

        for prompt_type, filename in prompt_files.items():
            prompt_file = self.prompts_dir / filename

            if prompt_file.exists():
                try:
                    with prompt_file.open("r", encoding="utf-8") as f:
                        prompts[prompt_type] = yaml.safe_load(f)
                    logger.debug("Loaded prompts from %s", prompt_file)
                except Exception as e:
                    logger.warning("Failed to load prompts from %s: %s", prompt_file, e)
            else:
                logger.warning("Prompt file not found: %s", prompt_file)

        logger.info(
            "PromptManager initialized (model=%s, backend=%s, dir=%s)", self.model, self.backend, self.prompts_dir
        )
        return prompts

    def get_prompt(self, category: str, prompt_type: str, prompt_key: str | None = None, **kwargs) -> str:
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
                        prompt_template = prompt_section.get("user", str(prompt_section))
                else:
                    # Direct access or string value
                    prompt_template = prompt_section

                # Format template with kwargs if provided
                if isinstance(prompt_template, str) and kwargs:
                    return prompt_template.format(**kwargs)
                return str(prompt_template)

        except Exception as e:
            logger.warning("Error getting prompt %s.%s: %s", category, prompt_type, e)

        # Fallback to hardcoded prompts
        return self._get_fallback_prompt(category, prompt_type, **kwargs)

    def _get_fallback_prompt(self, category: str, prompt_type: str, **kwargs) -> str:
        """Fallback prompts when YAML files are not available"""
        fallback_prompts = {
            "text_extraction": {
                "system": (
                    "You are an expert OCR system. Extract all visible text accurately, "
                    "maintaining original language and formatting."
                ),
                "user": (
                    "Extract all text from this image accurately. Maintain original language "
                    "and formatting. Return only the extracted text."
                ),
                "fallback": "Extract all visible text from this image accurately.",
            },
            "content_analysis": {
                "table_analysis": (
                    "Analyze this table and respond in JSON format:\n"
                    '{"markdown_table": "| Col1 | Col2 |\\\\n|------|------|\\\\n| Data1 | Data2 |",'
                    ' "summary": "Description", "educational_value": "Significance",'
                    ' "related_topics": ["Topic1", "Topic2"]}'
                ),
                "figure_analysis": (
                    "Analyze this image and respond in JSON format:\n"
                    '{"description": "Detailed description", "educational_value": "Significance",'
                    ' "related_topics": ["Topic1"], "exam_relevance": "Exam usage"}'
                ),
            },
            "text_correction": {
                "user": "Correct OCR errors in this text while preserving original language and special tags:\n{text}"
            },
        }

        try:
            if category in fallback_prompts:
                if prompt_type in fallback_prompts[category]:
                    prompt = fallback_prompts[category][prompt_type]
                    return prompt.format(**kwargs) if kwargs else prompt
        except Exception as e:
            logger.error("Error in fallback prompt: %s", e)

        return f"Process this content according to {category} guidelines."

    def get_prompt_for_region_type(self, region_type: str) -> str:
        """Get appropriate prompt for a block type (backend-agnostic)"""
        region_type_mapping = {
            "table": "table_analysis",
            "figure": "figure_analysis",
            "formula": "figure_analysis",  # Use figure analysis for formulas
            "title": "figure_analysis",
            "list": "figure_analysis",
            "plain text": "figure_analysis",
        }

        analysis_type = region_type_mapping.get(region_type, "figure_analysis")
        return self.get_prompt("content_analysis", analysis_type, "user")

    def get_gemini_prompt_for_region_type(self, region_type: str) -> str:
        """Get Gemini-specific prompt for a block type (deprecated, use get_prompt_for_region_type)"""
        return self.get_prompt_for_region_type(region_type)

    def reload_prompts(self) -> None:
        """Reload prompts from disk (useful for development)"""
        self.prompts = self._load_prompts()
        logger.info("Prompts reloaded from disk")
