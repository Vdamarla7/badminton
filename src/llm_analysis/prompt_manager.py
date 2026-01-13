"""
Centralized prompt management system for badminton analysis.
Handles loading and formatting of prompt templates and pose descriptions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from badminton.config import POSE_DESCRIPTIONS_PATH, PROMPT_TEMPLATES_PATH

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    template_type: str = "full"  # "full", "simplified", "base"
    include_examples: bool = True
    max_mappings: Optional[int] = None  # Limit number of pose mappings


class PromptManager:
    """
    Manages prompt templates and pose descriptions for LLM analysis.
    
    This class provides a centralized way to load, format, and generate
    prompts for different types of badminton analysis tasks.
    """
    
    def __init__(self):
        """Initialize the prompt manager."""
        self._pose_descriptions: Optional[Dict[str, Any]] = None
        self._prompt_templates: Optional[Dict[str, Any]] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load pose descriptions and prompt templates from JSON files."""
        try:
            # Load pose descriptions
            if POSE_DESCRIPTIONS_PATH.exists():
                with open(POSE_DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
                    self._pose_descriptions = json.load(f)
                logger.info(f"Loaded pose descriptions from {POSE_DESCRIPTIONS_PATH}")
            else:
                logger.error(f"Pose descriptions file not found: {POSE_DESCRIPTIONS_PATH}")
                self._pose_descriptions = {"pose_mappings": {}, "simplified_mappings": {}}
            
            # Load prompt templates
            if PROMPT_TEMPLATES_PATH.exists():
                with open(PROMPT_TEMPLATES_PATH, 'r', encoding='utf-8') as f:
                    self._prompt_templates = json.load(f)
                logger.info(f"Loaded prompt templates from {PROMPT_TEMPLATES_PATH}")
            else:
                logger.error(f"Prompt templates file not found: {PROMPT_TEMPLATES_PATH}")
                self._prompt_templates = {}
                
        except Exception as e:
            logger.error(f"Failed to load prompt data: {e}")
            self._pose_descriptions = {"pose_mappings": {}, "simplified_mappings": {}}
            self._prompt_templates = {}
    
    def get_pose_mappings(self, simplified: bool = False, max_mappings: Optional[int] = None) -> str:
        """
        Get formatted pose mappings for inclusion in prompts.
        
        Args:
            simplified: If True, use simplified mappings (4 basic directions)
            max_mappings: Maximum number of mappings to include
            
        Returns:
            Formatted string of pose mappings
        """
        if not self._pose_descriptions:
            return "No pose descriptions available"
        
        mappings_key = "simplified_mappings" if simplified else "pose_mappings"
        mappings = self._pose_descriptions.get(mappings_key, {})
        
        if max_mappings:
            # Take first N mappings
            mappings = dict(list(mappings.items())[:max_mappings])
        
        # Format as "Key, Description" pairs
        formatted_lines = []
        for key, description in mappings.items():
            formatted_lines.append(f"{key}, {description}")
        
        return "\n".join(formatted_lines)
    
    def get_pose_classification_prompt(self, 
                                    description: str, 
                                    config: Optional[PromptConfig] = None) -> str:
        """
        Generate a pose classification prompt.
        
        Args:
            description: The pose description to classify
            config: Configuration for prompt generation
            
        Returns:
            Formatted prompt string
        """
        if config is None:
            config = PromptConfig()
        
        templates = self._prompt_templates.get("pose_classification", {})
        
        # Select template based on config
        if config.template_type == "simplified":
            template = templates.get("simplified_template", templates.get("base_template", ""))
            pose_mappings = self.get_pose_mappings(simplified=True, max_mappings=config.max_mappings)
        else:
            template = templates.get("full_template", templates.get("base_template", ""))
            pose_mappings = self.get_pose_mappings(simplified=False, max_mappings=config.max_mappings)
        
        # Format template
        try:
            if config.template_type == "simplified":
                return template.format(
                    simplified_mappings=pose_mappings,
                    description=description
                )
            else:
                return template.format(
                    pose_mappings=pose_mappings,
                    description=description
                )
        except KeyError as e:
            logger.error(f"Template formatting error: {e}")
            return f"Error formatting template: {e}"
    
    def get_shot_classification_prompt(self, input_data: str) -> str:
        """
        Generate a shot classification prompt.
        
        Args:
            input_data: The input data (position and pose sequence)
            
        Returns:
            Formatted prompt string
        """
        templates = self._prompt_templates.get("shot_classification", {})
        
        if not templates:
            return f"Shot classification template not available.\n\nInput:\n{input_data}"
        
        # Format shot descriptions
        shot_descriptions = templates.get("shot_descriptions", {})
        shot_descriptions_formatted = self._format_shot_descriptions(shot_descriptions)
        
        # Get the full template
        full_template = templates.get("full_template", "")
        
        try:
            return full_template.format(
                identity=templates.get("identity", ""),
                task_description=templates.get("task_description", ""),
                input_format=templates.get("input_format", ""),
                attention_constraint=templates.get("attention_constraint", ""),
                output_format=templates.get("output_format", ""),
                additional_guidance=templates.get("additional_guidance", ""),
                shot_descriptions_formatted=shot_descriptions_formatted,
                input_data=input_data
            )
        except KeyError as e:
            logger.error(f"Shot classification template formatting error: {e}")
            return f"Error formatting shot classification template: {e}\n\nInput:\n{input_data}"
    
    def _format_shot_descriptions(self, shot_descriptions: Dict[str, Any]) -> str:
        """Format shot descriptions for inclusion in prompts."""
        formatted_sections = []
        
        for shot_type, details in shot_descriptions.items():
            section_lines = [f"    {shot_type}:"]
            
            for key, value in details.items():
                if key == "right_arm_orientations" and isinstance(value, dict):
                    section_lines.append(f"        Right_Arm Orientations:")
                    for sub_key, sub_value in value.items():
                        section_lines.append(f"            {sub_key.title()}: {sub_value}")
                elif key == "left_arm_orientations" and isinstance(value, dict):
                    section_lines.append(f"        Left_Arm Orientations:")
                    for sub_key, sub_value in value.items():
                        section_lines.append(f"            {sub_key.title()}: {sub_value}")
                else:
                    formatted_key = key.replace("_", " ").title()
                    if isinstance(value, str):
                        # Handle multi-line values
                        if "\n" in value:
                            section_lines.append(f"        {formatted_key}:")
                            for line in value.split("\n"):
                                section_lines.append(f"            {line}")
                        else:
                            section_lines.append(f"        {formatted_key}: {value}")
            
            section_lines.append("")  # Add blank line between shots
            formatted_sections.append("\n".join(section_lines))
        
        return "\n".join(formatted_sections)
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template types."""
        if not self._prompt_templates:
            return []
        
        templates = []
        for category, category_templates in self._prompt_templates.items():
            if isinstance(category_templates, dict):
                for template_name in category_templates.keys():
                    if template_name.endswith("_template"):
                        templates.append(f"{category}.{template_name}")
        
        return templates
    
    def reload_data(self) -> bool:
        """
        Reload prompt data from files.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            self._load_data()
            logger.info("Prompt data reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload prompt data: {e}")
            return False
    
    def validate_data(self) -> Dict[str, bool]:
        """
        Validate loaded data integrity.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "pose_descriptions_loaded": self._pose_descriptions is not None,
            "prompt_templates_loaded": self._prompt_templates is not None,
            "pose_mappings_available": False,
            "simplified_mappings_available": False,
            "pose_classification_templates_available": False,
            "shot_classification_templates_available": False
        }
        
        if self._pose_descriptions:
            results["pose_mappings_available"] = bool(self._pose_descriptions.get("pose_mappings"))
            results["simplified_mappings_available"] = bool(self._pose_descriptions.get("simplified_mappings"))
        
        if self._prompt_templates:
            results["pose_classification_templates_available"] = bool(
                self._prompt_templates.get("pose_classification")
            )
            results["shot_classification_templates_available"] = bool(
                self._prompt_templates.get("shot_classification")
            )
        
        return results


# Convenience functions for backward compatibility
def get_pose_classification_prompt(description: str, config_or_simplified = None) -> str:
    """
    Convenience function to get a pose classification prompt.
    
    Args:
        description: The pose description to classify
        config_or_simplified: Either a PromptConfig object or a boolean for simplified mode
        
    Returns:
        Formatted prompt string
    """
    manager = PromptManager()
    
    # Handle both old and new calling conventions
    if isinstance(config_or_simplified, PromptConfig):
        config = config_or_simplified
    elif isinstance(config_or_simplified, bool):
        config = PromptConfig(template_type="simplified" if config_or_simplified else "full")
    elif config_or_simplified is None:
        config = PromptConfig(template_type="full")
    else:
        # Default to full if we don't understand the input
        config = PromptConfig(template_type="full")
    
    return manager.get_pose_classification_prompt(description, config)


def get_shot_classification_prompt(input_data: str) -> str:
    """
    Convenience function to get a shot classification prompt.
    
    Args:
        input_data: The input data (position and pose sequence)
        
    Returns:
        Formatted prompt string
    """
    manager = PromptManager()
    return manager.get_shot_classification_prompt(input_data)


# Legacy compatibility - replicate old prompt variables
def get_legacy_prompts() -> Dict[str, str]:
    """
    Get legacy prompt formats for backward compatibility.
    
    Returns:
        Dictionary with legacy prompt variable names
    """
    manager = PromptManager()
    
    # Generate sample prompts to match old format
    sample_description = "Mostly straight pointing in up (north) with the shoulder elbow pointing up (north) and the elbow wrist pointing up (north)"
    
    return {
        "prompt": manager.get_pose_classification_prompt(sample_description),
        "PROMPT_SMALL": manager.get_pose_classification_prompt(sample_description, PromptConfig(template_type="simplified")),
        "PROMPT_FULL": manager.get_pose_classification_prompt(sample_description, PromptConfig(template_type="full")),
        "SC_BASE_PROMPT": manager._prompt_templates.get("shot_classification", {}).get("identity", "") if manager._prompt_templates else "",
        "SC_INPUT_PROMPT": "Input:\n    Identify the shot for this input data:\n    \n"
    }