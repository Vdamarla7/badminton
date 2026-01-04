"""
DEPRECATED: This file has been replaced by the centralized prompt management system.
Use badminton.llm_analysis.prompt_manager instead.

For backward compatibility, this module provides access to the old prompt format.
"""

import warnings
from .prompt_manager import get_pose_classification_prompt, PromptConfig

# Issue deprecation warning
warnings.warn(
    "badminton.llm_analysis.prompt is deprecated. Use badminton.llm_analysis.prompt_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

# Provide backward compatibility
def get_prompt(description: str = None) -> str:
    """
    Get pose classification prompt for backward compatibility.
    
    Args:
        description: Optional description to classify
        
    Returns:
        Formatted prompt string
    """
    if description is None:
        description = "Mostly straight pointing in up (north) with the shoulder elbow pointing up (north) and the elbow wrist pointing up (north)"
    
    return get_pose_classification_prompt(description, PromptConfig(template_type="full"))

# Legacy variable for backward compatibility
prompt = get_prompt()