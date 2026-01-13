"""
DEPRECATED: This file has been replaced by the centralized prompt management system.
Use badminton.llm_analysis.prompt_manager instead.

For backward compatibility, this module provides access to the old shot classification prompt format.
"""

import warnings
from .prompt_manager import get_shot_classification_prompt

# Issue deprecation warning
warnings.warn(
    "badminton.llm_analysis.shot_classification_prompt is deprecated. Use badminton.llm_analysis.prompt_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

# Provide backward compatibility
def get_sc_base_prompt() -> str:
    """Get the base shot classification prompt for backward compatibility."""
    return get_shot_classification_prompt("Sample input data will be inserted here")

SC_BASE_PROMPT = get_sc_base_prompt()

SC_INPUT_PROMPT = """
Input:
    Identify the shot for this input data:
    
"""