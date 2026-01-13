"""
DEPRECATED: This file has been replaced by the centralized prompt management system.
Use badminton.llm_analysis.prompt_manager instead.

For backward compatibility, this module provides access to the old prompt formats.
"""

import warnings
from .prompt_manager import get_pose_classification_prompt, PromptConfig

# Issue deprecation warning
warnings.warn(
    "badminton.llm_analysis.bd_prompt is deprecated. Use badminton.llm_analysis.prompt_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

# Provide backward compatibility
PROMPT_SMALL = get_pose_classification_prompt(
    "Mostly straight pointing in up (north) with the shoulder elbow pointing up (north) and the elbow wrist pointing up (north)",
    PromptConfig(template_type="simplified")
)

PROMPT_FULL = get_pose_classification_prompt(
    "Mostly straight pointing in up (north) with the shoulder elbow pointing up (north) and the elbow wrist pointing up (north)",
    PromptConfig(template_type="full")
)