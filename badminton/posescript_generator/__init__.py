"""
PoseScript Text Generator

A minimal system to generate text descriptions from pose data using the PoseScript library.
"""

__version__ = "0.1.0"
__author__ = "PoseScript Generator Team"

from .model_manager import ModelManager
from .models import (
    PoseData, 
    GenerationResult,
    setup_logging,
    validate_keypoints_format,
    create_sample_pose_data,
    save_pose_data,
    load_pose_data,
    save_generation_result,
    load_generation_result,
    calculate_pose_statistics,
    merge_generation_results
)

__all__ = [
    "ModelManager",
    "PoseData",
    "GenerationResult",
    "setup_logging",
    "validate_keypoints_format", 
    "create_sample_pose_data",
    "save_pose_data",
    "load_pose_data",
    "save_generation_result",
    "load_generation_result",
    "calculate_pose_statistics",
    "merge_generation_results"
]

# Note: TextGenerator and PoseProcessor will be added when implemented in future tasks