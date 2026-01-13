"""
Centralized configuration for the badminton analysis system.
All paths, model settings, and constants should be defined here.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "VB_DATA"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "output"

# Video and data paths
VIDEO_METADATA_PATH = DATA_ROOT / "video_badminton_metadata.csv"
POSES_DATA_PATH = DATA_ROOT / "poses"
ANNOTATED_VIDEO_PATH = PROJECT_ROOT / "annotated_video.mp4"

# Model paths
YOLO_MODEL_PATH = MODELS_ROOT / "yolov8m.pt"
POSITION_CLASSIFIER_PATH = MODELS_ROOT / "position_classifier.pkl"
POSITION_LABEL_ENCODER_PATH = MODELS_ROOT / "position_label_encoder.pkl"
POSITION_SCALER_PATH = MODELS_ROOT / "position_scaler.pkl"

# Sapiens model paths
SAPIENS_1B_MODEL_PATH = MODELS_ROOT / "sapiens_1b_coco_best_coco_AP_821.pth"
SAPIENS_2B_MODEL_PATH = MODELS_ROOT / "sapiens_2b_coco_best_coco_AP_822_torchscript.pt2"

# PoseScript paths (IGNORED - not part of refactoring)
# POSESCRIPT_ROOT = PROJECT_ROOT.parent / "posescript"
# POSESCRIPT_SRC_PATH = POSESCRIPT_ROOT / "src"
# POSESCRIPT_MODEL_PATH = POSESCRIPT_ROOT / "capgen" / "seed1" / "checkpoint_best.pth"

# Prompt and data files
POSE_DESCRIPTIONS_PATH = PROJECT_ROOT / "data" / "pose_descriptions.json"
PROMPT_TEMPLATES_PATH = PROJECT_ROOT / "data" / "prompt_templates.json"

# Output directories (excluding posescript_generator)
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "synthetic_data"

# Model configuration
MODEL_CONFIG = {
    "yolo": {
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
        "max_detections": 10
    },
    "sapiens": {
        "input_size": (256, 192),
        "confidence_threshold": 0.3,
        "device": "auto"  # "cuda", "cpu", or "auto"
    },
    "posescript": {
        "max_length": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "batch_size": 1
    }
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "keypoint_radius": 2,
    "keypoint_thickness": 2,
    "bbox_thickness": 2,
    "line_thickness": 1,
    "colors": {
        "player_a": (0, 255, 0),    # Green
        "player_b": (255, 0, 0),    # Blue (BGR format)
        "keypoints": (255, 255, 0), # Yellow
        "skeleton": (0, 255, 255)   # Cyan
    }
}

# Video processing settings
VIDEO_CONFIG = {
    "fps": 30,
    "frame_skip": 1,
    "max_frames": None,  # None for all frames
    "resize_factor": 1.0
}

# COCO keypoint configuration
COCO_CONFIG = {
    "num_keypoints": 17,
    "keypoint_names": [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ],
    "skeleton_connections": [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 12), (5, 11), (6, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "log_file": PROJECT_ROOT / "logs" / "badminton_analysis.log"
}

# Environment variable overrides
def get_env_path(env_var: str, default_path: Path) -> Path:
    """Get path from environment variable or use default."""
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value)
    return default_path

def get_env_config(env_var: str, default_value: Any) -> Any:
    """Get configuration value from environment variable or use default."""
    env_value = os.getenv(env_var)
    if env_value is not None:
        # Try to convert to same type as default
        if isinstance(default_value, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default_value, int):
            return int(env_value)
        elif isinstance(default_value, float):
            return float(env_value)
        return env_value
    return default_value

# Apply environment overrides
YOLO_MODEL_PATH = get_env_path("BADMINTON_YOLO_MODEL", YOLO_MODEL_PATH)
# POSESCRIPT_MODEL_PATH = get_env_path("BADMINTON_POSESCRIPT_MODEL", POSESCRIPT_MODEL_PATH)  # Ignored
MODEL_CONFIG["yolo"]["confidence_threshold"] = get_env_config("BADMINTON_YOLO_CONFIDENCE", 
                                                              MODEL_CONFIG["yolo"]["confidence_threshold"])

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        MODELS_ROOT,
        OUTPUT_ROOT,
        SYNTHETIC_DATA_PATH,
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_paths() -> Dict[str, bool]:
    """Validate that critical paths exist."""
    critical_paths = {
        "PROJECT_ROOT": PROJECT_ROOT.exists(),
        "DATA_ROOT": DATA_ROOT.exists(),
        "MODELS_ROOT": MODELS_ROOT.exists(),
    }
    
    optional_paths = {
        "VIDEO_METADATA": VIDEO_METADATA_PATH.exists(),
        "YOLO_MODEL": YOLO_MODEL_PATH.exists(),
        # "POSESCRIPT_MODEL": POSESCRIPT_MODEL_PATH.exists(),  # Ignored
    }
    
    return {**critical_paths, **optional_paths}

def get_config_summary() -> Dict[str, Any]:
    """Get a summary of current configuration."""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(DATA_ROOT),
        "models_root": str(MODELS_ROOT),
        "model_config": MODEL_CONFIG,
        "path_validation": validate_paths()
    }

# Initialize directories on import
ensure_directories()