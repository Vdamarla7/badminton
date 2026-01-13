"""
Data models for PoseScript text generator.
"""

import logging
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

# Configure logging for the models module
logger = logging.getLogger(__name__)


@dataclass
class PoseData:
    """Represents pose data with keypoints and metadata."""
    keypoints: List[Tuple[float, float, float]]  # x, y, confidence
    format: str = "coco"  # Input format identifier
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not isinstance(self.keypoints, list):
            raise ValueError("keypoints must be a list")
        if not all(isinstance(kp, tuple) and len(kp) == 3 for kp in self.keypoints):
            raise ValueError("Each keypoint must be a tuple of (x, y, confidence)")
        logger.debug(f"Created PoseData with {len(self.keypoints)} keypoints in {self.format} format")
    
    def get_visible_keypoints(self) -> List[Tuple[float, float, float]]:
        """Get only keypoints with confidence > 0."""
        return [kp for kp in self.keypoints if kp[2] > 0]
    
    def get_keypoint_array(self) -> np.ndarray:
        """Convert keypoints to numpy array for easier processing."""
        return np.array(self.keypoints)
    
    def get_coordinates_only(self) -> List[Tuple[float, float]]:
        """Get only x, y coordinates without confidence."""
        return [(kp[0], kp[1]) for kp in self.keypoints]
    
    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculate bounding box of visible keypoints.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) or None if no visible keypoints
        """
        visible = self.get_visible_keypoints()
        if not visible:
            return None
        
        coords = [(kp[0], kp[1]) for kp in visible]
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoseData':
        """Create PoseData from dictionary."""
        # Convert keypoints from lists back to tuples if needed
        keypoints = data.get('keypoints', [])
        if keypoints and isinstance(keypoints[0], list):
            keypoints = [tuple(kp) for kp in keypoints]
        
        return cls(
            keypoints=keypoints,
            format=data.get('format', 'coco'),
            metadata=data.get('metadata', {})
        )


@dataclass
class GenerationResult:
    """Represents the result of text generation."""
    description: str
    confidence: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        if not isinstance(self.confidence, (int, float)) or not (0 <= self.confidence <= 1):
            raise ValueError("confidence must be a number between 0 and 1")
        if not isinstance(self.processing_time, (int, float)) or self.processing_time < 0:
            raise ValueError("processing_time must be a non-negative number")
        
        logger.debug(f"Created GenerationResult: success={self.success}, "
                    f"confidence={self.confidence:.3f}, time={self.processing_time:.3f}s")
    
    def is_successful(self) -> bool:
        """Check if generation was successful."""
        return self.success and self.error_message is None
    
    def get_summary(self) -> str:
        """Get a summary string of the result."""
        if self.success:
            return f"Success: '{self.description[:50]}...' (conf: {self.confidence:.3f}, time: {self.processing_time:.3f}s)"
        else:
            return f"Failed: {self.error_message} (time: {self.processing_time:.3f}s)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationResult':
        """Create GenerationResult from dictionary."""
        return cls(**data)


# Utility functions for common operations
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the posescript generator.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured: level={level}, file={log_file}")


def validate_keypoints_format(keypoints: Any) -> bool:
    """
    Validate if keypoints are in correct format.
    
    Args:
        keypoints: Keypoints data to validate
        
    Returns:
        bool: True if valid format
    """
    try:
        if not isinstance(keypoints, list):
            return False
        
        for kp in keypoints:
            if not isinstance(kp, (tuple, list)) or len(kp) != 3:
                return False
            if not all(isinstance(coord, (int, float)) for coord in kp):
                return False
        
        return True
    except Exception:
        return False


def create_sample_pose_data(format_type: str = "coco") -> PoseData:
    """
    Create sample pose data for testing.
    
    Args:
        format_type: Format of pose data to create
        
    Returns:
        PoseData: Sample pose data
    """
    if format_type == "coco":
        # COCO format has 17 keypoints
        # Create a simple standing pose
        keypoints = [
            (100, 50, 1.0),   # nose
            (95, 45, 1.0),    # left_eye
            (105, 45, 1.0),   # right_eye
            (90, 50, 1.0),    # left_ear
            (110, 50, 1.0),   # right_ear
            (80, 80, 1.0),    # left_shoulder
            (120, 80, 1.0),   # right_shoulder
            (70, 120, 1.0),   # left_elbow
            (130, 120, 1.0),  # right_elbow
            (60, 160, 1.0),   # left_wrist
            (140, 160, 1.0),  # right_wrist
            (85, 150, 1.0),   # left_hip
            (115, 150, 1.0),  # right_hip
            (80, 200, 1.0),   # left_knee
            (120, 200, 1.0),  # right_knee
            (75, 250, 1.0),   # left_ankle
            (125, 250, 1.0),  # right_ankle
        ]
        
        metadata = {
            "sample": True,
            "pose_type": "standing",
            "created_by": "create_sample_pose_data"
        }
        
        return PoseData(keypoints=keypoints, format="coco", metadata=metadata)
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def save_pose_data(pose_data: PoseData, filepath: Union[str, Path]) -> None:
    """
    Save pose data to JSON file.
    
    Args:
        pose_data: PoseData to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(pose_data.to_dict(), f, indent=2)
    
    logger.info(f"Saved pose data to {filepath}")


def load_pose_data(filepath: Union[str, Path]) -> PoseData:
    """
    Load pose data from JSON file.
    
    Args:
        filepath: Path to load file from
        
    Returns:
        PoseData: Loaded pose data
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    pose_data = PoseData.from_dict(data)
    logger.info(f"Loaded pose data from {filepath}")
    return pose_data


def save_generation_result(result: GenerationResult, filepath: Union[str, Path]) -> None:
    """
    Save generation result to JSON file.
    
    Args:
        result: GenerationResult to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"Saved generation result to {filepath}")


def load_generation_result(filepath: Union[str, Path]) -> GenerationResult:
    """
    Load generation result from JSON file.
    
    Args:
        filepath: Path to load file from
        
    Returns:
        GenerationResult: Loaded generation result
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    result = GenerationResult.from_dict(data)
    logger.info(f"Loaded generation result from {filepath}")
    return result


def calculate_pose_statistics(pose_data: PoseData) -> Dict[str, Any]:
    """
    Calculate basic statistics about pose data.
    
    Args:
        pose_data: PoseData to analyze
        
    Returns:
        Dict: Statistics about the pose
    """
    visible_keypoints = pose_data.get_visible_keypoints()
    all_keypoints = pose_data.keypoints
    
    stats = {
        "total_keypoints": len(all_keypoints),
        "visible_keypoints": len(visible_keypoints),
        "visibility_ratio": len(visible_keypoints) / len(all_keypoints) if all_keypoints else 0,
        "format": pose_data.format
    }
    
    if visible_keypoints:
        coords = [(kp[0], kp[1]) for kp in visible_keypoints]
        confidences = [kp[2] for kp in visible_keypoints]
        
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        
        stats.update({
            "bounding_box": {
                "min_x": min(x_coords),
                "max_x": max(x_coords),
                "min_y": min(y_coords),
                "max_y": max(y_coords),
                "width": max(x_coords) - min(x_coords),
                "height": max(y_coords) - min(y_coords)
            },
            "confidence_stats": {
                "mean": np.mean(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "std": np.std(confidences)
            }
        })
    
    return stats


def merge_generation_results(results: List[GenerationResult]) -> Dict[str, Any]:
    """
    Merge multiple generation results into summary statistics.
    
    Args:
        results: List of GenerationResult objects
        
    Returns:
        Dict: Summary statistics
    """
    if not results:
        return {"total_results": 0}
    
    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    
    summary = {
        "total_results": len(results),
        "successful_results": len(successful_results),
        "failed_results": len(failed_results),
        "success_rate": len(successful_results) / len(results)
    }
    
    if successful_results:
        processing_times = [r.processing_time for r in successful_results]
        confidences = [r.confidence for r in successful_results]
        
        summary.update({
            "processing_time_stats": {
                "mean": np.mean(processing_times),
                "min": min(processing_times),
                "max": max(processing_times),
                "total": sum(processing_times)
            },
            "confidence_stats": {
                "mean": np.mean(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "std": np.std(confidences)
            }
        })
    
    if failed_results:
        error_messages = [r.error_message for r in failed_results if r.error_message]
        summary["common_errors"] = list(set(error_messages))
    
    return summary