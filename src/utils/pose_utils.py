"""
Shared utilities for pose processing and manipulation.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


def crop_image_with_bbox(img: np.ndarray, bbox: List[float]) -> np.ndarray:
    """
    Crop an image using bounding box coordinates.
    
    Args:
        img: Input image array
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    return img[y1:y2, x1:x2]


def normalize_keypoints_to_bbox(keypoints: Dict[str, Tuple[float, float, float]], 
                               bbox: List[float], 
                               target_width: int = 192, 
                               target_height: int = 256) -> Dict[str, Tuple[float, float, float]]:
    """
    Normalize keypoints relative to bounding box dimensions.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        target_width: Target width for normalization
        target_height: Target height for normalization
        
    Returns:
        Normalized keypoints dictionary
    """
    x1, y1, x2, y2 = bbox[:4]
    bbox_width, bbox_height = x2 - x1, y2 - y1
    
    normalized_keypoints = {}
    for name, (x, y, conf) in keypoints.items():
        # Normalize to 0-1 range within bbox, then scale to target dimensions
        norm_x = ((x - x1) / bbox_width) * target_width if bbox_width > 0 else 0
        norm_y = ((y - y1) / bbox_height) * target_height if bbox_height > 0 else 0
        normalized_keypoints[name] = (norm_x, norm_y, conf)
    
    return normalized_keypoints


def denormalize_keypoints_from_bbox(keypoints: Dict[str, Tuple[float, float, float]], 
                                   bbox: List[float], 
                                   source_width: int = 192, 
                                   source_height: int = 256) -> Dict[str, Tuple[float, float, float]]:
    """
    Denormalize keypoints from normalized coordinates back to image coordinates.
    
    Args:
        keypoints: Dictionary of normalized keypoint coordinates
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        source_width: Source width used for normalization
        source_height: Source height used for normalization
        
    Returns:
        Denormalized keypoints in image coordinates
    """
    x1, y1, x2, y2 = bbox[:4]
    bbox_width, bbox_height = x2 - x1, y2 - y1
    
    denormalized_keypoints = {}
    for name, (x, y, conf) in keypoints.items():
        # Scale from normalized dimensions back to bbox, then to image coordinates
        img_x = (x / source_width) * bbox_width + x1 if source_width > 0 else x1
        img_y = (y / source_height) * bbox_height + y1 if source_height > 0 else y1
        denormalized_keypoints[name] = (img_x, img_y, conf)
    
    return denormalized_keypoints


def filter_keypoints_by_confidence(keypoints: Dict[str, Tuple[float, float, float]], 
                                  min_confidence: float = 0.5) -> Dict[str, Tuple[float, float, float]]:
    """
    Filter keypoints based on confidence threshold.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered keypoints dictionary
    """
    return {name: coords for name, coords in keypoints.items() 
            if coords[2] >= min_confidence}


def get_keypoint_distances(keypoints: Dict[str, Tuple[float, float, float]], 
                          point1_name: str, 
                          point2_name: str) -> Optional[float]:
    """
    Calculate Euclidean distance between two keypoints.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        point1_name: Name of first keypoint
        point2_name: Name of second keypoint
        
    Returns:
        Distance between keypoints, or None if either keypoint is missing
    """
    if point1_name not in keypoints or point2_name not in keypoints:
        return None
    
    x1, y1, _ = keypoints[point1_name]
    x2, y2, _ = keypoints[point2_name]
    
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_pose_center(keypoints: Dict[str, Tuple[float, float, float]], 
                         confidence_threshold: float = 0.0) -> Optional[Tuple[float, float]]:
    """
    Calculate the center point of a pose based on visible keypoints.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        confidence_threshold: Minimum confidence for including keypoints
        
    Returns:
        (x, y) coordinates of pose center, or None if no valid keypoints
    """
    valid_points = [(x, y) for x, y, conf in keypoints.values() 
                   if conf >= confidence_threshold]
    
    if not valid_points:
        return None
    
    center_x = sum(x for x, y in valid_points) / len(valid_points)
    center_y = sum(y for x, y in valid_points) / len(valid_points)
    
    return (center_x, center_y)