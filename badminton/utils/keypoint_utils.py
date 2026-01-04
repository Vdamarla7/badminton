"""
Shared utilities for COCO keypoint handling and manipulation.
"""

from typing import Dict, List, Tuple, Optional
from ..utilities.coco_keypoints import COCO_KEYPOINTS, COCO_SKELETON_INFO


def validate_keypoints_dict(keypoints: Dict[str, Tuple[float, float, float]]) -> bool:
    """
    Validate that a keypoints dictionary contains all required COCO keypoints.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        
    Returns:
        True if all required keypoints are present, False otherwise
    """
    return all(kp_name in keypoints for kp_name in COCO_KEYPOINTS)


def get_missing_keypoints(keypoints: Dict[str, Tuple[float, float, float]]) -> List[str]:
    """
    Get list of missing keypoints from a keypoints dictionary.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        
    Returns:
        List of missing keypoint names
    """
    return [kp_name for kp_name in COCO_KEYPOINTS if kp_name not in keypoints]


def fill_missing_keypoints(keypoints: Dict[str, Tuple[float, float, float]], 
                          default_value: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Dict[str, Tuple[float, float, float]]:
    """
    Fill missing keypoints with default values.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        default_value: Default (x, y, confidence) tuple for missing keypoints
        
    Returns:
        Complete keypoints dictionary with all COCO keypoints
    """
    complete_keypoints = keypoints.copy()
    
    for kp_name in COCO_KEYPOINTS:
        if kp_name not in complete_keypoints:
            complete_keypoints[kp_name] = default_value
    
    return complete_keypoints


def keypoints_to_flat_list(keypoints: Dict[str, Tuple[float, float, float]]) -> List[float]:
    """
    Convert keypoints dictionary to flat list in COCO order.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        
    Returns:
        Flat list of [x1, y1, c1, x2, y2, c2, ...] in COCO keypoint order
    """
    flat_list = []
    for kp_name in COCO_KEYPOINTS:
        if kp_name in keypoints:
            x, y, conf = keypoints[kp_name]
            flat_list.extend([x, y, conf])
        else:
            flat_list.extend([0.0, 0.0, 0.0])  # Default for missing keypoints
    
    return flat_list


def flat_list_to_keypoints(values: List[float]) -> Dict[str, Tuple[float, float, float]]:
    """
    Convert flat list to keypoints dictionary.
    
    Args:
        values: Flat list of [x1, y1, c1, x2, y2, c2, ...] values
        
    Returns:
        Dictionary of keypoint names to (x, y, confidence) tuples
    """
    if len(values) != len(COCO_KEYPOINTS) * 3:
        raise ValueError(f"Expected {len(COCO_KEYPOINTS) * 3} values, got {len(values)}")
    
    keypoints = {}
    for i, kp_name in enumerate(COCO_KEYPOINTS):
        idx = i * 3
        x, y, conf = values[idx], values[idx + 1], values[idx + 2]
        keypoints[kp_name] = (float(x), float(y), float(conf))
    
    return keypoints


def get_skeleton_connections() -> List[Tuple[str, str]]:
    """
    Get list of keypoint pairs that form skeleton connections.
    
    Returns:
        List of (keypoint1, keypoint2) tuples representing skeleton connections
    """
    connections = []
    for link_info in COCO_SKELETON_INFO.values():
        pt1_name, pt2_name = link_info['link']
        connections.append((pt1_name, pt2_name))
    
    return connections


def get_body_part_keypoints(body_part: str) -> List[str]:
    """
    Get keypoints belonging to a specific body part.
    
    Args:
        body_part: Body part name ('head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg')
        
    Returns:
        List of keypoint names for the specified body part
    """
    body_part_map = {
        'head': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
        'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
        'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
        'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
        'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
        'right_leg': ['right_hip', 'right_knee', 'right_ankle']
    }
    
    return body_part_map.get(body_part.lower(), [])


def calculate_keypoint_visibility_ratio(keypoints: Dict[str, Tuple[float, float, float]], 
                                       confidence_threshold: float = 0.5) -> float:
    """
    Calculate the ratio of visible keypoints above confidence threshold.
    
    Args:
        keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
        confidence_threshold: Minimum confidence for considering a keypoint visible
        
    Returns:
        Ratio of visible keypoints (0.0 to 1.0)
    """
    if not keypoints:
        return 0.0
    
    visible_count = sum(1 for x, y, conf in keypoints.values() 
                       if conf >= confidence_threshold)
    
    return visible_count / len(COCO_KEYPOINTS)