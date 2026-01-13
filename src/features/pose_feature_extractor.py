"""
Pose feature extraction functionality for extracting poselets and other pose-based features.
"""

from typing import List, Dict, Tuple, Any

try:
    from ..utilities.poselet_classifier import classify_triplet
except (ImportError, ModuleNotFoundError):
    # Fallback for testing or when sklearn is not available
    from ..utilities.poselet_classifier_mock import classify_triplet


class PoseFeatureExtractor:
    """
    Handles extraction of pose-based features including poselets.
    
    This class is responsible for:
    - Extracting poselet features from keypoint data
    - Computing pose orientations for different body parts
    - Generating feature vectors for analysis
    """
    
    def __init__(self):
        """Initialize the pose feature extractor."""
        pass
    
    def extract_poselets_for_player(self, 
                                   player_data: List[Tuple[List[float], Dict[str, Tuple[float, float, float]]]]) -> List[Dict[str, str]]:
        """
        Extract poselet features for a player across all frames.
        
        Args:
            player_data: List of (bbox, keypoints) tuples for each frame
            
        Returns:
            List of dictionaries containing poselet features for each frame
        """
        extracted_features = []
        
        for bbox, keypoints in player_data:
            frame_features = self._extract_frame_poselets(keypoints)
            extracted_features.append(frame_features)
            
        return extracted_features
    
    def _extract_frame_poselets(self, keypoints: Dict[str, Tuple[float, float, float]]) -> Dict[str, str]:
        """
        Extract poselet features for a single frame.
        
        Args:
            keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
            
        Returns:
            Dictionary containing poselet classifications for different body parts
        """
        # Extract keypoint coordinates (ignoring confidence for poselet classification)
        left_ankle = (keypoints['left_ankle'][0], keypoints['left_ankle'][1])
        left_knee = (keypoints['left_knee'][0], keypoints['left_knee'][1])
        left_hip = (keypoints['left_hip'][0], keypoints['left_hip'][1])
        left_shoulder = (keypoints['left_shoulder'][0], keypoints['left_shoulder'][1])
        left_elbow = (keypoints['left_elbow'][0], keypoints['left_elbow'][1])
        left_wrist = (keypoints['left_wrist'][0], keypoints['left_wrist'][1])
        
        right_wrist = (keypoints['right_wrist'][0], keypoints['right_wrist'][1])
        right_elbow = (keypoints['right_elbow'][0], keypoints['right_elbow'][1])
        right_shoulder = (keypoints['right_shoulder'][0], keypoints['right_shoulder'][1])
        right_hip = (keypoints['right_hip'][0], keypoints['right_hip'][1])
        right_knee = (keypoints['right_knee'][0], keypoints['right_knee'][1])
        right_ankle = (keypoints['right_ankle'][0], keypoints['right_ankle'][1])

        # Classify orientations for different body parts
        left_arm_orientation = classify_triplet(left_shoulder, left_elbow, left_wrist)
        left_leg_orientation = classify_triplet(left_hip, left_knee, left_ankle)
        left_torso_orientation = classify_triplet(left_knee, left_hip, left_shoulder)
        
        right_arm_orientation = classify_triplet(right_shoulder, right_elbow, right_wrist)
        right_leg_orientation = classify_triplet(right_hip, right_knee, right_ankle)
        right_torso_orientation = classify_triplet(right_knee, right_hip, right_shoulder)

        return {
            'left_arm': left_arm_orientation,
            'left_leg': left_leg_orientation,
            'left_torso': left_torso_orientation,
            'right_arm': right_arm_orientation,
            'right_leg': right_leg_orientation,
            'right_torso': right_torso_orientation
        }
    
    def get_poselet_summary(self, poselets: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate a summary of poselet features across all frames.
        
        Args:
            poselets: List of poselet dictionaries for each frame
            
        Returns:
            Summary statistics and common patterns
        """
        if not poselets:
            return {}
        
        # Count occurrences of each poselet type
        poselet_counts = {}
        for frame_poselets in poselets:
            for body_part, orientation in frame_poselets.items():
                if body_part not in poselet_counts:
                    poselet_counts[body_part] = {}
                if orientation not in poselet_counts[body_part]:
                    poselet_counts[body_part][orientation] = 0
                poselet_counts[body_part][orientation] += 1
        
        # Find most common poselet for each body part
        most_common = {}
        for body_part, orientations in poselet_counts.items():
            most_common[body_part] = max(orientations.items(), key=lambda x: x[1])
        
        return {
            'total_frames': len(poselets),
            'poselet_counts': poselet_counts,
            'most_common_poselets': most_common
        }