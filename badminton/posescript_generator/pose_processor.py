"""
Pose data processing module for converting input formats to PoseScript format.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional

try:
    from .models import PoseData
except ImportError:
    # Fallback for direct execution
    from models import PoseData


class PoseProcessor:
    """Handles pose data processing and format conversion for PoseScript integration."""
    
    # COCO keypoint indices (17 keypoints)
    COCO_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self):
        """Initialize the pose processor."""
        pass
    
    def process_coco_pose(self, pose_data: Union[Dict, List, np.ndarray]) -> PoseData:
        """
        Convert COCO format pose data to PoseScript input format.
        
        Args:
            pose_data: Pose data in COCO format. Can be:
                - Dict with 'keypoints' key containing list/array of coordinates
                - List/array of keypoint coordinates [x1, y1, v1, x2, y2, v2, ...]
                - 2D array of shape (17, 3) for COCO keypoints
        
        Returns:
            PoseData: Processed pose data ready for PoseScript
            
        Raises:
            ValueError: If pose data format is invalid
        """
        try:
            # Handle different input formats
            if isinstance(pose_data, dict):
                if 'keypoints' in pose_data:
                    keypoints_raw = pose_data['keypoints']
                else:
                    raise ValueError("Dictionary input must contain 'keypoints' key")
            else:
                keypoints_raw = pose_data
            
            # Convert to numpy array for easier processing
            if isinstance(keypoints_raw, list):
                keypoints_array = np.array(keypoints_raw)
            elif isinstance(keypoints_raw, np.ndarray):
                keypoints_array = keypoints_raw
            else:
                raise ValueError(f"Unsupported keypoints format: {type(keypoints_raw)}")
            
            # Handle different array shapes
            if keypoints_array.ndim == 1:
                # Flat array [x1, y1, v1, x2, y2, v2, ...]
                if len(keypoints_array) % 3 != 0:
                    raise ValueError("Flat keypoints array length must be divisible by 3")
                keypoints_array = keypoints_array.reshape(-1, 3)
            elif keypoints_array.ndim == 2:
                # 2D array (n_keypoints, 3)
                if keypoints_array.shape[1] != 3:
                    raise ValueError("2D keypoints array must have shape (n_keypoints, 3)")
            else:
                raise ValueError("Keypoints array must be 1D or 2D")
            
            # Validate COCO format (17 keypoints expected)
            if keypoints_array.shape[0] != 17:
                raise ValueError(f"COCO format requires 17 keypoints, got {keypoints_array.shape[0]}")
            
            # Convert to list of tuples (x, y, confidence)
            keypoints_list = [(float(kp[0]), float(kp[1]), float(kp[2])) 
                             for kp in keypoints_array]
            
            # Create metadata
            metadata = {
                'original_format': 'coco',
                'num_keypoints': len(keypoints_list),
                'keypoint_names': self.COCO_KEYPOINTS
            }
            
            # Add any additional metadata from input
            if isinstance(pose_data, dict):
                for key, value in pose_data.items():
                    if key != 'keypoints':
                        metadata[key] = value
            
            return PoseData(
                keypoints=keypoints_list,
                format="coco",
                metadata=metadata
            )
            
        except Exception as e:
            raise ValueError(f"Failed to process COCO pose data: {str(e)}")
    
    def validate_pose_data(self, pose_data: PoseData) -> bool:
        """
        Validate pose data format and content.
        
        Args:
            pose_data: PoseData object to validate
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            ValueError: If validation fails with specific error message
        """
        try:
            # Check if pose_data is PoseData instance
            if not isinstance(pose_data, PoseData):
                raise ValueError("Input must be a PoseData instance")
            
            # Check keypoints format
            if not isinstance(pose_data.keypoints, list):
                raise ValueError("Keypoints must be a list")
            
            if len(pose_data.keypoints) == 0:
                raise ValueError("Keypoints list cannot be empty")
            
            # Validate each keypoint
            for i, keypoint in enumerate(pose_data.keypoints):
                if not isinstance(keypoint, tuple) or len(keypoint) != 3:
                    raise ValueError(f"Keypoint {i} must be a tuple of length 3 (x, y, confidence)")
                
                x, y, z = keypoint
                
                # Check if coordinates are numeric
                if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
                    raise ValueError(f"Keypoint {i} coordinates must be numeric")
                
            # Format-specific validation
            if pose_data.format == "coco":
                if len(pose_data.keypoints) != 17:
                    raise ValueError("COCO format requires exactly 17 keypoints")
            
            # Check metadata
            if not isinstance(pose_data.metadata, dict):
                raise ValueError("Metadata must be a dictionary")
            
            return True
            
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")
    
    def normalize_pose(self, pose_data: PoseData) -> PoseData:
        """
        Normalize pose coordinates for consistent processing.
        
        Args:
            pose_data: PoseData object to normalize
            
        Returns:
            PoseData: Normalized pose data
        """
        try:
            # Validate input first
            self.validate_pose_data(pose_data)
            
            # Extract coordinates
            keypoints = pose_data.keypoints
            coords = np.array([(kp[0], kp[1]) for kp in keypoints])
            confidences = [kp[2] for kp in keypoints]
            
            # Filter out invisible keypoints (confidence = 0)
            visible_mask = np.array([conf > 0 for conf in confidences])
            
            if not np.any(visible_mask):
                # No visible keypoints, return original
                return pose_data
            
            visible_coords = coords[visible_mask]
            
            # Calculate bounding box of visible keypoints
            min_coords = np.min(visible_coords, axis=0)
            max_coords = np.max(visible_coords, axis=0)
            
            # Calculate center and scale
            center = (min_coords + max_coords) / 2
            scale = np.max(max_coords - min_coords)
            
            # Avoid division by zero
            if scale == 0:
                scale = 1.0
            
            # Normalize coordinates to [-1, 1] range centered at origin
            normalized_coords = (coords - center) / (scale / 2)
            
            # Create normalized keypoints
            normalized_keypoints = [
                (float(normalized_coords[i][0]), float(normalized_coords[i][1]), float(confidences[i]))
                for i in range(len(keypoints))
            ]
            
            # Update metadata
            new_metadata = pose_data.metadata.copy()
            new_metadata.update({
                'normalized': True,
                'original_center': center.tolist(),
                'original_scale': float(scale),
                'normalization_method': 'bounding_box'
            })
            
            return PoseData(
                keypoints=normalized_keypoints,
                format=pose_data.format,
                metadata=new_metadata
            )
            
        except Exception as e:
            raise ValueError(f"Failed to normalize pose data: {str(e)}")
    
    def convert_to_posescript_format(self, pose_data: PoseData) -> Dict[str, Any]:
        """
        Convert validated pose data to PoseScript expected format.
        
        Args:
            pose_data: Validated PoseData object
            
        Returns:
            Dict: Pose data in PoseScript expected format
        """
        try:
            # Validate input
            self.validate_pose_data(pose_data)
            
            # Extract keypoint coordinates and confidences
            keypoints_array = np.array(pose_data.keypoints)
            
            # PoseScript typically expects different format - this is a placeholder
            # The actual format would depend on PoseScript library requirements
            posescript_format = {
                'pose': keypoints_array[:, :2].tolist(),  # x, y coordinates
                'confidence': keypoints_array[:, 2].tolist(),  # confidence scores
                'format': pose_data.format,
                'metadata': pose_data.metadata
            }
            
            return posescript_format
            
        except Exception as e:
            raise ValueError(f"Failed to convert to PoseScript format: {str(e)}")


# Convenience functions for direct use
def process_coco_pose(pose_data: Union[Dict, List, np.ndarray]) -> PoseData:
    """Convenience function to process COCO pose data."""
    processor = PoseProcessor()
    return processor.process_coco_pose(pose_data)


def validate_pose_data(pose_data: PoseData) -> bool:
    """Convenience function to validate pose data."""
    processor = PoseProcessor()
    return processor.validate_pose_data(pose_data)


def normalize_pose(pose_data: PoseData) -> PoseData:
    """Convenience function to normalize pose data."""
    processor = PoseProcessor()
    return processor.normalize_pose(pose_data)