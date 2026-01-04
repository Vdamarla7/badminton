"""
Pose visualization functionality for drawing keypoints, bounding boxes, and creating annotated videos.
"""

import cv2
import imageio
import numpy as np
from typing import List, Dict, Tuple, Optional

from ..utilities.coco_keypoints import COCO_SKELETON_INFO
from ..config import VISUALIZATION_CONFIG


class PoseVisualizer:
    """
    Handles visualization of poses including keypoints, bounding boxes, and video annotation.
    
    This class is responsible for:
    - Drawing keypoints and skeleton connections
    - Drawing bounding boxes around players
    - Creating annotated video frames
    - Generating annotated videos
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pose visualizer.
        
        Args:
            config: Optional visualization configuration. Uses default if None.
        """
        self.config = config or VISUALIZATION_CONFIG
    
    def draw_bounding_box(self, img: np.ndarray, bbox: List[float], 
                         color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
        """
        Draw a bounding box on an image.
        
        Args:
            img: Input image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            color: RGB color tuple
            thickness: Line thickness
            
        Returns:
            Image with bounding box drawn
        """
        draw_img = img.copy()
        x1, y1, x2, y2 = map(int, bbox[:4])
        draw_img = cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)
        return draw_img
    
    def draw_keypoints(self, img: np.ndarray, bbox: List[float], 
                      keypoints: Dict[str, Tuple[float, float, float]], 
                      color: Tuple[int, int, int], confidence: float = 0.0) -> np.ndarray:
        """
        Draw keypoints and skeleton connections on an image.
        
        Args:
            img: Input image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            keypoints: Dictionary of keypoint names to (x, y, confidence) tuples
            color: RGB color tuple
            confidence: Minimum confidence threshold for drawing keypoints
            
        Returns:
            Image with keypoints and skeleton drawn
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        bbox_width, bbox_height = x2 - x1, y2 - y1
        img_copy = img.copy()

        # Draw keypoints on the image
        for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
            if conf > confidence:  # Only draw confident keypoints
                x_coord = int(x * bbox_width / 192) + x1
                y_coord = int(y * bbox_height / 256) + y1
                cv2.circle(img_copy, (x_coord, y_coord), 
                          self.config["keypoint_radius"], color, -1)

        # Draw skeleton connections
        for _, link_info in COCO_SKELETON_INFO.items():
            pt1_name, pt2_name = link_info['link']
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                if pt1[2] > confidence and pt2[2] > confidence:
                    x1_coord = int(pt1[0] * bbox_width / 192) + x1
                    y1_coord = int(pt1[1] * bbox_height / 256) + y1
                    x2_coord = int(pt2[0] * bbox_width / 192) + x1
                    y2_coord = int(pt2[1] * bbox_height / 256) + y1
                    cv2.line(img_copy, (x1_coord, y1_coord), (x2_coord, y2_coord), 
                            color, self.config["line_thickness"])

        return img_copy
    
    def annotate_frame(self, frame: np.ndarray, 
                      player_data: List[Tuple[List[float], Dict[str, Tuple[float, float, float]]]], 
                      include_bboxes: bool = True, 
                      players: List[str] = ['green', 'blue']) -> np.ndarray:
        """
        Annotate a single frame with pose data for specified players.
        
        Args:
            frame: Input video frame
            player_data: List of (bbox, keypoints) tuples for [player_a, player_b]
            include_bboxes: Whether to draw bounding boxes
            players: List of players to draw ('green' for A, 'blue' for B)
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw player A (green)
        if 'green' in players and len(player_data) > 0:
            color = self.config["colors"]["player_a"]
            bbox_a, keypoints_a = player_data[0]
            if include_bboxes:
                annotated_frame = self.draw_bounding_box(
                    annotated_frame, bbox_a, color=color, 
                    thickness=self.config["bbox_thickness"]
                )
            annotated_frame = self.draw_keypoints(
                annotated_frame, bbox_a, keypoints_a, color=color, confidence=0.0
            )

        # Draw player B (blue)
        if 'blue' in players and len(player_data) > 1:
            color = self.config["colors"]["player_b"]
            bbox_b, keypoints_b = player_data[1]
            if include_bboxes:
                annotated_frame = self.draw_bounding_box(
                    annotated_frame, bbox_b, color=color,
                    thickness=self.config["bbox_thickness"]
                )
            annotated_frame = self.draw_keypoints(
                annotated_frame, bbox_b, keypoints_b, color=color, confidence=0.0
            )
        
        return annotated_frame
    
    def create_annotated_video(self, video_frames: List[np.ndarray], 
                              pose_data_list: List[Tuple[
                                  Tuple[List[float], Dict[str, Tuple[float, float, float]]],
                                  Tuple[List[float], Dict[str, Tuple[float, float, float]]]
                              ]], 
                              output_path: str, 
                              include_background: bool = True,
                              include_bboxes: bool = True, 
                              players: List[str] = ['green', 'blue'],
                              fps: int = 30) -> None:
        """
        Create an annotated video with pose overlays.
        
        Args:
            video_frames: List of video frames
            pose_data_list: List of (player_a_data, player_b_data) tuples for each frame
            output_path: Path to save the annotated video
            include_background: Whether to include the original video background
            include_bboxes: Whether to draw bounding boxes
            players: List of players to draw ('green' for A, 'blue' for B)
            fps: Frames per second for output video
        """
        if not video_frames:
            raise ValueError("No video frames provided")
        
        frame_height, frame_width = video_frames[0].shape[:2]
        
        with imageio.get_writer(output_path, fps=fps) as writer:
            for i, frame in enumerate(video_frames):
                if i >= len(pose_data_list):
                    break
                
                if include_background:
                    annotated_frame = frame.copy()
                else:
                    annotated_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                
                # Get pose data for this frame
                player_a_data, player_b_data = pose_data_list[i]
                frame_player_data = [player_a_data, player_b_data]
                
                # Annotate the frame
                annotated_frame = self.annotate_frame(
                    annotated_frame, frame_player_data, include_bboxes, players
                )
                
                # Convert color space for video writer
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                writer.append_data(annotated_frame)