"""
Refactored VideoPoseDataset class that uses modular components.
This class serves as a high-level interface that coordinates the different modules.
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np

from .pose_data_loader import PoseDataLoader
from ..visualization.pose_visualizer import PoseVisualizer
from ..features.pose_feature_extractor import PoseFeatureExtractor
from ..analysis.shot_descriptor import ShotDescriptor
from ..utilities.utilities import get_video_metadata, get_video_frames


class VideoPoseDataset:
    """
    Refactored VideoPoseDataset that coordinates pose data loading, visualization, 
    feature extraction, and analysis using modular components.
    
    This class maintains the same interface as the original but delegates 
    responsibilities to specialized modules.
    """
    
    def __init__(self, poses_path: str | Path, video_path: str | Path):
        """
        Initialize the video pose dataset.
        
        Args:
            poses_path: Path to CSV file containing pose data
            video_path: Path to video file
        """
        self.poses_path = Path(poses_path)
        self.video_path = Path(video_path)
        
        # Initialize modular components
        self.data_loader = PoseDataLoader(poses_path)
        self.visualizer = PoseVisualizer()
        self.feature_extractor = PoseFeatureExtractor()
        self.shot_descriptor = ShotDescriptor()
        
        # Get video metadata
        self.frame_count, self.frames_per_second, self.duration, self.frame_shape = get_video_metadata(str(video_path))
        self.video_frames: Optional[List[np.ndarray]] = None
    
    def load_video_frames(self) -> None:
        """Load video frames into memory if not already loaded."""
        if self.video_frames is None:
            self.video_frames = get_video_frames(str(self.video_path))[4]
    
    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.data_loader)
    
    @property
    def playera(self) -> List[Tuple[List[float], Dict[str, Tuple[float, float, float]]]]:
        """Get player A data for backward compatibility."""
        return self.data_loader.playera
    
    @property
    def playerb(self) -> List[Tuple[List[float], Dict[str, Tuple[float, float, float]]]]:
        """Get player B data for backward compatibility."""
        return self.data_loader.playerb
    
    @property
    def poses_data(self) -> List[List[str]]:
        """Get raw poses data for backward compatibility."""
        return self.data_loader.poses_data
    
    def annotate_frame_with_poses(self, frame_idx: int, 
                                 include_background: bool = True, 
                                 include_bboxes: bool = True, 
                                 players: List[str] = ['green', 'blue']) -> np.ndarray:
        """
        Annotate a single frame with pose data.
        
        Args:
            frame_idx: Frame index to annotate
            include_background: Whether to include original video background
            include_bboxes: Whether to draw bounding boxes
            players: List of players to draw ('green' for A, 'blue' for B)
            
        Returns:
            Annotated frame
        """
        self.load_video_frames()
        frame_width, frame_height = self.frame_shape[1], self.frame_shape[0]
        
        if include_background:
            frame = self.video_frames[frame_idx].copy()
        else:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Get pose data for this frame
        player_a_data, player_b_data = self.data_loader.get_frame_data(frame_idx)
        frame_player_data = [player_a_data, player_b_data]
        
        # Use visualizer to annotate the frame
        return self.visualizer.annotate_frame(frame, frame_player_data, include_bboxes, players)
    
    def annotate_video_with_poses(self, output_path: str | Path, 
                                 include_background: bool = True,
                                 include_bboxes: bool = True, 
                                 players: List[str] = ['green', 'blue']) -> None:
        """
        Create an annotated video with pose overlays.
        
        Args:
            output_path: Path to save the annotated video
            include_background: Whether to include original video background
            include_bboxes: Whether to draw bounding boxes
            players: List of players to draw ('green' for A, 'blue' for B)
        """
        self.load_video_frames()
        
        # Prepare pose data for all frames
        pose_data_list = []
        for i in range(len(self.data_loader)):
            pose_data_list.append(self.data_loader.get_frame_data(i))
        
        # Use visualizer to create annotated video
        self.visualizer.create_annotated_video(
            self.video_frames, pose_data_list, str(output_path),
            include_background, include_bboxes, players, fps=30
        )
    
    def get_poselets_for_player(self, player: str = 'green') -> List[Dict[str, str]]:
        """
        Extract poselet features for the specified player.
        
        Args:
            player: 'green' for player A, 'blue' for player B
            
        Returns:
            List of dictionaries containing poselet features for each frame
        """
        player_data = self.data_loader.get_player_data(player)
        return self.feature_extractor.extract_poselets_for_player(player_data)
    
    def get_shot_description_for_player(self, player: str = 'green') -> str:
        """
        Generate a textual description of the player's pose for each frame.
        
        Args:
            player: 'green' for player A, 'blue' for player B
            
        Returns:
            Formatted prompt string with position and poselet data
        """
        # Get poselet features
        poselets = self.get_poselets_for_player(player)
        
        # Generate shot description using the shot descriptor
        return self.shot_descriptor.generate_shot_description(
            poselets, poses_path=self.poses_path
        )
    
    def analyze_shot_pattern(self, player: str = 'green') -> Dict[str, any]:
        """
        Analyze patterns in the shot data for insights.
        
        Args:
            player: 'green' for player A, 'blue' for player B
            
        Returns:
            Dictionary containing analysis results
        """
        poselets = self.get_poselets_for_player(player)
        return self.shot_descriptor.analyze_shot_pattern(poselets)
    
    # Additional convenience methods
    def get_frame_pose_data(self, frame_idx: int, player: str) -> Tuple[List[float], Dict[str, Tuple[float, float, float]]]:
        """
        Get pose data for a specific player at a specific frame.
        
        Args:
            frame_idx: Frame index
            player: 'green' for player A, 'blue' for player B
            
        Returns:
            Tuple of (bbox, keypoints) for the specified player and frame
        """
        player_data = self.data_loader.get_player_data(player)
        if frame_idx >= len(player_data):
            raise IndexError(f"Frame index {frame_idx} out of range")
        return player_data[frame_idx]
    
    def get_dataset_summary(self) -> Dict[str, any]:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            'poses_path': str(self.poses_path),
            'video_path': str(self.video_path),
            'frame_count': self.frame_count,
            'fps': self.frames_per_second,
            'duration': self.duration,
            'frame_shape': self.frame_shape,
            'total_frames': len(self.data_loader)
        }