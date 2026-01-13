"""
Pose data loading and CSV parsing functionality.
Handles loading pose data from CSV files and organizing it by player.
"""

import csv
from typing import List, Tuple, Dict, Any
from pathlib import Path

from ..utilities.coco_keypoints import create_keypoints_dict


class PoseDataLoader:
    """
    Handles loading and parsing pose data from CSV files.
    
    This class is responsible for:
    - Reading CSV files containing pose data
    - Parsing bounding boxes and keypoints for each player
    - Organizing data by frame and player
    """
    
    def __init__(self, poses_path: str | Path):
        """
        Initialize the pose data loader.
        
        Args:
            poses_path: Path to the CSV file containing pose data
        """
        self.poses_path = Path(poses_path)
        self.poses_data: List[List[str]] = []
        self.playera: List[Tuple[List[float], Dict[str, Tuple[float, float, float]]]] = []
        self.playerb: List[Tuple[List[float], Dict[str, Tuple[float, float, float]]]] = []
        
        self._load_poses_data()
        self._parse_player_data()
    
    def _load_poses_data(self) -> None:
        """Load raw pose data from CSV file."""
        poses_data = []
        with open(self.poses_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header row if present
            for row in csvreader:
                poses_data.append(row)
        
        self.poses_data = poses_data
    
    def _parse_player_data(self) -> None:
        """Parse player bounding boxes and keypoints from raw data."""
        playera = []
        playerb = []
        
        for row in self.poses_data:
            # Parse bounding boxes
            playera_bbox = list(map(float, row[1:5]))
            playerb_bbox = list(map(float, row[5:9]))
            
            # Parse keypoints
            playera_keypoints = create_keypoints_dict(list(map(float, row[9:60])))
            playerb_keypoints = create_keypoints_dict(list(map(float, row[60:])))
            
            playera.append((playera_bbox, playera_keypoints))
            playerb.append((playerb_bbox, playerb_keypoints))
        
        self.playera = playera
        self.playerb = playerb
    
    def get_player_data(self, player: str) -> List[Tuple[List[float], Dict[str, Tuple[float, float, float]]]]:
        """
        Get pose data for a specific player.
        
        Args:
            player: 'green' for player A, 'blue' for player B
            
        Returns:
            List of tuples containing (bbox, keypoints) for each frame
        """
        if player == 'green':
            return self.playera
        elif player == 'blue':
            return self.playerb
        else:
            raise ValueError(f"Invalid player '{player}'. Must be 'green' or 'blue'.")
    
    def get_frame_data(self, frame_idx: int) -> Tuple[
        Tuple[List[float], Dict[str, Tuple[float, float, float]]],
        Tuple[List[float], Dict[str, Tuple[float, float, float]]]
    ]:
        """
        Get pose data for both players at a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Tuple of (player_a_data, player_b_data)
        """
        if frame_idx >= len(self.poses_data):
            raise IndexError(f"Frame index {frame_idx} out of range")
        
        return self.playera[frame_idx], self.playerb[frame_idx]
    
    def __len__(self) -> int:
        """Return the number of frames in the dataset."""
        return len(self.poses_data)
    
    @property
    def frame_count(self) -> int:
        """Get the total number of frames."""
        return len(self.poses_data)