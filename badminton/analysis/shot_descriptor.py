"""
Shot description functionality for generating textual descriptions of player poses and shots.
"""

import io
import csv
from typing import List, Dict, Optional
from pathlib import Path

from ..features.pose_feature_extractor import PoseFeatureExtractor


class ShotDescriptor:
    """
    Handles generation of textual descriptions for badminton shots based on pose data.
    
    This class is responsible for:
    - Inferring shot location from file paths or other context
    - Generating CSV-formatted poselet descriptions
    - Creating prompts for LLM analysis
    """
    
    def __init__(self):
        """Initialize the shot descriptor."""
        self.feature_extractor = PoseFeatureExtractor()
        
        # Shot type to location mapping
        self.shot_location_map = {
            "00_Short_Serve": "ServeLine",
            "13_Long_Serve": "ServeLine", 
            "14_Smash": "BackCourt or MidCourt",
            "05_Drop_Shot": "FrontCourt",
            "07_Transitional_Slice": "BackCourt",
            "16_Rear_Court_Flat_Drive": "BackCourt"
        }
    
    def infer_location_from_path(self, poses_path: str | Path) -> str:
        """
        Infer the court location based on the file path.
        
        Args:
            poses_path: Path to the poses CSV file
            
        Returns:
            Inferred court location string
        """
        poses_path = Path(poses_path)
        
        # Extract shot type from path (assuming it's in the parent directory name)
        try:
            shot_type = poses_path.parent.name
            return self.shot_location_map.get(shot_type, "MidCourt")
        except (AttributeError, IndexError):
            return "MidCourt"  # Default location
    
    def generate_shot_description(self, 
                                 player_data: List[Dict[str, str]], 
                                 location: Optional[str] = None,
                                 poses_path: Optional[str | Path] = None) -> str:
        """
        Generate a textual description of the player's shot based on poselet data.
        
        Args:
            player_data: List of poselet dictionaries for each frame
            location: Court location string. If None, will try to infer from poses_path
            poses_path: Path to poses file (used for location inference if location is None)
            
        Returns:
            Formatted prompt string with position and poselet data
        """
        # Determine location
        if location is None:
            if poses_path is not None:
                location = self.infer_location_from_path(poses_path)
            else:
                location = "MidCourt"  # Default
        
        # Generate CSV-formatted poselet data
        poselet_csv = self._generate_poselet_csv(player_data)
        
        # Create the prompt
        prompt = f"Position: {location}\n{poselet_csv}"
        return prompt
    
    def _generate_poselet_csv(self, player_data: List[Dict[str, str]]) -> str:
        """
        Generate CSV-formatted string of poselet data.
        
        Args:
            player_data: List of poselet dictionaries for each frame
            
        Returns:
            CSV-formatted string with headers and poselet data
        """
        output = io.StringIO()
        fieldnames = ['Frame', 'left_arm', 'left_leg', 'left_torso', 'right_arm', 'right_leg', 'right_torso']

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        # Add frame numbers and write rows
        for i, row in enumerate(player_data):
            row_with_frame = row.copy()
            row_with_frame['Frame'] = i + 1
            writer.writerow(row_with_frame)

        return output.getvalue()
    
    def analyze_shot_pattern(self, player_data: List[Dict[str, str]]) -> Dict[str, any]:
        """
        Analyze patterns in the shot data to provide insights.
        
        Args:
            player_data: List of poselet dictionaries for each frame
            
        Returns:
            Dictionary containing analysis results
        """
        if not player_data:
            return {"error": "No data provided"}
        
        # Get summary from feature extractor
        summary = self.feature_extractor.get_poselet_summary(player_data)
        
        # Add shot-specific analysis
        analysis = {
            "frame_count": len(player_data),
            "poselet_summary": summary,
            "shot_phases": self._identify_shot_phases(player_data)
        }
        
        return analysis
    
    def _identify_shot_phases(self, player_data: List[Dict[str, str]]) -> List[str]:
        """
        Identify different phases of the shot based on poselet changes.
        
        Args:
            player_data: List of poselet dictionaries for each frame
            
        Returns:
            List of identified shot phases
        """
        if len(player_data) < 3:
            return ["incomplete_shot"]
        
        phases = []
        
        # Simple heuristic: look for changes in arm positions
        prev_arm_state = None
        for frame_data in player_data:
            current_arm_state = (frame_data.get('left_arm', ''), frame_data.get('right_arm', ''))
            
            if prev_arm_state is None:
                phases.append("preparation")
            elif current_arm_state != prev_arm_state:
                phases.append("transition")
            else:
                phases.append("hold")
            
            prev_arm_state = current_arm_state
        
        return phases