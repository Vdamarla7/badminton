"""
Shared utilities for video processing and file operations.
"""

import os
import cv2
from pathlib import Path
from typing import List, Tuple, Optional


def replace_mp4_with_csv(filename: str) -> str:
    """
    Replace .mp4 extension with .csv extension.
    
    Args:
        filename: Input filename
        
    Returns:
        Filename with .csv extension
    """
    name, ext = os.path.splitext(filename)
    
    if ext.lower() == '.mp4':
        return f"{name}.csv"
    else:
        return filename


def list_video_files(folder_path: str | Path, 
                    video_extensions: Optional[List[str]] = None) -> List[str]:
    """
    List all video files in a directory and its subdirectories.
    
    Args:
        folder_path: Directory to search
        video_extensions: List of video file extensions to look for
        
    Returns:
        List of relative paths to video files
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv']

    files_list = []
    folder_path = Path(folder_path)
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                # Test if video file is readable
                full_path = os.path.join(root, file)
                cap = cv2.VideoCapture(full_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:  # Only include readable video files
                    parent_dir = os.path.basename(root)
                    relative_path = os.path.join(parent_dir, file)
                    files_list.append(relative_path)
                
    return files_list


def get_video_info(video_path: str | Path) -> Tuple[int, float, float, Tuple[int, int]]:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (frame_count, fps, duration, (height, width))
    """
    cap = cv2.VideoCapture(str(video_path))
    
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return frame_count, fps, duration, (height, width)
    
    finally:
        cap.release()


def validate_video_file(video_path: str | Path) -> bool:
    """
    Check if a video file is valid and readable.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        return ret
    except Exception:
        return False


def extract_frame_at_time(video_path: str | Path, 
                         time_seconds: float) -> Optional[Tuple[bool, any]]:
    """
    Extract a single frame from video at specified time.
    
    Args:
        video_path: Path to video file
        time_seconds: Time in seconds to extract frame
        
    Returns:
        Tuple of (success, frame) or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(time_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        return (ret, frame) if ret else None
    
    finally:
        cap.release()


def resize_frame(frame, scale_factor: float = 1.0, 
                target_size: Optional[Tuple[int, int]] = None):
    """
    Resize a video frame.
    
    Args:
        frame: Input frame
        scale_factor: Scaling factor (ignored if target_size is provided)
        target_size: Target (width, height) tuple
        
    Returns:
        Resized frame
    """
    if target_size is not None:
        return cv2.resize(frame, target_size)
    elif scale_factor != 1.0:
        height, width = frame.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(frame, (new_width, new_height))
    else:
        return frame