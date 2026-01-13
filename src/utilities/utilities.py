"""
Core utility functions for video and pose processing.

This module provides essential utilities for video file handling, metadata extraction,
and basic pose processing operations used throughout the badminton analysis system.
"""

import logging
import numpy as np
import cv2
from typing import Tuple, List, Optional
from pathlib import Path

from badminton.config import VIDEO_CONFIG

logger = logging.getLogger(__name__)


def get_video_metadata(video_path: str) -> Tuple[int, float, float, Optional[Tuple[int, int, int]]]:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (frame_count, fps, duration_seconds, frame_shape)
        frame_shape is (height, width, channels) or None if video can't be read
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file is invalid or corrupted
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            logger.warning(f"Invalid FPS ({fps}) detected, using default: {VIDEO_CONFIG['fps']}")
            fps = VIDEO_CONFIG['fps']
            
        duration = frame_count / fps if fps > 0 else 0
        
        # Try to read first frame to get shape
        ret, first_frame = cap.read()
        frame_shape = first_frame.shape if ret else None
        
        if not ret:
            logger.warning(f"Could not read first frame from {video_path}")
        
        logger.info(f"Video metadata - Frames: {frame_count}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        return frame_count, fps, duration, frame_shape
        
    except Exception as e:
        logger.error(f"Error extracting video metadata: {e}")
        raise ValueError(f"Failed to extract metadata from {video_path}: {e}")
    finally:
        cap.release()


def get_video_frames(video_path: str, 
                    max_frames: Optional[int] = None,
                    frame_skip: int = 1) -> Tuple[int, float, float, Optional[Tuple[int, int, int]], List[np.ndarray]]:
    """
    Load all frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to load (None for all)
        frame_skip: Skip every N frames (1 = load all frames)
        
    Returns:
        Tuple of (frame_count, fps, duration, frame_shape, frames_list)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file is invalid
        MemoryError: If video is too large to load into memory
    """
    # Get metadata first
    frame_count, fps, duration, frame_shape = get_video_metadata(video_path)
    
    # Apply configuration defaults
    if max_frames is None:
        max_frames = VIDEO_CONFIG.get('max_frames')
    if frame_skip is None:
        frame_skip = VIDEO_CONFIG.get('frame_skip', 1)
    
    # Estimate memory usage
    if frame_shape:
        estimated_frames = min(frame_count // frame_skip, max_frames or frame_count)
        estimated_memory_mb = (estimated_frames * np.prod(frame_shape) * 1) / (1024 * 1024)  # 1 byte per pixel
        
        if estimated_memory_mb > 1000:  # Warn if > 1GB
            logger.warning(f"Loading {estimated_frames} frames will use ~{estimated_memory_mb:.1f}MB of memory")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    frame_idx = 0
    loaded_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply frame skipping
            if frame_idx % frame_skip == 0:
                frames.append(frame)
                loaded_count += 1
                
                # Check max frames limit
                if max_frames and loaded_count >= max_frames:
                    break
            
            frame_idx += 1
        
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
        return frame_count, fps, duration, frame_shape, frames
        
    except MemoryError:
        logger.error(f"Not enough memory to load video frames from {video_path}")
        raise MemoryError(f"Video too large to load into memory: {video_path}")
    except Exception as e:
        logger.error(f"Error loading video frames: {e}")
        raise ValueError(f"Failed to load frames from {video_path}: {e}")
    finally:
        cap.release()


def find_bboxes_closest_to_center(bboxes: np.ndarray, 
                                 center_x: float, 
                                 num_bboxes: int = 2) -> List[List[float]]:
    """
    Find bounding boxes closest to a center point.
    
    This function is typically used to identify the two main players
    in a badminton court by finding the two people closest to the center.
    
    Args:
        bboxes: Array of bounding boxes with shape (N, 4+) where each row is [x1, y1, x2, y2, ...]
        center_x: X-coordinate of the center point (usually image width / 2)
        num_bboxes: Number of closest bboxes to return
        
    Returns:
        List of the closest bounding boxes as lists [x1, y1, x2, y2]
        
    Raises:
        ValueError: If bboxes array is invalid or empty
    """
    if bboxes is None or len(bboxes) == 0:
        logger.warning("No bounding boxes provided")
        return []
    
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes)
    
    if bboxes.shape[1] < 4:
        raise ValueError(f"Bounding boxes must have at least 4 columns, got {bboxes.shape[1]}")
    
    # Calculate the midpoints of the bounding boxes (center of bottom edge)
    midpoints = [(bbox[2] + bbox[0]) / 2 for bbox in bboxes]
    
    # Calculate distances from center
    distances = [abs(midpoint - center_x) for midpoint in midpoints]
    
    # Get indices of the closest bounding boxes
    closest_indices = np.argsort(distances)[:num_bboxes]
    
    # Return the closest bounding boxes as lists
    closest_bboxes = [bboxes[i][:4].tolist() for i in closest_indices]
    
    logger.debug(f"Found {len(closest_bboxes)} closest bboxes to center {center_x}")
    return closest_bboxes


def validate_video_file(video_path: str) -> bool:
    """
    Validate that a video file exists and can be opened.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid, False otherwise
    """
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            return False
        
        cap = cv2.VideoCapture(str(video_path))
        is_valid = cap.isOpened()
        cap.release()
        
        return is_valid
        
    except Exception:
        return False


def resize_frame(frame: np.ndarray, 
                scale_factor: float = 1.0,
                target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Resize a video frame.
    
    Args:
        frame: Input frame as numpy array
        scale_factor: Scaling factor (ignored if target_size is provided)
        target_size: Target size as (width, height)
        
    Returns:
        Resized frame
        
    Raises:
        ValueError: If frame is invalid
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame provided")
    
    if target_size:
        width, height = target_size
        return cv2.resize(frame, (width, height))
    elif scale_factor != 1.0:
        height, width = frame.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(frame, (new_width, new_height))
    else:
        return frame


def get_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
    """
    Extract a single frame at a specific time.
    
    Args:
        video_path: Path to video file
        time_seconds: Time in seconds
        
    Returns:
        Frame at specified time, or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(time_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        return frame if ret else None
        
    except Exception as e:
        logger.error(f"Failed to extract frame at {time_seconds}s: {e}")
        return None
    closest_bboxes = bboxes[closest_indices]
    return closest_bboxes