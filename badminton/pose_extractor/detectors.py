"""
Pose detection and estimation module for badminton analysis.

This module provides classes for detecting people in images/videos and estimating
their poses using YOLO for detection and Sapiens for pose estimation.
"""

import os
import requests
import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from ultralytics import YOLO
from enum import Enum
from dataclasses import dataclass
from torchvision import transforms
import torch
from huggingface_hub import hf_hub_url

# Import from centralized config
from badminton.config import (
    YOLO_MODEL_PATH, 
    SAPIENS_1B_MODEL_PATH, 
    SAPIENS_2B_MODEL_PATH,
    MODEL_CONFIG,
    COCO_CONFIG
)
from badminton.utilities.utilities import get_video_frames, find_bboxes_closest_to_center

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Supported task types for Sapiens models."""
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"
    POSE = "pose"


class SapiensPoseEstimationType(Enum):
    """Available Sapiens pose estimation models."""
    COCO_POSE_ESTIMATION_1B = SAPIENS_1B_MODEL_PATH
    COCO_POSE_ESTIMATION_2B = SAPIENS_2B_MODEL_PATH


def download_file(url: str, filename: str) -> None:
    """
    Download a file from URL with progress bar.
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        
    Raises:
        requests.RequestException: If download fails
    """
    logger.info(f"Downloading {url} to {filename}")
    
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)
    
    logger.info(f"Download completed: {filename}")


def download_hf_model(model_name: str, task_type: TaskType, model_dir: str = 'models') -> str:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_name: Name of the model file
        task_type: Type of task (pose, depth, etc.)
        model_dir: Directory to save model to
        
    Returns:
        Path to downloaded model
        
    Raises:
        Exception: If download fails
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    path = os.path.join(model_dir, model_name)
    if os.path.exists(path):
        logger.info(f"Model {model_name} already exists at {path}")
        return path

    logger.info(f"Model {model_name} not found, downloading from Hugging Face Hub...")

    try:
        model_version = "_".join(model_name.split("_")[:2])
        repo_id = "facebook/sapiens"
        subdirectory = f"sapiens_lite_host/torchscript/{task_type.value}/checkpoints/{model_version}"

        url = hf_hub_url(repo_id=repo_id, filename=model_name, subfolder=subdirectory)
        download_file(url, path)
        logger.info(f"Model downloaded successfully to {path}")
        
        return path
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise


def create_preprocessor(input_size: Tuple[int, int],
                       mean: List[float] = [0.485, 0.456, 0.406],
                       std: List[float] = [0.229, 0.224, 0.225]) -> transforms.Compose:
    """
    Create image preprocessing pipeline.
    
    Args:
        input_size: Target image size (height, width)
        mean: Normalization mean values
        std: Normalization standard deviation values
        
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])


@dataclass
class DetectorConfig:
    """Configuration for person detector."""
    model_path: str = str(YOLO_MODEL_PATH)
    person_id: int = 0  # COCO person class ID
    conf_thres: float = MODEL_CONFIG["yolo"]["confidence_threshold"]
    iou_thres: float = MODEL_CONFIG["yolo"]["iou_threshold"]
    max_detections: int = MODEL_CONFIG["yolo"]["max_detections"]


class Detector:
    """
    Person detection using YOLO.
    
    This class handles detection of people in images using YOLO models.
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize the detector.
        
        Args:
            config: Detector configuration. If None, uses default config.
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if config is None:
            config = DetectorConfig()
            
        self.config = config
        
        # Validate model path
        model_path = config.model_path
        if not model_path.endswith(".pt"):
            model_path = model_path.split(".")[0] + ".pt"
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        self.person_id = config.person_id
        self.conf_thres = config.conf_thres

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Make the detector callable."""
        return self.detect(img)

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Detect people in an image.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Array of bounding boxes [x1, y1, x2, y2] for detected people
            
        Raises:
            ValueError: If image is invalid
        """
        if img is None or img.size == 0:
            raise ValueError("Invalid input image")
            
        start = time.perf_counter()
        
        try:
            results = self.model(img, conf=self.conf_thres)
            detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, cls)

            # Filter out only person detections
            person_detections = detections[detections[:, -1] == self.person_id]
            boxes = person_detections[:, :-2].astype(int)

            inference_time = time.perf_counter() - start
            logger.debug(f"Detection inference took: {inference_time:.4f} seconds")
            
            return boxes
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return np.array([])


@dataclass 
class SapiensPoseConfig:
    """Configuration for Sapiens pose estimation."""
    model_type: SapiensPoseEstimationType = SapiensPoseEstimationType.COCO_POSE_ESTIMATION_2B
    device: str = MODEL_CONFIG["sapiens"]["device"]
    input_size: Tuple[int, int] = MODEL_CONFIG["sapiens"]["input_size"]
    confidence_threshold: float = MODEL_CONFIG["sapiens"]["confidence_threshold"]
    dtype: str = "float32"


class SapiensPoseEstimation:
    """
    Pose estimation using Sapiens models.
    
    This class handles pose estimation for detected people using Sapiens models.
    """
    
    def __init__(self, config: Optional[SapiensPoseConfig] = None):
        """
        Initialize pose estimation.
        
        Args:
            config: Pose estimation configuration. If None, uses default config.
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if config is None:
            config = SapiensPoseConfig()
            
        self.config = config
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
            
        # Setup dtype
        self.dtype = getattr(torch, config.dtype)
        
        # Load model
        model_path = str(config.model_type.value)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Sapiens model not found: {model_path}")
            
        try:
            self.model = torch.jit.load(model_path).eval().to(self.device).to(self.dtype)
            logger.info(f"Loaded Sapiens model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load Sapiens model: {e}")
            raise
        
        # Setup preprocessor
        self.preprocessor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 768)),  # Sapiens expects this size
            transforms.ToTensor(),
        ])

        # Initialize the YOLO-based detector
        self.detector = Detector()

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[List[float]], List[Dict[str, Tuple[float, float, float]]]]:
        """
        Detect people and estimate their poses.
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Tuple of (all_bboxes, closest_bboxes, keypoints_list)
            
        Raises:
            ValueError: If image is invalid
        """
        if img is None or img.size == 0:
            raise ValueError("Invalid input image")
            
        start = time.perf_counter()

        # Detect persons in the image
        bboxes = self.detector.detect(img)
        
        if len(bboxes) < 2:
            logger.warning(f"Expected 2 people, found {len(bboxes)}")
            
        closest_bboxes = find_bboxes_closest_to_center(bboxes, img.shape[1] // 2)
        
        # Sort bboxes by y-coordinate (bottom player A, then top player B)
        if len(closest_bboxes) >= 2 and closest_bboxes[0][1] > closest_bboxes[1][1]:
            closest_bboxes = [closest_bboxes[1], closest_bboxes[0]]        
        
        # Process the image and estimate the pose
        keypoints = self.estimate_pose(img, closest_bboxes)

        inference_time = time.perf_counter() - start
        logger.debug(f"Pose estimation inference took: {inference_time:.4f} seconds")
        
        return bboxes, closest_bboxes, keypoints

    @torch.inference_mode()
    def estimate_pose(self, img: np.ndarray, bboxes: List[List[float]]) -> List[Dict[str, Tuple[float, float, float]]]:
        """
        Estimate poses for detected people.
        
        Args:
            img: Input image
            bboxes: List of bounding boxes
            
        Returns:
            List of keypoint dictionaries for each person
        """
        all_keypoints = []

        for bbox in bboxes:
            try:
                cropped_img = self._crop_image(img, bbox)
                tensor = self.preprocessor(cropped_img).unsqueeze(0).to(self.device).to(self.dtype)

                heatmaps = self.model(tensor)
                keypoints = self._heatmaps_to_keypoints(heatmaps[0].cpu().numpy())
                all_keypoints.append(keypoints)
                
            except Exception as e:
                logger.error(f"Failed to estimate pose for bbox {bbox}: {e}")
                # Return empty keypoints for this person
                empty_keypoints = {name: (0.0, 0.0, 0.0) for name in COCO_CONFIG["keypoint_names"]}
                all_keypoints.append(empty_keypoints)

        return all_keypoints

    def _crop_image(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Crop image to bounding box.
        
        Args:
            img: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Ensure coordinates are within image bounds
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))
        
        return img[y1:y2, x1:x2]

    def _heatmaps_to_keypoints(self, heatmaps: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """
        Convert heatmaps to keypoint coordinates.
        
        Args:
            heatmaps: Model output heatmaps
            
        Returns:
            Dictionary mapping keypoint names to (x, y, confidence) tuples
        """
        keypoints = {}
        keypoint_names = COCO_CONFIG["keypoint_names"]
        
        for i, name in enumerate(keypoint_names):
            if i < heatmaps.shape[0]:
                y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
                conf = float(heatmaps[i, y, x])
                keypoints[name] = (float(x), float(y), conf)
            else:
                keypoints[name] = (0.0, 0.0, 0.0)
                
        return keypoints
    
    def process_video(self, input_file: str) -> Tuple[List[np.ndarray], List[List[List[float]]], List[List[Dict[str, Tuple[float, float, float]]]]]:
        """
        Process entire video for pose estimation.
        
        Args:
            input_file: Path to input video file
            
        Returns:
            Tuple of (bboxes_list, closest_bboxes_list, keypoints_list) for all frames
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            Exception: If video processing fails
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Video file not found: {input_file}")
            
        try:
            _, _, _, _, frames = get_video_frames(input_file)
            logger.info(f"Processing {len(frames)} frames from {input_file}")
            
            bboxes_list = []
            closest_bboxes_list = []
            keypoints_list = []
            
            for count, img in enumerate(frames):
                logger.debug(f"Processing frame #{count + 1}")
                start_time = time.perf_counter()
                
                bboxes, closest_bboxes, keypoints = self.detect(img)
                bboxes_list.append(bboxes)
                closest_bboxes_list.append(closest_bboxes)
                keypoints_list.append(keypoints)
                
                frame_time = time.perf_counter() - start_time
                logger.debug(f"Frame {count + 1} processed in {frame_time:.4f} seconds")

            logger.info(f"Video processing completed: {len(frames)} frames")
            return bboxes_list, closest_bboxes_list, keypoints_list
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise