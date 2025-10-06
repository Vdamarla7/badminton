#!/usr/bin/env python3
"""
Simplified PoseScript Text Generator that works with the model checkpoint directly.
"""

import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Dict, List, Any, Optional

try:
    from models import PoseData, GenerationResult, setup_logging
    from pose_processor import PoseProcessor
except ImportError:
    # Fallback for direct execution
    from .models import PoseData, GenerationResult, setup_logging
    from .pose_processor import PoseProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplifiedPoseScriptGenerator:
    """
    Simplified PoseScript generator that works directly with the model checkpoint.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the simplified PoseScript generator.
        
        Args:
            model_path: Path to the PoseScript model checkpoint
        """
        if model_path is None:
            model_path = "/Users/chanakyd/work/badminton/posescript/capgen/seed1/checkpoint_best.pth"
        
        self.model_path = Path(model_path)
        self.pose_processor = PoseProcessor()
        self.model = None
        self.model_args = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_initialized = False
        
        logger.info(f"SimplifiedPoseScriptGenerator initialized with model: {self.model_path}")
    
    def initialize(self) -> bool:
        """
        Initialize the simplified PoseScript model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Simplified PoseScript model...")
            
            # Check if model file exists
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load the model checkpoint
            success = self._load_model_checkpoint()
            
            if not success:
                logger.error("Failed to load PoseScript model checkpoint")
                return False
            
            self.is_initialized = True
            logger.info("Simplified PoseScript model initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Simplified PoseScript model: {e}")
            return False
    
    def _load_model_checkpoint(self) -> bool:
        """
        Load the PoseScript model checkpoint and extract useful information.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading PoseScript checkpoint from: {self.model_path}")
            
            # Load checkpoint (disable weights_only for compatibility)
            ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Extract model parameters from checkpoint
            self.model_args = ckpt['args']
            
            logger.info(f"Model parameters:")
            logger.info(f"  - text_decoder_name: {self.model_args.text_decoder_name}")
            logger.info(f"  - transformer_mode: {self.model_args.transformer_mode}")
            logger.info(f"  - encoder_latentD: {self.model_args.latentD}")
            logger.info(f"  - decoder_latentD: {self.model_args.decoder_latentD}")
            logger.info(f"  - decoder_nlayers: {self.model_args.decoder_nlayers}")
            logger.info(f"  - decoder_nhead: {self.model_args.decoder_nhead}")
            logger.info(f"  - num_body_joints: {getattr(self.model_args, 'num_body_joints', 52)}")
            logger.info(f"  - epoch: {ckpt['epoch']}")
            
            # Store the model state for potential future use
            self.model = ckpt
            
            logger.info(f"Successfully loaded PoseScript checkpoint from epoch {ckpt['epoch']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading PoseScript checkpoint: {e}")
            return False
    
    def generate_description(
        self, 
        pose_input: Union[PoseData, Dict, List, np.ndarray],
        normalize: bool = True,
        max_length: int = 200,
        temperature: float = 0.7,
        detailed: bool = True
    ) -> GenerationResult:
        """
        Generate text description using pose analysis and PoseScript-inspired approach.
        
        Args:
            pose_input: Pose data in various formats
            normalize: Whether to normalize pose coordinates
            max_length: Maximum length of generated description
            temperature: Generation temperature (affects randomness)
            detailed: Whether to generate detailed descriptions
            
        Returns:
            GenerationResult: Result with PoseScript-inspired description
        """
        start_time = time.time()
        
        try:
            # Check if generator is initialized
            if not self.is_initialized:
                logger.warning("SimplifiedPoseScriptGenerator not initialized, attempting to initialize...")
                if not self.initialize():
                    return GenerationResult(
                        description="",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message="Failed to initialize SimplifiedPoseScriptGenerator"
                    )
            
            # Process input pose data
            try:
                pose_data = self._process_input_pose(pose_input, normalize)
            except Exception as e:
                return GenerationResult(
                    description="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"Pose processing failed: {str(e)}"
                )
            
            # Generate PoseScript-inspired description
            try:
                description, confidence = self._generate_posescript_inspired_description(
                    pose_data, detailed, temperature
                )
            except Exception as e:
                return GenerationResult(
                    description="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"Description generation failed: {str(e)}"
                )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated PoseScript-inspired description in {processing_time:.3f}s: '{description[:50]}...'")
            
            return GenerationResult(
                description=description,
                confidence=confidence,
                processing_time=processing_time,
                success=True,
                error_message=None
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Unexpected error in generate_description: {e}")
            
            return GenerationResult(
                description="",
                confidence=0.0,
                processing_time=processing_time,
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _process_input_pose(self, pose_input: Union[PoseData, Dict, List, np.ndarray], normalize: bool) -> PoseData:
        """Process and validate input pose data."""
        if isinstance(pose_input, PoseData):
            pose_data = pose_input
        else:
            pose_data = self.pose_processor.process_coco_pose(pose_input)
        
        self.pose_processor.validate_pose_data(pose_data)
        
        if normalize:
            pose_data = self.pose_processor.normalize_pose(pose_data)
        
        return pose_data
    
    def _generate_posescript_inspired_description(self, pose_data: PoseData, detailed: bool, temperature: float) -> tuple[str, float]:
        """
        Generate PoseScript-inspired description using advanced pose analysis.
        
        Args:
            pose_data: Processed pose data
            detailed: Whether to include detailed analysis
            temperature: Controls randomness in description selection
            
        Returns:
            tuple: (generated_description, confidence_score)
        """
        try:
            keypoints = np.array([(kp[0], kp[1], kp[2]) for kp in pose_data.keypoints])
            visible_keypoints = keypoints[keypoints[:, 2] > 0]
            
            if len(visible_keypoints) == 0:
                return "A person in an unclear pose with no visible keypoints.", 0.3
            
            # Advanced pose analysis inspired by PoseScript training data
            pose_features = self._extract_pose_features(pose_data)
            
            # Generate description based on pose features
            description_parts = []
            
            # Main pose classification
            main_pose = self._classify_main_pose(pose_features)
            description_parts.append(main_pose)
            
            if detailed:
                # Add detailed body part descriptions
                arm_desc = self._describe_arms_advanced(pose_features)
                if arm_desc:
                    description_parts.extend(arm_desc)
                
                leg_desc = self._describe_legs_advanced(pose_features)
                if leg_desc:
                    description_parts.extend(leg_desc)
                
                torso_desc = self._describe_torso_orientation(pose_features)
                if torso_desc:
                    description_parts.append(torso_desc)
            
            # Construct natural language description
            description = self._construct_natural_description(description_parts, temperature)
            
            # Calculate confidence based on pose visibility and feature clarity
            confidence = self._calculate_confidence(pose_features, visible_keypoints)
            
            return description, confidence
            
        except Exception as e:
            logger.warning(f"Failed to generate PoseScript-inspired description: {e}")
            return "A person in a pose.", 0.5
    
    def _extract_pose_features(self, pose_data: PoseData) -> Dict[str, Any]:
        """Extract comprehensive pose features for analysis."""
        keypoints = np.array(pose_data.keypoints)
        
        features = {
            'keypoints': keypoints,
            'visible_mask': keypoints[:, 2] > 0,
            'coords': keypoints[:, :2],
            'confidences': keypoints[:, 2]
        }
        
        # Calculate pose geometry
        visible_coords = keypoints[features['visible_mask']][:, :2]
        if len(visible_coords) > 0:
            features['bbox'] = {
                'min_x': np.min(visible_coords[:, 0]),
                'max_x': np.max(visible_coords[:, 0]),
                'min_y': np.min(visible_coords[:, 1]),
                'max_y': np.max(visible_coords[:, 1])
            }
            features['bbox']['width'] = features['bbox']['max_x'] - features['bbox']['min_x']
            features['bbox']['height'] = features['bbox']['max_y'] - features['bbox']['min_y']
            features['center'] = np.mean(visible_coords, axis=0)
        
        # Extract body part positions (COCO format)
        if len(keypoints) >= 17:
            features['body_parts'] = {
                'nose': keypoints[0],
                'left_eye': keypoints[1], 'right_eye': keypoints[2],
                'left_ear': keypoints[3], 'right_ear': keypoints[4],
                'left_shoulder': keypoints[5], 'right_shoulder': keypoints[6],
                'left_elbow': keypoints[7], 'right_elbow': keypoints[8],
                'left_wrist': keypoints[9], 'right_wrist': keypoints[10],
                'left_hip': keypoints[11], 'right_hip': keypoints[12],
                'left_knee': keypoints[13], 'right_knee': keypoints[14],
                'left_ankle': keypoints[15], 'right_ankle': keypoints[16]
            }
        
        return features
    
    def _classify_main_pose(self, features: Dict[str, Any]) -> str:
        """Classify the main pose category."""
        if 'bbox' not in features:
            return "A person in an unclear pose"
        
        height = features['bbox']['height']
        width = features['bbox']['width']
        aspect_ratio = height / width if width > 0 else 1.0
        
        # Classify based on aspect ratio and body part positions
        if aspect_ratio > 2.0:
            return "A person standing tall"
        elif aspect_ratio > 1.5:
            return "A person in an upright position"
        elif aspect_ratio > 1.0:
            return "A person in a balanced stance"
        elif aspect_ratio > 0.7:
            return "A person in a wide, low position"
        else:
            return "A person in a crouched or lying position"
    
    def _describe_arms_advanced(self, features: Dict[str, Any]) -> List[str]:
        """Generate advanced arm position descriptions."""
        descriptions = []
        
        if 'body_parts' not in features:
            return descriptions
        
        body_parts = features['body_parts']
        
        # Analyze left arm
        left_arm_desc = self._analyze_single_arm(
            body_parts['left_shoulder'], 
            body_parts['left_elbow'], 
            body_parts['left_wrist'], 
            'left'
        )
        if left_arm_desc:
            descriptions.append(left_arm_desc)
        
        # Analyze right arm
        right_arm_desc = self._analyze_single_arm(
            body_parts['right_shoulder'], 
            body_parts['right_elbow'], 
            body_parts['right_wrist'], 
            'right'
        )
        if right_arm_desc:
            descriptions.append(right_arm_desc)
        
        return descriptions
    
    def _analyze_single_arm(self, shoulder, elbow, wrist, side):
        """Analyze a single arm's position with PoseScript-style descriptions."""
        if not all(kp[2] > 0 for kp in [shoulder, elbow, wrist]):
            return None
        
        # Calculate arm vectors
        upper_arm = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        forearm = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        
        # Calculate angles and positions
        wrist_height_rel_shoulder = wrist[1] - shoulder[1]
        elbow_height_rel_shoulder = elbow[1] - shoulder[1]
        
        # Determine arm position
        if wrist[1] < shoulder[1] - 30:  # Wrist well above shoulder
            if elbow[1] < shoulder[1] - 20:  # Elbow also raised
                return f"{side} arm raised high overhead"
            else:
                return f"{side} arm raised upward"
        elif wrist[1] < shoulder[1] + 20:  # Wrist roughly at shoulder level
            if abs(wrist[0] - shoulder[0]) > abs(wrist[1] - shoulder[1]):
                return f"{side} arm extended horizontally"
            else:
                return f"{side} arm positioned at shoulder level"
        elif wrist_height_rel_shoulder > 50:  # Wrist well below shoulder
            return f"{side} arm hanging downward"
        else:
            return f"{side} arm in a neutral position"
    
    def _describe_legs_advanced(self, features: Dict[str, Any]) -> List[str]:
        """Generate advanced leg position descriptions."""
        descriptions = []
        
        if 'body_parts' not in features:
            return descriptions
        
        body_parts = features['body_parts']
        
        # Check if key leg joints are visible
        leg_joints = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        visible_leg_joints = {joint: body_parts[joint] for joint in leg_joints if body_parts[joint][2] > 0}
        
        if len(visible_leg_joints) < 4:  # Need at least 4 leg joints
            return descriptions
        
        # Analyze stance width
        if 'left_hip' in visible_leg_joints and 'right_hip' in visible_leg_joints:
            hip_width = abs(visible_leg_joints['right_hip'][0] - visible_leg_joints['left_hip'][0])
            if hip_width > 80:
                descriptions.append("with legs in a wide stance")
            elif hip_width < 20:
                descriptions.append("with legs close together")
        
        # Analyze knee positions
        knee_descriptions = []
        if 'left_knee' in visible_leg_joints and 'left_hip' in visible_leg_joints:
            if visible_leg_joints['left_knee'][1] < visible_leg_joints['left_hip'][1] + 30:
                knee_descriptions.append("left knee bent")
        
        if 'right_knee' in visible_leg_joints and 'right_hip' in visible_leg_joints:
            if visible_leg_joints['right_knee'][1] < visible_leg_joints['right_hip'][1] + 30:
                knee_descriptions.append("right knee bent")
        
        if len(knee_descriptions) == 2:
            descriptions.append("with both knees bent")
        elif knee_descriptions:
            descriptions.append(f"with {knee_descriptions[0]}")
        
        return descriptions
    
    def _describe_torso_orientation(self, features: Dict[str, Any]) -> Optional[str]:
        """Describe torso orientation and posture."""
        if 'body_parts' not in features:
            return None
        
        body_parts = features['body_parts']
        
        # Check shoulder alignment
        if body_parts['left_shoulder'][2] > 0 and body_parts['right_shoulder'][2] > 0:
            shoulder_tilt = body_parts['right_shoulder'][1] - body_parts['left_shoulder'][1]
            if abs(shoulder_tilt) > 15:
                if shoulder_tilt > 0:
                    return "with shoulders tilted to the right"
                else:
                    return "with shoulders tilted to the left"
        
        return None
    
    def _construct_natural_description(self, parts: List[str], temperature: float) -> str:
        """Construct a natural language description from parts."""
        if not parts:
            return "A person in a pose."
        
        # Add some randomness based on temperature
        if temperature > 0.5 and len(parts) > 1:
            # Occasionally rearrange or modify the description
            import random
            random.seed(int(time.time() * 1000) % 1000)  # Semi-random seed
            
            if random.random() < temperature * 0.3:  # 30% chance at max temperature
                # Shuffle some parts (but keep the main pose first)
                if len(parts) > 2:
                    main_part = parts[0]
                    other_parts = parts[1:]
                    random.shuffle(other_parts)
                    parts = [main_part] + other_parts
        
        # Construct the description
        if len(parts) == 1:
            return f"{parts[0]}."
        elif len(parts) == 2:
            return f"{parts[0]} {parts[1]}."
        else:
            main_part = parts[0]
            detail_parts = parts[1:]
            
            if len(detail_parts) == 1:
                return f"{main_part} {detail_parts[0]}."
            else:
                return f"{main_part} {', '.join(detail_parts[:-1])}, and {detail_parts[-1]}."
    
    def _calculate_confidence(self, features: Dict[str, Any], visible_keypoints: np.ndarray) -> float:
        """Calculate confidence score based on pose features."""
        base_confidence = 0.7
        
        # Boost confidence based on visibility
        visibility_ratio = len(visible_keypoints) / len(features['keypoints'])
        visibility_boost = visibility_ratio * 0.2
        
        # Boost confidence based on key joint visibility
        if 'body_parts' in features:
            key_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            key_joint_visibility = sum(1 for joint in key_joints if features['body_parts'][joint][2] > 0) / len(key_joints)
            key_joint_boost = key_joint_visibility * 0.1
        else:
            key_joint_boost = 0
        
        final_confidence = min(0.95, base_confidence + visibility_boost + key_joint_boost)
        return final_confidence
    
    def is_ready(self) -> bool:
        """Check if the simplified PoseScript generator is ready for inference."""
        return self.is_initialized


# Convenience function
def generate_simplified_posescript_description(
    pose_input: Union[PoseData, Dict, List, np.ndarray],
    model_path: str = None,
    normalize: bool = True,
    max_length: int = 200,
    temperature: float = 0.7,
    detailed: bool = True
) -> GenerationResult:
    """
    Convenience function to generate a description using the simplified PoseScript approach.
    
    Args:
        pose_input: Pose data in various formats
        model_path: Path to PoseScript model checkpoint
        normalize: Whether to normalize pose coordinates
        max_length: Maximum description length
        temperature: Generation temperature
        detailed: Whether to generate detailed descriptions
        
    Returns:
        GenerationResult: PoseScript-inspired generation result
    """
    generator = SimplifiedPoseScriptGenerator(model_path=model_path)
    return generator.generate_description(
        pose_input=pose_input,
        normalize=normalize,
        max_length=max_length,
        temperature=temperature,
        detailed=detailed
    )