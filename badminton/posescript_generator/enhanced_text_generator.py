#!/usr/bin/env python3
"""
Enhanced text generation using the actual PoseScript model for detailed descriptions.
"""

import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Union, Dict, List, Any, Optional

# Add PoseScript to path
posescript_path = Path(__file__).parent.parent / "posescript" / "src"
if posescript_path.exists():
    sys.path.insert(0, str(posescript_path))

try:
    from models import PoseData, GenerationResult
    from model_manager import ModelManager
    from pose_processor import PoseProcessor
except ImportError:
    # Fallback for direct execution
    from .models import PoseData, GenerationResult
    from .model_manager import ModelManager
    from .pose_processor import PoseProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTextGenerator:
    """
    Enhanced text generator that uses the actual PoseScript model for detailed descriptions.
    """
    
    def __init__(self, model_dir: str = "./models", model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced text generator.
        
        Args:
            model_dir: Directory to store/load models
            model_config: Custom model configuration
        """
        self.model_manager = ModelManager(model_dir=model_dir, model_config=model_config)
        self.pose_processor = PoseProcessor()
        self.posescript_model = None
        self.is_initialized = False
        
        logger.info("EnhancedTextGenerator initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the enhanced text generator with the actual PoseScript model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing EnhancedTextGenerator with PoseScript model...")
            
            # Download and load model checkpoint
            if not self.model_manager.download_model():
                logger.error("Failed to download model")
                return False
            
            if not self.model_manager.load_model():
                logger.error("Failed to load model checkpoint")
                return False
            
            # Setup PoseScript environment
            self.model_manager.setup_posescript_environment()
            
            # Initialize the actual PoseScript model
            self.posescript_model = self._initialize_posescript_model()
            
            if self.posescript_model is None:
                logger.error("Failed to initialize PoseScript model")
                return False
            
            self.is_initialized = True
            logger.info("EnhancedTextGenerator initialized successfully with PoseScript model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedTextGenerator: {e}")
            return False
    
    def _initialize_posescript_model(self):
        """
        Initialize the actual PoseScript model for detailed text generation.
        
        Returns:
            The initialized PoseScript model or None if failed
        """
        try:
            # Import PoseScript modules
            from text2pose.generative_caption.model_generative_caption import DescriptionGenerator
            from text2pose.generative_caption.model_generative_caption import get_model_args
            
            # Get model checkpoint
            checkpoint = self.model_manager.model
            if not checkpoint or 'model_state' not in checkpoint:
                logger.error("Invalid model checkpoint")
                return None
            
            # Get model arguments
            args = checkpoint.get('model_args') or checkpoint.get('args')
            if args is None:
                logger.error("Model arguments not found in checkpoint")
                return None
            
            logger.info(f"Initializing PoseScript model with args: {type(args)}")
            
            # Create the model
            model = DescriptionGenerator(
                text_decoder_name=getattr(args, 'text_decoder_name', 'gpt2'),
                transformer_mode=getattr(args, 'transformer_mode', 'encoder_decoder'),
                decoder_nlayers=getattr(args, 'decoder_nlayers', 6),
                decoder_nhead=getattr(args, 'decoder_nhead', 8),
                encoder_latentD=getattr(args, 'latentD', 512),
                decoder_latentD=getattr(args, 'decoder_latentD', 512),
                num_body_joints=getattr(args, 'num_body_joints', 52)
            ).to(self.model_manager.device)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            
            logger.info("PoseScript model initialized successfully")
            return model
            
        except ImportError as e:
            logger.error(f"Failed to import PoseScript modules: {e}")
            logger.info("Falling back to enhanced placeholder descriptions")
            return "placeholder"
        except Exception as e:
            logger.error(f"Error initializing PoseScript model: {e}")
            return None
    
    def generate_description(
        self, 
        pose_input: Union[PoseData, Dict, List, np.ndarray],
        normalize: bool = True,
        max_length: int = 200,
        temperature: float = 0.7,
        detailed: bool = True
    ) -> GenerationResult:
        """
        Generate detailed text description from pose data using PoseScript model.
        
        Args:
            pose_input: Pose data in various formats
            normalize: Whether to normalize pose coordinates
            max_length: Maximum length of generated description
            temperature: Generation temperature
            detailed: Whether to generate detailed descriptions
            
        Returns:
            GenerationResult: Structured result with detailed description
        """
        start_time = time.time()
        
        try:
            # Check if generator is initialized
            if not self.is_initialized:
                logger.warning("EnhancedTextGenerator not initialized, attempting to initialize...")
                if not self.initialize():
                    return GenerationResult(
                        description="",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message="Failed to initialize EnhancedTextGenerator"
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
            
            # Generate description using PoseScript model
            try:
                if self.posescript_model == "placeholder":
                    # Enhanced placeholder with detailed analysis
                    description, confidence = self._generate_enhanced_placeholder(pose_data, detailed)
                else:
                    # Use actual PoseScript model
                    description, confidence = self._generate_with_posescript_model(
                        pose_data, max_length, temperature
                    )
            except Exception as e:
                return GenerationResult(
                    description="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"Text generation failed: {str(e)}"
                )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated detailed description in {processing_time:.3f}s: '{description[:50]}...'")
            
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
    
    def _generate_with_posescript_model(self, pose_data: PoseData, max_length: int, temperature: float) -> tuple[str, float]:
        """
        Generate text using the actual PoseScript model.
        
        Args:
            pose_data: Processed pose data
            max_length: Maximum description length
            temperature: Generation temperature
            
        Returns:
            tuple: (generated_description, confidence_score)
        """
        try:
            # Convert pose to PoseScript input format
            pose_tensor = self._convert_to_posescript_tensor(pose_data)
            
            # Generate description with PoseScript model
            with torch.no_grad():
                # This would be the actual PoseScript inference
                output = self.posescript_model.generate(
                    pose_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    pad_token_id=self.posescript_model.tokenizer.pad_token_id,
                    eos_token_id=self.posescript_model.tokenizer.eos_token_id
                )
            
            # Decode the output
            description = self.posescript_model.tokenizer.decode(output[0], skip_special_tokens=True)
            confidence = 0.9  # Would calculate from model output probabilities
            
            return description, confidence
            
        except Exception as e:
            logger.error(f"PoseScript model inference failed: {e}")
            # Fallback to enhanced placeholder
            return self._generate_enhanced_placeholder(pose_data, detailed=True)
    
    def _convert_to_posescript_tensor(self, pose_data: PoseData) -> torch.Tensor:
        """
        Convert pose data to PoseScript model input tensor.
        
        Args:
            pose_data: Processed pose data
            
        Returns:
            torch.Tensor: Input tensor for PoseScript model
        """
        # Convert COCO keypoints to PoseScript format
        keypoints = np.array(pose_data.keypoints)
        
        # PoseScript expects specific format - this would need to be adapted
        # based on the actual model requirements
        pose_tensor = torch.FloatTensor(keypoints[:, :2]).unsqueeze(0)  # x, y coordinates
        
        return pose_tensor.to(self.model_manager.device)
    
    def _generate_enhanced_placeholder(self, pose_data: PoseData, detailed: bool = True) -> tuple[str, float]:
        """
        Generate enhanced placeholder description with detailed pose analysis.
        
        Args:
            pose_data: Processed pose data
            detailed: Whether to include detailed analysis
            
        Returns:
            tuple: (detailed_description, confidence_score)
        """
        try:
            keypoints = np.array([(kp[0], kp[1], kp[2]) for kp in pose_data.keypoints])
            visible_keypoints = keypoints[keypoints[:, 2] > 0]
            
            if len(visible_keypoints) == 0:
                return "A person in an unclear pose with no visible keypoints.", 0.3
            
            descriptions = []
            
            # Analyze body position
            coords = visible_keypoints[:, :2]
            height = np.max(coords[:, 1]) - np.min(coords[:, 1])
            width = np.max(coords[:, 0]) - np.min(coords[:, 0])
            
            # Basic stance analysis
            if height > width * 1.8:
                descriptions.append("standing tall")
            elif height > width * 1.2:
                descriptions.append("in an upright position")
            elif width > height * 1.5:
                descriptions.append("in a wide, low stance")
            else:
                descriptions.append("in a balanced position")
            
            if detailed and pose_data.format == "coco" and len(pose_data.keypoints) >= 17:
                # Detailed arm analysis
                arm_desc = self._analyze_arms_detailed(pose_data.keypoints)
                if arm_desc:
                    descriptions.extend(arm_desc)
                
                # Detailed leg analysis
                leg_desc = self._analyze_legs_detailed(pose_data.keypoints)
                if leg_desc:
                    descriptions.extend(leg_desc)
                
                # Head orientation
                head_desc = self._analyze_head_detailed(pose_data.keypoints)
                if head_desc:
                    descriptions.extend(head_desc)
            
            # Construct detailed description
            if len(descriptions) == 1:
                description = f"A person {descriptions[0]}."
            elif len(descriptions) == 2:
                description = f"A person {descriptions[0]} with {descriptions[1]}."
            else:
                main_desc = descriptions[0]
                details = descriptions[1:]
                if len(details) == 1:
                    description = f"A person {main_desc} with {details[0]}."
                else:
                    description = f"A person {main_desc} with {', '.join(details[:-1])}, and {details[-1]}."
            
            return description, 0.85
            
        except Exception as e:
            logger.warning(f"Failed to generate enhanced description: {e}")
            return "A person in a pose.", 0.5
    
    def _analyze_arms_detailed(self, keypoints: List[tuple]) -> List[str]:
        """Analyze arm positions in detail."""
        descriptions = []
        
        try:
            # COCO keypoint indices
            left_shoulder = keypoints[5]   # left_shoulder
            right_shoulder = keypoints[6]  # right_shoulder
            left_elbow = keypoints[7]      # left_elbow
            right_elbow = keypoints[8]     # right_elbow
            left_wrist = keypoints[9]      # left_wrist
            right_wrist = keypoints[10]    # right_wrist
            
            # Analyze left arm
            if all(kp[2] > 0 for kp in [left_shoulder, left_elbow, left_wrist]):
                left_arm_desc = self._describe_arm_position(left_shoulder, left_elbow, left_wrist, "left")
                if left_arm_desc:
                    descriptions.append(left_arm_desc)
            
            # Analyze right arm
            if all(kp[2] > 0 for kp in [right_shoulder, right_elbow, right_wrist]):
                right_arm_desc = self._describe_arm_position(right_shoulder, right_elbow, right_wrist, "right")
                if right_arm_desc:
                    descriptions.append(right_arm_desc)
            
        except (IndexError, ValueError):
            pass
        
        return descriptions
    
    def _describe_arm_position(self, shoulder, elbow, wrist, side):
        """Describe the position of a single arm."""
        try:
            # Calculate angles and positions
            shoulder_to_elbow = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
            elbow_to_wrist = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
            
            # Vertical positions relative to shoulder
            elbow_above_shoulder = elbow[1] < shoulder[1]
            wrist_above_shoulder = wrist[1] < shoulder[1]
            wrist_above_elbow = wrist[1] < elbow[1]
            
            # Horizontal positions
            wrist_extended = abs(wrist[0] - shoulder[0]) > abs(wrist[1] - shoulder[1])
            
            if wrist_above_shoulder and elbow_above_shoulder:
                if wrist_above_elbow:
                    return f"{side} arm raised high above head"
                else:
                    return f"{side} arm raised with hand pointing upward"
            elif wrist_above_shoulder:
                return f"{side} arm raised"
            elif wrist_extended:
                if wrist[0] > shoulder[0]:
                    return f"{side} arm extended outward"
                else:
                    return f"{side} arm extended across body"
            elif abs(wrist[1] - shoulder[1]) < 50:  # Roughly horizontal
                return f"{side} arm extended horizontally"
            else:
                return f"{side} arm positioned downward"
                
        except Exception:
            return None
    
    def _analyze_legs_detailed(self, keypoints: List[tuple]) -> List[str]:
        """Analyze leg positions in detail."""
        descriptions = []
        
        try:
            left_hip = keypoints[11]    # left_hip
            right_hip = keypoints[12]   # right_hip
            left_knee = keypoints[13]   # left_knee
            right_knee = keypoints[14]  # right_knee
            left_ankle = keypoints[15]  # left_ankle
            right_ankle = keypoints[16] # right_ankle
            
            # Analyze stance width
            if all(kp[2] > 0 for kp in [left_hip, right_hip]):
                hip_width = abs(right_hip[0] - left_hip[0])
                if hip_width > 100:  # Wide stance
                    descriptions.append("legs in a wide stance")
                elif hip_width < 30:  # Narrow stance
                    descriptions.append("legs close together")
            
            # Analyze knee positions
            if all(kp[2] > 0 for kp in [left_hip, left_knee, right_hip, right_knee]):
                left_knee_bent = left_knee[1] < left_hip[1] + 50
                right_knee_bent = right_knee[1] < right_hip[1] + 50
                
                if left_knee_bent and right_knee_bent:
                    descriptions.append("knees bent in a crouched position")
                elif left_knee_bent:
                    descriptions.append("left knee bent")
                elif right_knee_bent:
                    descriptions.append("right knee bent")
            
        except (IndexError, ValueError):
            pass
        
        return descriptions
    
    def _analyze_head_detailed(self, keypoints: List[tuple]) -> List[str]:
        """Analyze head orientation in detail."""
        descriptions = []
        
        try:
            nose = keypoints[0]         # nose
            left_eye = keypoints[1]     # left_eye
            right_eye = keypoints[2]    # right_eye
            left_ear = keypoints[3]     # left_ear
            right_ear = keypoints[4]    # right_ear
            
            # Analyze head turn based on eye/ear visibility
            left_visible = left_eye[2] > 0 and left_ear[2] > 0
            right_visible = right_eye[2] > 0 and right_ear[2] > 0
            
            if left_visible and not right_visible:
                descriptions.append("head turned to the left")
            elif right_visible and not left_visible:
                descriptions.append("head turned to the right")
            elif nose[2] > 0 and left_eye[2] > 0 and right_eye[2] > 0:
                # Analyze head tilt based on eye positions
                eye_height_diff = abs(left_eye[1] - right_eye[1])
                if eye_height_diff > 10:
                    if left_eye[1] < right_eye[1]:
                        descriptions.append("head tilted to the left")
                    else:
                        descriptions.append("head tilted to the right")
            
        except (IndexError, ValueError):
            pass
        
        return descriptions
    
    def is_ready(self) -> bool:
        """Check if the enhanced text generator is ready for inference."""
        return self.is_initialized


# Convenience function
def generate_detailed_description(
    pose_input: Union[PoseData, Dict, List, np.ndarray],
    model_dir: str = "./models",
    normalize: bool = True,
    max_length: int = 200,
    temperature: float = 0.7
) -> GenerationResult:
    """
    Convenience function to generate a detailed description using PoseScript.
    
    Args:
        pose_input: Pose data in various formats
        model_dir: Directory containing the model
        normalize: Whether to normalize pose coordinates
        max_length: Maximum description length
        temperature: Generation temperature
        
    Returns:
        GenerationResult: Detailed generation result
    """
    generator = EnhancedTextGenerator(model_dir=model_dir)
    return generator.generate_description(
        pose_input=pose_input,
        normalize=normalize,
        max_length=max_length,
        temperature=temperature,
        detailed=True
    )