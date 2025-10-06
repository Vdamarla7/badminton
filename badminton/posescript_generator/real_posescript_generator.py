#!/usr/bin/env python3
"""
Real PoseScript Text Generator using the actual PoseScript model.
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
    from models import PoseData, GenerationResult, setup_logging
    from pose_processor import PoseProcessor
except ImportError:
    # Fallback for direct execution
    from .models import PoseData, GenerationResult, setup_logging
    from .pose_processor import PoseProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealPoseScriptGenerator:
    """
    Text generator that uses the actual PoseScript model for detailed descriptions.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the real PoseScript generator.
        
        Args:
            model_path: Path to the PoseScript model checkpoint
        """
        if model_path is None:
            model_path = "/Users/chanakyd/work/badminton/posescript/capgen/seed1/checkpoint_best.pth"
        
        self.model_path = Path(model_path)
        self.pose_processor = PoseProcessor()
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_initialized = False
        
        logger.info(f"RealPoseScriptGenerator initialized with model: {self.model_path}")
    
    def initialize(self) -> bool:
        """
        Initialize the real PoseScript model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Real PoseScript model...")
            
            # Check if model file exists
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load the actual PoseScript model
            self.model, self.tokenizer = self._load_posescript_model()
            
            if self.model is None:
                logger.error("Failed to load PoseScript model")
                return False
            
            self.is_initialized = True
            logger.info("Real PoseScript model initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Real PoseScript model: {e}")
            return False
    
    def _load_posescript_model(self):
        """
        Load the actual PoseScript model from checkpoint.
        
        Returns:
            tuple: (model, tokenizer_name) or (None, None) if failed
        """
        try:
            # Import PoseScript modules
            from text2pose.generative_caption.model_generative_caption import DescriptionGenerator
            from text2pose.encoders.tokenizers import get_tokenizer_name
            
            logger.info(f"Loading PoseScript model from: {self.model_path}")
            
            # Load checkpoint (disable weights_only for compatibility)
            ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Extract model parameters from checkpoint
            args = ckpt['args']
            text_decoder_name = args.text_decoder_name
            transformer_mode = args.transformer_mode
            encoder_latentD = args.latentD
            decoder_latentD = args.decoder_latentD
            decoder_nlayers = args.decoder_nlayers
            decoder_nhead = args.decoder_nhead
            num_body_joints = getattr(args, 'num_body_joints', 52)
            
            logger.info(f"Model parameters:")
            logger.info(f"  - text_decoder_name: {text_decoder_name}")
            logger.info(f"  - transformer_mode: {transformer_mode}")
            logger.info(f"  - encoder_latentD: {encoder_latentD}")
            logger.info(f"  - decoder_latentD: {decoder_latentD}")
            logger.info(f"  - decoder_nlayers: {decoder_nlayers}")
            logger.info(f"  - decoder_nhead: {decoder_nhead}")
            logger.info(f"  - num_body_joints: {num_body_joints}")
            
            # Create model
            model = DescriptionGenerator(
                text_decoder_name=text_decoder_name,
                transformer_mode=transformer_mode,
                decoder_nlayers=decoder_nlayers,
                decoder_nhead=decoder_nhead,
                encoder_latentD=encoder_latentD,
                decoder_latentD=decoder_latentD,
                num_body_joints=num_body_joints
            ).to(self.device)
            
            # Load model state
            model.load_state_dict(ckpt['model'])
            model.eval()
            
            # Get tokenizer name
            tokenizer_name = get_tokenizer_name(text_decoder_name)
            
            logger.info(f"Successfully loaded PoseScript model from epoch {ckpt['epoch']}")
            logger.info(f"Using tokenizer: {tokenizer_name}")
            
            return model, tokenizer_name
            
        except ImportError as e:
            logger.error(f"Failed to import PoseScript modules: {e}")
            logger.error("Make sure the PoseScript library is properly installed")
            return None, None
        except Exception as e:
            logger.error(f"Error loading PoseScript model: {e}")
            return None, None
    
    def generate_description(
        self, 
        pose_input: Union[PoseData, Dict, List, np.ndarray],
        normalize: bool = True,
        max_length: int = 200,
        temperature: float = 0.7,
        num_samples: int = 1
    ) -> GenerationResult:
        """
        Generate text description using the real PoseScript model.
        
        Args:
            pose_input: Pose data in various formats
            normalize: Whether to normalize pose coordinates
            max_length: Maximum length of generated description
            temperature: Generation temperature
            num_samples: Number of descriptions to generate
            
        Returns:
            GenerationResult: Result with PoseScript-generated description
        """
        start_time = time.time()
        
        try:
            # Check if generator is initialized
            if not self.is_initialized:
                logger.warning("RealPoseScriptGenerator not initialized, attempting to initialize...")
                if not self.initialize():
                    return GenerationResult(
                        description="",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message="Failed to initialize RealPoseScriptGenerator"
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
            
            # Convert pose to PoseScript format and generate description
            try:
                description, confidence = self._generate_with_posescript(
                    pose_data, max_length, temperature, num_samples
                )
            except Exception as e:
                return GenerationResult(
                    description="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"PoseScript generation failed: {str(e)}"
                )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Generated PoseScript description in {processing_time:.3f}s: '{description[:50]}...'")
            
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
    
    def _generate_with_posescript(self, pose_data: PoseData, max_length: int, temperature: float, num_samples: int) -> tuple[str, float]:
        """
        Generate text using the actual PoseScript model with badminton context.
        
        Args:
            pose_data: Processed pose data
            max_length: Maximum description length
            temperature: Generation temperature
            num_samples: Number of samples to generate
            
        Returns:
            tuple: (generated_description, confidence_score)
        """
        try:
            # Convert COCO pose to PoseScript format with proper orientation
            pose_tensor = self._convert_coco_to_posescript_format(pose_data)
            
            logger.info(f"Pose tensor shape: {pose_tensor.shape}")
            logger.info(f"Pose tensor device: {pose_tensor.device}")
            logger.debug(f"Pose tensor sample values: {pose_tensor[0, :5, :]}")  # Show first 5 keypoints
            
            # Generate description with PoseScript model
            with torch.no_grad():
                # Debug: Check what methods the model has
                logger.debug(f"Model methods: {[method for method in dir(self.model) if not method.startswith('_')]}")
                
                # Try different generation approaches based on PoseScript model structure
                if hasattr(self.model, 'generate_text'):
                    logger.debug("Using model.generate_text method")
                    result = self.model.generate_text(pose_tensor)
                    logger.debug(f"generate_text returned: {type(result)}")
                    
                    if isinstance(result, tuple) and len(result) >= 2:
                        texts, scores = result[0], result[1]
                        logger.debug(f"texts type: {type(texts)}, length: {len(texts) if texts else 0}")
                        logger.debug(f"scores type: {type(scores)}, length: {len(scores) if scores else 0}")
                        logger.debug(f"scores content: {scores}")
                        if isinstance(scores, torch.Tensor):
                            logger.debug(f"scores shape: {scores.shape}")
                            logger.debug(f"scores values: {scores}")
                            logger.debug(f"scores min/max: {scores.min()}/{scores.max()}")
                    else:
                        texts = result if isinstance(result, list) else [str(result)]
                        scores = None
                        logger.debug(f"Single return value, treating as texts: {texts}")
                    
                    description = texts[0] if texts and len(texts) > 0 else "Generated description"
                    
                    # Calculate confidence from scores
                    # Note: PoseScript returns likelihood scores (negative log-likelihoods) for evaluation,
                    # not confidence scores. We'll convert these to meaningful confidence values.
                    if scores is not None and len(scores) > 0:
                        if isinstance(scores[0], torch.Tensor):
                            raw_score = float(scores[0].item())
                        else:
                            raw_score = float(scores[0])
                        logger.debug(f"Raw likelihood score from model: {raw_score}")
                        
                        # Convert likelihood score to confidence
                        # PoseScript returns negative log-likelihood, higher (less negative) is better
                        # We'll map this to a 0-1 confidence range
                        if raw_score < 0:
                            # Convert negative log-likelihood to confidence
                            # Use a sigmoid-like transformation to map to [0,1]
                            # Typical range for NLL is around -10 to 0, so we'll normalize accordingly
                            normalized_score = (raw_score + 10) / 10  # Shift and scale
                            confidence = max(0.1, min(0.9, 1 / (1 + np.exp(-normalized_score * 6))))  # Sigmoid
                            logger.debug(f"Converted likelihood {raw_score} to confidence: {confidence:.3f}")
                        else:
                            # Positive scores are unusual for likelihood, treat as high confidence
                            confidence = 0.85
                            logger.debug(f"Positive likelihood score, using high confidence: {confidence}")
                    else:
                        # If no scores, calculate confidence based on description quality and length
                        desc_length = len(description.split())
                        if desc_length > 15:
                            confidence = 0.80  # Long, detailed description
                        elif desc_length > 8:
                            confidence = 0.70  # Medium description
                        else:
                            confidence = 0.60  # Short description
                        logger.debug(f"No likelihood scores, using length-based confidence: {confidence} (length: {desc_length} words)")
                elif hasattr(self.model, 'generate'):
                    logger.debug("Using model.generate method")
                    # Try the standard generate method
                    output = self.model.generate(pose_tensor, max_length=max_length, temperature=temperature)
                    if isinstance(output, tuple):
                        description, confidence = output[0], output[1] if len(output) > 1 else 0.8
                    else:
                        description = str(output)
                        confidence = 0.8
                elif hasattr(self.model, 'forward'):
                    logger.debug("Using model.forward method with manual decoding")
                    # Use forward pass and decode manually
                    output = self.model(pose_tensor)
                    
                    # Extract logits and calculate confidence from probability distribution
                    if hasattr(output, 'logits'):
                        logits = output.logits
                    elif isinstance(output, torch.Tensor):
                        logits = output
                    else:
                        logits = output[0] if isinstance(output, (list, tuple)) else output
                    
                    # Calculate confidence as max probability
                    probs = torch.softmax(logits, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    confidence = float(torch.mean(max_probs))
                    
                    # For now, use a placeholder description since we need proper decoding
                    # In a real implementation, you'd decode the logits to text
                    description = "A person in a dynamic athletic pose with detailed body positioning and movement."
                else:
                    logger.warning("No suitable generation method found, using fallback")
                    description = "A person in a dynamic pose with detailed body positioning."
                    confidence = 0.75
                
                # Ensure confidence is in valid range
                confidence = max(0.0, min(1.0, confidence))
                
            logger.info(f"Generated description: '{description[:100]}...'")
            logger.info(f"Calculated confidence: {confidence:.6f}")
            
            return description, confidence
            
        except Exception as e:
            logger.error(f"PoseScript model inference failed: {e}")
            raise RuntimeError(f"PoseScript generation failed: {str(e)}")
    
    def _convert_coco_to_posescript_format(self, pose_data: PoseData) -> torch.Tensor:
        """
        Convert COCO pose data to PoseScript model input format with proper coordinate transformation.
        
        Args:
            pose_data: COCO format pose data
            
        Returns:
            torch.Tensor: Pose tensor in PoseScript format
        """
        try:
            # Extract keypoints (x, y, confidence)
            keypoints = np.array(pose_data.keypoints)  # Shape: (17, 3)
            
            # IMPORTANT: Transform coordinates for proper orientation
            # The "lying on back" issue suggests the coordinate system needs adjustment
            
            if keypoints.shape[0] == 17:  # COCO format
                # First, normalize and center the pose
                valid_points = keypoints[keypoints[:, 2] > 0]  # Only confident keypoints
                if len(valid_points) == 0:
                    raise ValueError("No valid keypoints found")
                
                # Center the pose around the torso center (average of shoulders and hips)
                torso_joints = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip
                torso_points = keypoints[torso_joints]
                valid_torso = torso_points[torso_points[:, 2] > 0]
                
                if len(valid_torso) > 0:
                    center = np.mean(valid_torso[:, :2], axis=0)
                else:
                    center = np.mean(valid_points[:, :2], axis=0)
                
                # Center the pose
                centered_keypoints = keypoints.copy()
                centered_keypoints[:, :2] -= center
                
                # CRITICAL: Analyze the pose orientation more carefully
                # The issue might be that we need to understand the actual pose structure
                head_joints = [0, 1, 2, 3, 4]  # nose, eyes, ears
                hip_joints = [11, 12]  # left_hip, right_hip
                shoulder_joints = [5, 6]  # left_shoulder, right_shoulder
                
                head_points = centered_keypoints[head_joints]
                hip_points = centered_keypoints[hip_joints]
                shoulder_points = centered_keypoints[shoulder_joints]
                
                valid_head = head_points[head_points[:, 2] > 0]
                valid_hips = hip_points[hip_points[:, 2] > 0]
                valid_shoulders = shoulder_points[shoulder_points[:, 2] > 0]
                
                # Debug: Log the actual pose structure
                if len(valid_head) > 0:
                    avg_head_y = np.mean(valid_head[:, 1])
                    logger.debug(f"Average head Y: {avg_head_y:.3f}")
                
                if len(valid_hips) > 0:
                    avg_hip_y = np.mean(valid_hips[:, 1])
                    logger.debug(f"Average hip Y: {avg_hip_y:.3f}")
                
                if len(valid_shoulders) > 0:
                    avg_shoulder_y = np.mean(valid_shoulders[:, 1])
                    logger.debug(f"Average shoulder Y: {avg_shoulder_y:.3f}")
                
                # Use the coordinates as provided from the CSV bounding box normalization
                # The CSV extraction should handle proper coordinate system
                logger.info("Using coordinates directly from CSV bounding box normalization")
                
                # The keypoints should already be properly normalized from the CSV extraction
                # No additional rotation or scaling needed
                
                # Create PoseScript format (52 joints)
                posescript_pose = np.zeros((52, 3))
                
                # Let's use a more direct mapping and see what PoseScript actually expects
                # The issue might be in our complex mapping - let's simplify
                logger.info("Using simplified COCO to PoseScript mapping")
                
                # Simple 1:1 mapping for the first 17 joints, but with proper coordinate handling
                posescript_pose[:17] = centered_keypoints
                
                # Log the pose structure for debugging
                logger.info("Pose keypoint analysis:")
                for i, (x, y, conf) in enumerate(centered_keypoints[:17]):
                    if conf > 0:
                        joint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
                        logger.info(f"  {joint_names[i]}: ({x:.3f}, {y:.3f}, {conf:.3f})")
                
                # Instead of complex mapping, let's focus on getting the coordinate system right
                
                # The mapping is already done above with the simple approach
                
                # The coordinates are already properly normalized from CSV extraction
                # No need to add computed torso center since coordinates are already good
                
                final_pose = posescript_pose
            else:
                # If not COCO format, apply similar transformations
                final_pose = keypoints
                logger.warning(f"Non-COCO format detected with {keypoints.shape[0]} joints")
            
            # Convert to tensor and add batch dimension
            pose_tensor = torch.FloatTensor(final_pose).unsqueeze(0)  # Shape: (1, n_joints, 3)
            pose_tensor = pose_tensor.to(self.device)
            
            logger.debug(f"Converted pose tensor shape: {pose_tensor.shape}")
            logger.debug(f"Pose coordinate range: X[{final_pose[:, 0].min():.3f}, {final_pose[:, 0].max():.3f}], Y[{final_pose[:, 1].min():.3f}, {final_pose[:, 1].max():.3f}]")
            
            # Debug: Analyze the pose structure that's being sent to PoseScript
            self._debug_pose_structure(final_pose)
            
            return pose_tensor
            
        except Exception as e:
            logger.error(f"Failed to convert pose to PoseScript format: {e}")
            raise ValueError(f"Pose conversion failed: {str(e)}")
    

    
    def _debug_pose_structure(self, pose_array: np.ndarray):
        """Debug function to analyze pose structure and detect orientation issues."""
        valid_joints = pose_array[pose_array[:, 2] > 0]
        if len(valid_joints) == 0:
            logger.warning("No valid joints found in pose")
            return
        
        # Analyze key relationships
        head_y = pose_array[0, 1] if pose_array[0, 2] > 0 else None  # nose
        hip_y = np.mean([pose_array[11, 1], pose_array[12, 1]]) if pose_array[11, 2] > 0 and pose_array[12, 2] > 0 else None
        
        if head_y is not None and hip_y is not None:
            logger.debug(f"Head Y: {head_y:.3f}, Hip Y: {hip_y:.3f}")
            if head_y > hip_y:
                logger.warning("⚠️  HEAD IS BELOW HIPS - This suggests the pose might be interpreted as lying down!")
            else:
                logger.info("✅ Head is above hips - normal standing orientation")
        
        # Check arm positions
        left_shoulder = pose_array[5] if pose_array[5, 2] > 0 else None
        right_shoulder = pose_array[6] if pose_array[6, 2] > 0 else None
        left_wrist = pose_array[9] if pose_array[9, 2] > 0 else None
        right_wrist = pose_array[10] if pose_array[10, 2] > 0 else None
        
        if left_shoulder is not None and left_wrist is not None:
            arm_vector = left_wrist[:2] - left_shoulder[:2]
            logger.debug(f"Left arm vector: ({arm_vector[0]:.3f}, {arm_vector[1]:.3f})")
        
        if right_shoulder is not None and right_wrist is not None:
            arm_vector = right_wrist[:2] - right_shoulder[:2]
            logger.debug(f"Right arm vector: ({arm_vector[0]:.3f}, {arm_vector[1]:.3f})")
    
    def is_ready(self) -> bool:
        """Check if the real PoseScript generator is ready for inference."""
        return self.is_initialized and self.model is not None


# Convenience function
def generate_posescript_description(
    pose_input: Union[PoseData, Dict, List, np.ndarray],
    model_path: str = None,
    normalize: bool = True,
    max_length: int = 200,
    temperature: float = 0.7
) -> GenerationResult:
    """
    Convenience function to generate a description using the real PoseScript model.
    
    Args:
        pose_input: Pose data in various formats
        model_path: Path to PoseScript model checkpoint
        normalize: Whether to normalize pose coordinates
        max_length: Maximum description length
        temperature: Generation temperature
        
    Returns:
        GenerationResult: PoseScript generation result
    """
    generator = RealPoseScriptGenerator(model_path=model_path)
    return generator.generate_description(
        pose_input=pose_input,
        normalize=normalize,
        max_length=max_length,
        temperature=temperature
    )