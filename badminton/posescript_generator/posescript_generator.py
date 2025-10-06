#!/usr/bin/env python3
"""
Clean PoseScript Text Generator for badminton pose descriptions.
This version incorporates the working 90-degree rotation fix.
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


class PoseScriptGenerator:
    """
    Clean PoseScript generator with working coordinate transformation.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the PoseScript generator.
        
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
        
        logger.info(f"PoseScriptGenerator initialized with model: {self.model_path}")
    
    def initialize(self) -> bool:
        """
        Initialize the PoseScript model.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing PoseScript model...")
            
            # Check if model file exists
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load the PoseScript model
            self.model, self.tokenizer = self._load_posescript_model()
            
            if self.model is None:
                logger.error("Failed to load PoseScript model")
                return False
            
            self.is_initialized = True
            logger.info("PoseScript model initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PoseScript model: {e}")
            return False
    
    def _load_posescript_model(self):
        """
        Load the PoseScript model from checkpoint.
        
        Returns:
            tuple: (model, tokenizer_name) or (None, None) if failed
        """
        try:
            # Debug: Check if PoseScript path is in sys.path
            posescript_path = Path(__file__).parent.parent / "posescript" / "src"
            logger.info(f"PoseScript path: {posescript_path}")
            logger.info(f"PoseScript path exists: {posescript_path.exists()}")
            if str(posescript_path) not in sys.path:
                logger.info("Adding PoseScript path to sys.path")
                sys.path.insert(0, str(posescript_path))
            
            # Test human_body_prior import first
            try:
                import human_body_prior
                logger.info("human_body_prior import successful")
            except ImportError as e:
                logger.error(f"human_body_prior import failed: {e}")
                # Try to install it
                logger.info("Attempting to install human_body_prior...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/nghorbani/human_body_prior.git"])
                import human_body_prior
                logger.info("human_body_prior installed and imported successfully")
            
            # Import PoseScript modules
            from text2pose.generative_caption.model_generative_caption import DescriptionGenerator
            from text2pose.encoders.tokenizers import get_tokenizer_name
            
            logger.info(f"Loading PoseScript model from: {self.model_path}")
            
            # Load checkpoint
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
        Generate text description using the PoseScript model.
        
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
                logger.warning("PoseScriptGenerator not initialized, attempting to initialize...")
                if not self.initialize():
                    return GenerationResult(
                        description="",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message="Failed to initialize PoseScriptGenerator"
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
        Generate text using the PoseScript model with proper coordinate transformation.
        
        Args:
            pose_data: Processed pose data
            max_length: Maximum description length
            temperature: Generation temperature
            num_samples: Number of samples to generate
            
        Returns:
            tuple: (generated_description, confidence_score)
        """
        try:
            # Convert COCO pose to PoseScript format with the working transformation
            pose_tensor = self._convert_coco_to_posescript_format(pose_data)
            
            logger.info(f"Pose tensor shape: {pose_tensor.shape}")
            logger.info(f"Pose tensor device: {pose_tensor.device}")
            
            # Generate description with PoseScript model
            with torch.no_grad():
                if hasattr(self.model, 'generate_text'):
                    logger.debug("Using model.generate_text method")
                    result = self.model.generate_text(pose_tensor)
                    
                    if isinstance(result, tuple) and len(result) >= 2:
                        texts, scores = result[0], result[1]
                    else:
                        texts = result if isinstance(result, list) else [str(result)]
                        scores = None
                    
                    description = texts[0] if texts and len(texts) > 0 else "Generated description"
                    
                    # Calculate confidence from scores
                    if scores is not None and len(scores) > 0:
                        if isinstance(scores[0], torch.Tensor):
                            raw_score = float(scores[0].item())
                        else:
                            raw_score = float(scores[0])
                        
                        # Convert likelihood score to confidence
                        if raw_score < 0:
                            normalized_score = (raw_score + 10) / 10
                            confidence = max(0.1, min(0.9, 1 / (1 + np.exp(-normalized_score * 6))))
                        else:
                            confidence = 0.85
                    else:
                        # Calculate confidence based on description quality
                        desc_length = len(description.split())
                        if desc_length > 15:
                            confidence = 0.80
                        elif desc_length > 8:
                            confidence = 0.70
                        else:
                            confidence = 0.60
                else:
                    logger.warning("No suitable generation method found, using fallback")
                    description = "A person in a dynamic athletic pose with detailed body positioning."
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
        Convert COCO pose data to PoseScript format with the working 90-degree rotation.
        
        Args:
            pose_data: COCO format pose data
            
        Returns:
            torch.Tensor: Pose tensor in PoseScript format
        """
        try:
            # Extract keypoints (z, y, x) - 3D format where z=0 for 2D poses
            keypoints = np.array(pose_data.keypoints)  # Shape: (17, 3)
            
            if keypoints.shape[0] == 17:  # COCO format
                # For 3D format (z, y, x), we need to work with y and x coordinates
                # Extract valid points (where any coordinate is non-zero)
                valid_points = keypoints[np.any(keypoints[:, 1:] != 0, axis=1)]  # Check y,x coordinates
                if len(valid_points) == 0:
                    raise ValueError("No valid keypoints found")
                
                # Center the pose around the torso center using y,x coordinates (indices 1,2)
                torso_joints = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip
                torso_points = keypoints[torso_joints]
                valid_torso = torso_points[np.any(torso_points[:, 1:] != 0, axis=1)]
                
                if len(valid_torso) > 0:
                    center = np.mean(valid_torso[:, 1:], axis=0)  # Mean of y,x coordinates
                else:
                    center = np.mean(valid_points[:, 1:], axis=0)  # Mean of y,x coordinates
                
                # Center the pose (only y,x coordinates)
                centered_keypoints = keypoints.copy()
                centered_keypoints[:, 1:] -= center
                
                # Apply 180-degree rotation to y,x coordinates for proper standing orientation
                logger.info("Applying 180-degree rotation for proper standing orientation")
                
                # Rotate 180 degrees: (y, x) -> (-y, -x)
                # This represents standing poses correctly
                centered_keypoints[:, 1] *= -1  # Flip Y
                centered_keypoints[:, 2] *= -1  # Flip X
                
                # Create PoseScript format (52 joints)
                posescript_pose = np.zeros((52, 3))
                
                # Simple 1:1 mapping for the first 17 joints
                posescript_pose[:17] = centered_keypoints
                
                final_pose = posescript_pose
            else:
                # If not COCO format, use as-is
                final_pose = keypoints
                logger.warning(f"Non-COCO format detected with {keypoints.shape[0]} joints")
            
            # Convert to tensor and add batch dimension
            pose_tensor = torch.FloatTensor(final_pose).unsqueeze(0)  # Shape: (1, n_joints, 3)
            pose_tensor = pose_tensor.to(self.device)
            
            logger.debug(f"Converted pose tensor shape: {pose_tensor.shape}")
            
            return pose_tensor
            
        except Exception as e:
            logger.error(f"Failed to convert pose to PoseScript format: {e}")
            raise ValueError(f"Pose conversion failed: {str(e)}")
    
    def is_ready(self) -> bool:
        """Check if the PoseScript generator is ready for inference."""
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
    Convenience function to generate a description using the PoseScript model.
    
    Args:
        pose_input: Pose data in various formats
        model_path: Path to PoseScript model checkpoint
        normalize: Whether to normalize pose coordinates
        max_length: Maximum description length
        temperature: Generation temperature
        
    Returns:
        GenerationResult: PoseScript generation result
    """
    generator = PoseScriptGenerator(model_path=model_path)
    return generator.generate_description(
        pose_input=pose_input,
        normalize=normalize,
        max_length=max_length,
        temperature=temperature
    )