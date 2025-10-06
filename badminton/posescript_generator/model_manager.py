"""
Model manager for downloading and loading PoseScript models.
"""

import os
import sys
import requests
import zipfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages downloading, loading, and initialization of PoseScript models."""
    
    # Default model configuration - using the pose description generation model
    DEFAULT_MODEL_CONFIG = {
        "name": "capgen_CAtransfPSA2H2_dataPSA2ftPSH2",
        "url": "https://download.europe.naverlabs.com/ComputerVision/PoseFix/capgen_CAtransfPSA2H2_dataPSA2ftPSH2.zip",
        "description": "Pose description generation model"
    }
    
    def __init__(self, model_dir: str = "./models", model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory to store downloaded models
            model_config: Custom model configuration (uses default if None)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = model_config or self.DEFAULT_MODEL_CONFIG
        self.model_name = self.config["name"]
        self.model_url = self.config["url"]
        
        self.model_path = self.model_dir / self.model_name
        self.checkpoint_path = self.model_path / "seed1" / "checkpoint_best.pth"
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"ModelManager initialized for {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def download_model(self, force_download: bool = False) -> bool:
        """
        Download the pre-trained model if it doesn't exist.
        
        Args:
            force_download: Force re-download even if model exists
            
        Returns:
            True if download successful or model already exists, False otherwise
        """
        if self.checkpoint_path.exists() and not force_download:
            logger.info(f"Model already exists at {self.checkpoint_path}")
            return True
        
        try:
            logger.info(f"Downloading model from {self.model_url}")
            
            # Download with progress bar
            response = requests.get(self.model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            zip_path = self.model_dir / f"{self.model_name}.zip"
            
            with open(zip_path, 'wb') as f, tqdm(
                desc=f"Downloading {self.model_name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract the zip file
            logger.info(f"Extracting model to {self.model_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.model_dir)
            
            # Clean up zip file
            zip_path.unlink()
            
            if self.checkpoint_path.exists():
                logger.info("Model downloaded and extracted successfully")
                return True
            else:
                logger.error(f"Expected checkpoint not found at {self.checkpoint_path}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Network error during download: {e}")
            logger.error("Please check your internet connection and try again")
            return False
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            logger.error("The downloaded file may be corrupted. Try downloading again.")
            # Clean up corrupted zip file
            if zip_path.exists():
                zip_path.unlink()
            return False
        except PermissionError as e:
            logger.error(f"Permission error: {e}")
            logger.error(f"Please check write permissions for directory: {self.model_dir}")
            return False
        except OSError as e:
            logger.error(f"OS error during download: {e}")
            logger.error("Please check available disk space and directory permissions")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load the pre-trained model into memory.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.checkpoint_path.exists():
            logger.error(f"Model checkpoint not found at {self.checkpoint_path}")
            logger.info("Attempting to download model...")
            if not self.download_model():
                return False
        
        try:
            logger.info(f"Loading model from {self.checkpoint_path}")
            
            # Load the checkpoint with weights_only=False for compatibility with older models
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            
            # Store checkpoint and extract model info for future PoseScript integration
            self.model = {
                'checkpoint': checkpoint,
                'model_args': checkpoint.get('args', None),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'model_state': checkpoint.get('model', None)
            }
            
            # Log model information
            if self.model['model_args']:
                logger.info(f"Model loaded from epoch {self.model['epoch']}")
                logger.info(f"Model type: {self.config['description']}")
            else:
                logger.warning("Model args not found in checkpoint")
            
            logger.info("Model loaded successfully")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            return False
        except torch.serialization.pickle.UnpicklingError as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            return False
    
    def is_model_ready(self) -> bool:
        """
        Check if the model is loaded and ready for inference.
        
        Returns:
            True if model is ready, False otherwise
        """
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "name": self.model_name,
            "description": self.config.get("description", ""),
            "model_path": str(self.model_path),
            "checkpoint_path": str(self.checkpoint_path),
            "is_downloaded": self.checkpoint_path.exists(),
            "is_loaded": self.is_model_ready(),
            "device": str(self.device)
        }
    
    def setup_posescript_environment(self) -> bool:
        """
        Set up the PoseScript environment and dependencies.
        This is a placeholder for future PoseScript integration.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Add posescript to Python path if it exists
            posescript_path = Path("./posescript/src")
            if posescript_path.exists():
                sys.path.insert(0, str(posescript_path))
                logger.info(f"Added PoseScript to Python path: {posescript_path}")
                return True
            else:
                logger.warning("PoseScript source directory not found")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up PoseScript environment: {e}")
            return False
    
    def initialize_posescript_model(self):
        """
        Initialize the actual PoseScript model for inference.
        This method will be implemented when PoseScript integration is needed.
        
        Returns:
            The initialized PoseScript model or None if failed
        """
        if not self.is_model_ready():
            logger.error("Model checkpoint not loaded. Call load_model() first.")
            return None
        
        try:
            # This is a placeholder for actual PoseScript model initialization
            # The real implementation would look like:
            # 
            # from text2pose.generative_caption.model_generative_caption import DescriptionGenerator
            # 
            # args = self.model['model_args']
            # model = DescriptionGenerator(
            #     text_decoder_name=args.text_decoder_name,
            #     transformer_mode=args.transformer_mode,
            #     decoder_nlayers=args.decoder_nlayers,
            #     decoder_nhead=args.decoder_nhead,
            #     encoder_latentD=args.latentD,
            #     decoder_latentD=args.decoder_latentD,
            #     num_body_joints=getattr(args, 'num_body_joints', 52)
            # ).to(self.device)
            # model.load_state_dict(self.model['model_state'])
            # model.eval()
            # return model
            
            logger.info("PoseScript model initialization placeholder - ready for integration")
            return self.model  # Return checkpoint for now
            
        except Exception as e:
            logger.error(f"Error initializing PoseScript model: {e}")
            return None