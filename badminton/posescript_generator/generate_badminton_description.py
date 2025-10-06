#!/usr/bin/env python3
"""
Script to read badminton CSV data and generate descriptions using RealPoseScriptGenerator.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import PoseData, setup_logging
from posescript_generator import PoseScriptGenerator


def extract_player_keypoints(df_row, player='a', confidence_threshold=0.3):
    """Extract player keypoints from a CSV row using bounding box normalization.
    
    Args:
        df_row: DataFrame row containing pose data
        player: Player identifier ('a' or 'b')
        confidence_threshold: Minimum confidence score for keypoints (0.0 to 1.0)
        
    Returns:
        List of (x, y, confidence) tuples representing COCO keypoint coordinates
    """
    if player not in ['a', 'b']:
        raise ValueError("Player must be 'a' or 'b'")
    
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Get bounding box from CSV - this provides proper normalization context
    bbox_prefix = f'{player}bb'  # 'abb' for player a, 'bbb' for player b
    x_min = df_row[f'{bbox_prefix}_xmin']
    y_min = df_row[f'{bbox_prefix}_ymin'] 
    x_max = df_row[f'{bbox_prefix}_xmax']
    y_max = df_row[f'{bbox_prefix}_ymax']
    
    # Calculate bounding box dimensions
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    bbox_center_x = (x_min + x_max) / 2.0
    bbox_center_y = (y_min + y_max) / 2.0
    
    keypoints = []
    
    for kp_name in keypoint_names:
        x_col = f'{player}_{kp_name}_x'
        y_col = f'{player}_{kp_name}_y'
        conf_col = f'{player}_{kp_name}_confidence'
        
        x_img = df_row[x_col] 
        y_img = df_row[y_col] 
        conf = df_row[conf_col] 
        
        # Normalize coordinates relative to bounding box
        # X: normalize to [-1, 1] range within bounding box
        x_norm = (float(x_img) - bbox_center_x) / (bbox_width / 2.0) if bbox_width > 0 else 0.0
        
        # Y: normalize to [-1, 1] range within bounding box 
        # Keep original orientation (no flipping here - the rotation happens in PoseScript generator)
        # y_img = y_max - y_img
        y_norm = (float(y_img) - bbox_center_y) / (bbox_height / 2.0) if bbox_height > 0 else 0.0
        
        # For low-confidence keypoints, set coordinates to 0
        if conf < confidence_threshold:
            x_norm, y_norm = 0.0, 0.0
        
        # Return 3D format for PoseScript: (z, y, x) where z=0 for 2D poses
        keypoints.append((0.0, y_norm, x_norm))
    
    return keypoints


def main(player='a', confidence_threshold=0.3):
    """Main function to process CSV and generate descriptions.
    
    Args:
        player: Player to analyze ('a' or 'b')
        confidence_threshold: Minimum confidence score for keypoints (0.0 to 1.0)
    """
    # Set up logging
    setup_logging(level="INFO")
    
    # CSV file path
    csv_path = "/Users/chanakyd/work/badminton/VB_DATA/poses/14_Smash/2022-08-30_18-00-09_dataset_set1_087_006675_006699_B_14.csv"
    player = 'b'
    
    print("🏸 Badminton Pose Description Generator")
    print("=" * 50)
    print(f"📁 Reading CSV: {csv_path}")
    print(f"� Analyozing Player: {player.upper()}")
    
    print(f"🎯 Confidence Threshold: {confidence_threshold}")
    
    # Load CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} frames from badminton smash sequence")
        print(f"📋 Columns: {len(df.columns)} total columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Initialize the PoseScriptGenerator
    print("\n🤖 Initializing PoseScriptGenerator...") 
    generator = PoseScriptGenerator()
    
    if not generator.initialize():
        print("❌ Failed to initialize PoseScriptGenerator")
        return
    
    print("✅ PoseScriptGenerator initialized successfully!")
    
    # Process a few sample frames
    sample_frames = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
    
    print(f"\n🎯 Processing {len(sample_frames)} sample frames for Player {player.upper()}...")
    
    results = []
    
    for i, frame_idx in enumerate(sample_frames):
        print(f"\n--- Frame {frame_idx + 1}/{len(df)} ---")
        
        try:
            # Extract keypoints for the specified player
            keypoints = extract_player_keypoints(df.iloc[frame_idx], player=player, confidence_threshold=confidence_threshold)
            
            # Debug: Check if we have valid keypoints
            valid_keypoints = [kp for kp in keypoints if kp[2] >= confidence_threshold]  # confidence >= threshold
            print(f"🔍 Debug: Found {len(valid_keypoints)}/{len(keypoints)} valid keypoints (threshold: {confidence_threshold})")
            if len(valid_keypoints) == 0:
                print("⚠️  No valid keypoints found - checking data...")
                # Show first few keypoints for debugging
                for i, (x, y, conf) in enumerate(keypoints[:5]):
                    print(f"   Keypoint {i}: ({x:.3f}, {y:.3f}, conf={conf:.3f})")
            
            # Create PoseData object
            pose_data = PoseData(
                keypoints=keypoints,
                format="coco",
                metadata={
                    "frame_index": frame_idx,
                    "source": "badminton_smash",
                    "player": player.upper()
                }
            )
            
            # Generate description
            print("🔄 Generating description...")
            result = generator.generate_description(
                pose_input=pose_data,
                normalize=True,
                max_length=200,
                temperature=0.2
            )
            
            if result.success:
                print(f"✅ Success! (confidence: {result.confidence:.3f}, time: {result.processing_time:.3f}s)")
                print(f"📝 Description: {result.description}")
                
                results.append({
                    'frame': frame_idx + 1,
                    'player': player.upper(),
                    'description': result.description,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                })
            else:
                print(f"❌ Failed: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Error processing frame {frame_idx}: {e}")
    
    # Summary
    print(f"\n📈 Summary for Player {player.upper()}")
    print("=" * 35)
    print(f"Total frames processed: {len(results)}")
    
    if results:
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_time = np.mean([r['processing_time'] for r in results])
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average processing time: {avg_time:.3f}s")
        
        print(f"\n🎯 All Generated Descriptions for Player {player.upper()}:")
        print("-" * 50)
        for result in results:
            print(f"Frame {result['frame']}: {result['description']}")
            print(f"  → Confidence: {result['confidence']:.3f}")
            print()
    
    print("🎉 Processing complete!")


if __name__ == "__main__":
    import sys
    
    # Default values
    player = 'a'  # Default to player A
    confidence_threshold = 0.3  # Default confidence threshold
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        player_arg = sys.argv[1].lower()
        if player_arg in ['a', 'b']:
            player = player_arg
        else:
            print("❌ Invalid player argument. Use 'a' or 'b'")
            print("Usage: python generate_badminton_description.py [a|b] [confidence_threshold]")
            print("  player: 'a' or 'b' (default: 'a')")
            print("  confidence_threshold: 0.0 to 1.0 (default: 0.3)")
            sys.exit(1)
    
    # Parse confidence threshold if provided
    if len(sys.argv) > 2:
        try:
            confidence_threshold = float(sys.argv[2])
            if not (0.0 <= confidence_threshold <= 1.0):
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        except ValueError as e:
            print(f"❌ Invalid confidence threshold: {e}")
            print("Usage: python generate_badminton_description.py [a|b] [confidence_threshold]")
            print("  player: 'a' or 'b' (default: 'a')")
            print("  confidence_threshold: 0.0 to 1.0 (default: 0.3)")
            sys.exit(1)
    
    main(player, confidence_threshold)