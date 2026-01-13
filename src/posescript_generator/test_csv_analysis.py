#!/usr/bin/env python3
"""
Complete script to process badminton CSV data and generate pose descriptions for Player B using PoseScript.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Tuple, Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import PoseScript components
try:
    from models import PoseData, GenerationResult, setup_logging
    from posescript_generator import PoseScriptGenerator
    print("✅ Successfully imported PoseScript components")
except ImportError as e:
    print(f"❌ Error importing PoseScript components: {e}")
    print("Please ensure you're running from the posescript_generator directory")
    sys.exit(1)

def extract_player_b_keypoints(df_row, confidence_threshold=0.3):
    """
    Extract Player B keypoints from a CSV row.
    
    Args:
        df_row: DataFrame row containing pose data
        confidence_threshold: Minimum confidence score for keypoints (0.0 to 1.0)
        
    Returns:
        List of (x, y, confidence) tuples representing COCO keypoint coordinates
    """
    # COCO keypoint names in order
    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Get Player B bounding box
    x_min = df_row['bbb_xmin']
    y_min = df_row['bbb_ymin'] 
    x_max = df_row['bbb_xmax']
    y_max = df_row['bbb_ymax']
    
    # Calculate bounding box dimensions
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    bbox_center_x = (x_min + x_max) / 2.0
    bbox_center_y = (y_min + y_max) / 2.0
    
    keypoints = []
    
    for kp_name in keypoint_names:
        # Get Player B keypoint data
        x_col = f'b_{kp_name}_x'
        y_col = f'b_{kp_name}_y'
        conf_col = f'b_{kp_name}_confidence'
        
        x_img = df_row[x_col] 
        y_img = df_row[y_col] 
        conf = df_row[conf_col] 
        
        # Normalize coordinates relative to bounding box
        if bbox_width > 0 and bbox_height > 0:
            x_norm = (float(x_img) - bbox_center_x) / (bbox_width / 2.0)
            y_norm = (float(y_img) - bbox_center_y) / (bbox_height / 2.0)
        else:
            x_norm, y_norm = 0.0, 0.0
        
        # For low-confidence keypoints, set coordinates to 0
        if conf < confidence_threshold:
            x_norm, y_norm = 0.0, 0.0
        
        # Return normalized coordinates with confidence
        keypoints.append((x_norm, y_norm, float(conf)))
    
    return keypoints

def process_frame_with_posescript(generator, df, frame_idx, confidence_threshold=0.3):
    """
    Process a single frame with PoseScript to generate description.
    
    Args:
        generator: Initialized PoseScriptGenerator
        df: DataFrame containing pose data
        frame_idx: Index of frame to process
        confidence_threshold: Minimum confidence for keypoints
        
    Returns:
        Dict with results or None if failed
    """
    try:
        print(f"\n--- Processing Frame {frame_idx + 1} ---")
        
        # Extract keypoints for Player B
        keypoints = extract_player_b_keypoints(df.iloc[frame_idx], confidence_threshold)
        
        # Check if we have valid keypoints
        valid_keypoints = [kp for kp in keypoints if kp[2] >= confidence_threshold]
        print(f"🔍 Valid keypoints: {len(valid_keypoints)}/{len(keypoints)}")
        
        if len(valid_keypoints) < 5:  # Need at least 5 keypoints for meaningful description
            print("⚠️  Too few valid keypoints, skipping frame")
            return None
        
        # Create PoseData object
        pose_data = PoseData(
            keypoints=keypoints,
            format="coco",
            metadata={
                "frame_index": frame_idx,
                "source": "badminton_smash",
                "player": "B",
                "confidence_threshold": confidence_threshold
            }
        )
        
        # Generate description using PoseScript
        print("🔄 Generating description with PoseScript...")
        start_time = time.time()
        
        result = generator.generate_description(
            pose_input=pose_data,
            normalize=True,
            max_length=200,
            temperature=0.2
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            print(f"✅ Success! (confidence: {result.confidence:.3f}, time: {processing_time:.3f}s)")
            print(f"📝 Description: {result.description}")
            
            return {
                'frame': frame_idx + 1,
                'player': 'B',
                'description': result.description,
                'confidence': result.confidence,
                'processing_time': processing_time,
                'valid_keypoints': len(valid_keypoints),
                'total_keypoints': len(keypoints)
            }
        else:
            print(f"❌ PoseScript generation failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Error processing frame {frame_idx}: {e}")
        return None

def main():
    """Main function to process CSV and generate descriptions."""
    print("🏸 Badminton Pose Description Generator for Player B")
    print("=" * 60)
    
    # Set up logging
    setup_logging(level="INFO")
    
    # CSV file path
    csv_path = "/Users/chanakyd/work/badminton/VB_DATA/poses/14_Smash/2022-08-30_18-00-09_dataset_set1_087_006675_006699_B_14.csv"
    
    print(f"📁 Reading CSV: {csv_path}")
    
    # Load CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded CSV with {len(df)} frames")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Initialize PoseScriptGenerator
    print("\n🤖 Initializing PoseScriptGenerator...")
    generator = PoseScriptGenerator()
    
    if not generator.initialize():
        print("❌ Failed to initialize PoseScriptGenerator")
        return
    
    print("✅ PoseScriptGenerator initialized successfully!")
    
    # Process sample frames
    sample_frames = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
    confidence_threshold = 0.3
    
    print(f"\n🎯 Processing {len(sample_frames)} sample frames for Player B...")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    
    results = []
    
    for frame_idx in sample_frames:
        result = process_frame_with_posescript(generator, df, frame_idx, confidence_threshold)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n📈 Summary for Player B")
    print("=" * 35)
    print(f"Total frames processed: {len(results)}")
    
    if results:
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_time = np.mean([r['processing_time'] for r in results])
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average processing time: {avg_time:.3f}s")
        
        print(f"\n🎯 All Generated Descriptions for Player B:")
        print("-" * 50)
        for result in results:
            print(f"Frame {result['frame']}: {result['description']}")
            print(f"  → Confidence: {result['confidence']:.3f}")
            print(f"  → Valid keypoints: {result['valid_keypoints']}/{result['total_keypoints']}")
            print()
        
        # Save results to JSON
        output_file = Path("./output/player_b_descriptions.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
    
    print("🎉 Processing complete!")

if __name__ == "__main__":
    main()