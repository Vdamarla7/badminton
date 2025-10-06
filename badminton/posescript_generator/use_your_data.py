#!/usr/bin/env python3
"""
PoseScript Text Generator - Use Your Own Data

This script shows how to use the PoseScript Text Generator with your own pose data.
Modify the pose data below to test with your own keypoints.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("🎯 Using Your Own Pose Data")
    print("=" * 50)
    
    try:
        from text_generator import TextGenerator
        from models import PoseData, setup_logging
        
        # Set up logging
        setup_logging(level="INFO")
        
        # ============================================================
        # MODIFY THIS SECTION WITH YOUR OWN POSE DATA
        # ============================================================
        
        # Example: Replace these keypoints with your own data
        # COCO format requires 17 keypoints: (x, y, confidence)
        # confidence: 0 = not visible, 1 = visible, 2 = occluded
        
        your_keypoints = [
            # Head keypoints
            (320, 100, 1.0),  # 0: nose
            (310, 90, 1.0),   # 1: left_eye
            (330, 90, 1.0),   # 2: right_eye
            (300, 100, 1.0),  # 3: left_ear
            (340, 100, 1.0),  # 4: right_ear
            
            # Upper body keypoints
            (280, 150, 1.0),  # 5: left_shoulder
            (360, 150, 1.0),  # 6: right_shoulder
            (250, 200, 1.0),  # 7: left_elbow
            (390, 200, 1.0),  # 8: right_elbow
            (220, 250, 1.0),  # 9: left_wrist
            (420, 250, 1.0),  # 10: right_wrist
            
            # Lower body keypoints
            (290, 300, 1.0),  # 11: left_hip
            (350, 300, 1.0),  # 12: right_hip
            (280, 400, 1.0),  # 13: left_knee
            (360, 400, 1.0),  # 14: right_knee
            (270, 500, 1.0),  # 15: left_ankle
            (370, 500, 1.0),  # 16: right_ankle
        ]
        
        # Optional: Add metadata about your pose
        your_metadata = {
            "source": "my_pose_detection_system",
            "image_width": 640,
            "image_height": 480,
            "person_id": 1,
            "timestamp": "2024-01-01T12:00:00Z"
        }
        
        # ============================================================
        # END OF MODIFICATION SECTION
        # ============================================================
        
        print("1. Creating pose data from your keypoints...")
        pose_data = PoseData(
            keypoints=your_keypoints,
            format="coco",
            metadata=your_metadata
        )
        
        print(f"   ✅ Created pose with {len(pose_data.keypoints)} keypoints")
        
        # Analyze your pose data
        visible_keypoints = pose_data.get_visible_keypoints()
        bounding_box = pose_data.get_bounding_box()
        
        print(f"   📊 Visible keypoints: {len(visible_keypoints)}/{len(pose_data.keypoints)}")
        if bounding_box:
            print(f"   📏 Bounding box: ({bounding_box[0]:.0f}, {bounding_box[1]:.0f}) to ({bounding_box[2]:.0f}, {bounding_box[3]:.0f})")
            print(f"   📐 Size: {bounding_box[2]-bounding_box[0]:.0f} x {bounding_box[3]-bounding_box[1]:.0f} pixels")
        
        print("\n2. Initializing text generator...")
        generator = TextGenerator(model_dir="./models")
        
        if not generator.initialize():
            print("❌ Failed to initialize generator")
            return False
        
        print("3. Generating description from your pose data...")
        result = generator.generate_description(pose_data, normalize=True)
        
        print("\n" + "=" * 50)
        print("📝 GENERATED DESCRIPTION:")
        print("=" * 50)
        
        if result.success:
            print(f"✅ Description: '{result.description}'")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Processing time: {result.processing_time:.3f} seconds")
        else:
            print(f"❌ Generation failed: {result.error_message}")
            return False
        
        # Optional: Save your results
        print("\n4. Saving results...")
        from models import save_pose_data, save_generation_result
        
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        
        save_pose_data(pose_data, output_dir / "your_pose.json")
        save_generation_result(result, output_dir / "your_result.json")
        
        print(f"   💾 Saved pose data to: {output_dir / 'your_pose.json'}")
        print(f"   💾 Saved result to: {output_dir / 'your_result.json'}")
        
        print("\n" + "=" * 50)
        print("🎉 Successfully processed your pose data!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_keypoint_format():
    """Show the COCO keypoint format for reference."""
    print("\n📋 COCO Keypoint Format Reference:")
    print("=" * 40)
    
    keypoint_names = [
        "0: nose", "1: left_eye", "2: right_eye", "3: left_ear", "4: right_ear",
        "5: left_shoulder", "6: right_shoulder", "7: left_elbow", "8: right_elbow",
        "9: left_wrist", "10: right_wrist", "11: left_hip", "12: right_hip",
        "13: left_knee", "14: right_knee", "15: left_ankle", "16: right_ankle"
    ]
    
    for i, name in enumerate(keypoint_names):
        if i % 2 == 0:
            print(f"{name:<20}", end="")
        else:
            print(f"{name}")
    
    print("\nEach keypoint: (x, y, confidence)")
    print("Confidence: 0=not visible, 1=visible, 2=occluded")


if __name__ == "__main__":
    show_keypoint_format()
    success = main()
    sys.exit(0 if success else 1)