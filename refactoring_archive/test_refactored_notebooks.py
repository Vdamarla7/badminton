#!/usr/bin/env python3
"""
Test script to verify that the refactored notebooks will work correctly.
This simulates the key operations from both notebooks.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add the badminton package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'badminton'))

def test_refactored_notebook_imports():
    """Test that all imports from the refactored notebooks work."""
    print("Testing imports...")
    
    try:
        # Test new refactored imports
        from badminton.data.video_pose_dataset import VideoPoseDataset
        from badminton.data.pose_data_loader import PoseDataLoader
        from badminton.visualization.pose_visualizer import PoseVisualizer
        from badminton.features.pose_feature_extractor import PoseFeatureExtractor
        from badminton.analysis.shot_descriptor import ShotDescriptor
        
        # Test utility imports
        from badminton.utils.video_utils import list_video_files, replace_mp4_with_csv
        from badminton.utils.pose_utils import crop_image_with_bbox
        from badminton.utils.keypoint_utils import validate_keypoints_dict
        
        # Test legacy imports still work
        from badminton.utilities.coco_keypoints import create_keypoints_dict
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_video_utils():
    """Test the video utility functions."""
    print("\nTesting video utilities...")
    
    try:
        from badminton.utils.video_utils import replace_mp4_with_csv
        
        # Test file extension replacement
        result = replace_mp4_with_csv("test_video.mp4")
        assert result == "test_video.csv", f"Expected 'test_video.csv', got '{result}'"
        
        # Test non-mp4 file
        result = replace_mp4_with_csv("test_file.avi")
        assert result == "test_file.avi", f"Expected 'test_file.avi', got '{result}'"
        
        print("✅ Video utilities working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Video utilities test failed: {e}")
        return False

def test_modular_components():
    """Test that individual modular components can be instantiated."""
    print("\nTesting modular components...")
    
    try:
        from badminton.data.pose_data_loader import PoseDataLoader
        from badminton.visualization.pose_visualizer import PoseVisualizer
        from badminton.features.pose_feature_extractor import PoseFeatureExtractor
        from badminton.analysis.shot_descriptor import ShotDescriptor
        
        # Test component instantiation
        visualizer = PoseVisualizer()
        feature_extractor = PoseFeatureExtractor()
        shot_descriptor = ShotDescriptor()
        
        print("✅ All modular components instantiated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Modular components test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that the old interface still works with deprecation warnings."""
    print("\nTesting backward compatibility...")
    
    try:
        import warnings
        
        # Capture deprecation warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # This should work but show a deprecation warning
            from badminton.utilities.visualization_utilities import VideoPoseDataset
            
            # Check if deprecation warning was issued
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            
            print("✅ Backward compatibility maintained!")
            if deprecation_warnings:
                print(f"   (Deprecation warning correctly shown: {len(deprecation_warnings)} warnings)")
            
            return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def test_mock_dataset_creation():
    """Test creating a VideoPoseDataset with mocked data."""
    print("\nTesting dataset creation with mocked data...")
    
    try:
        from badminton.data.pose_data_loader import PoseDataLoader
        
        # Test just the data loader component which doesn't need video files
        with patch('builtins.open'):
            with patch('csv.reader') as mock_csv:
                # Create mock CSV data with proper structure
                mock_csv_data = [
                    ['frame'] + ['bbox_col'] * 8 + [f'kp_{i}' for i in range(102)],  # Header
                    ['0'] + ['10.0'] * 8 + ['1.0'] * 102,  # Data row
                    ['1'] + ['11.0'] * 8 + ['1.1'] * 102   # Data row
                ]
                mock_csv.return_value = iter(mock_csv_data)
                
                # Create data loader
                loader = PoseDataLoader("test.csv")
                
                # Test basic functionality
                assert len(loader) == 2, f"Expected 2 frames, got {len(loader)}"
                assert hasattr(loader, 'playera'), "Missing playera attribute"
                assert hasattr(loader, 'playerb'), "Missing playerb attribute"
                
                # Test data access
                green_data = loader.get_player_data('green')
                blue_data = loader.get_player_data('blue')
                assert len(green_data) == 2, "Expected 2 frames for green player"
                assert len(blue_data) == 2, "Expected 2 frames for blue player"
                
                print("✅ Data loading and basic functionality working!")
                return True
        
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction with mock data."""
    print("\nTesting feature extraction...")
    
    try:
        from badminton.features.pose_feature_extractor import PoseFeatureExtractor
        
        # Create mock keypoints data
        mock_keypoints = {
            'left_ankle': (10.0, 20.0, 0.9),
            'left_knee': (15.0, 15.0, 0.8),
            'left_hip': (20.0, 10.0, 0.7),
            'left_shoulder': (25.0, 5.0, 0.9),
            'left_elbow': (30.0, 8.0, 0.8),
            'left_wrist': (35.0, 12.0, 0.7),
            'right_wrist': (5.0, 12.0, 0.7),
            'right_elbow': (10.0, 8.0, 0.8),
            'right_shoulder': (15.0, 5.0, 0.9),
            'right_hip': (20.0, 10.0, 0.7),
            'right_knee': (25.0, 15.0, 0.8),
            'right_ankle': (30.0, 20.0, 0.9)
        }
        
        mock_player_data = [([0, 0, 100, 100], mock_keypoints)]
        
        extractor = PoseFeatureExtractor()
        features = extractor.extract_poselets_for_player(mock_player_data)
        
        assert len(features) == 1, f"Expected 1 frame of features, got {len(features)}"
        assert 'left_arm' in features[0], "Missing left_arm feature"
        assert 'right_arm' in features[0], "Missing right_arm feature"
        
        # Test summary
        summary = extractor.get_poselet_summary(features)
        assert 'total_frames' in summary, "Missing total_frames in summary"
        
        print("✅ Feature extraction working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING REFACTORED NOTEBOOKS COMPATIBILITY")
    print("=" * 60)
    
    tests = [
        test_refactored_notebook_imports,
        test_video_utils,
        test_modular_components,
        test_backward_compatibility,
        test_mock_dataset_creation,
        test_feature_extraction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total*100):.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! The refactored notebooks should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)