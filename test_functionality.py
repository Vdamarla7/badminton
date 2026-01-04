#!/usr/bin/env python3
"""
Test actual functionality of refactored components.
"""

import sys
import warnings
import numpy as np

# Add badminton to path
sys.path.insert(0, 'badminton')

def test_prompt_generation():
    """Test that prompt generation produces expected output."""
    print("🧪 Testing Prompt Generation Functionality...")
    
    try:
        from badminton.llm_analysis.prompt_manager import PromptManager, PromptConfig
        
        manager = PromptManager()
        
        # Test pose classification prompt
        test_description = "Very bent with the shoulder elbow pointing right (east), elbow wrist pointing up (north)"
        
        # Generate full prompt
        full_prompt = manager.get_pose_classification_prompt(
            test_description,
            PromptConfig(template_type="full")
        )
        
        # Verify prompt structure
        assert "helpful assistant" in full_prompt.lower(), "Should contain assistant identity"
        assert "key and value" in full_prompt.lower(), "Should contain instruction"
        assert test_description in full_prompt, "Should contain the test description"
        assert "P_0.0_90.0" in full_prompt, "Should contain pose mappings"
        
        # Test simplified prompt
        simple_prompt = manager.get_pose_classification_prompt(
            test_description,
            PromptConfig(template_type="simplified")
        )
        
        assert len(simple_prompt) < len(full_prompt), "Simplified should be shorter"
        assert "P_90.0_90.0" in simple_prompt, "Should contain basic mappings"
        assert test_description in simple_prompt, "Should contain description"
        
        # Test shot classification prompt
        shot_input = """Position: ServeLine
Frames: t0,P_210_330,P_90_90,P_90_90,P_210_330,P_90_90,P_90_90
t1,P_210_330,P_90_90,P_90_90,P_210_330,P_90_90,P_90_90"""
        
        shot_prompt = manager.get_shot_classification_prompt(shot_input)
        
        assert "badminton assistant coach" in shot_prompt.lower(), "Should contain coach identity"
        assert "00_Short_Serve" in shot_prompt, "Should contain shot types"
        assert "Position: ServeLine" in shot_prompt, "Should contain input data"
        assert "JSON" in shot_prompt, "Should specify JSON output format"
        
        print(f"  ✅ Generated prompts: Full={len(full_prompt)}, Simple={len(simple_prompt)}, Shot={len(shot_prompt)}")
        return True
        
    except Exception as e:
        print(f"  ❌ Prompt generation failed: {e}")
        return False


def test_config_functionality():
    """Test configuration system functionality."""
    print("🧪 Testing Configuration Functionality...")
    
    try:
        from badminton.config import (
            PROJECT_ROOT, MODEL_CONFIG, VISUALIZATION_CONFIG,
            validate_paths, get_config_summary, ensure_directories
        )
        
        # Test path validation
        validation = validate_paths()
        
        # Should have these keys
        expected_keys = ['PROJECT_ROOT', 'DATA_ROOT', 'MODELS_ROOT']
        for key in expected_keys:
            assert key in validation, f"Validation should include {key}"
            assert validation[key] == True, f"{key} should be valid"
        
        # Test config structure
        assert 'yolo' in MODEL_CONFIG, "Should have YOLO config"
        assert 'sapiens' in MODEL_CONFIG, "Should have Sapiens config"
        assert 'posescript' in MODEL_CONFIG, "Should have PoseScript config"
        
        # Test YOLO config values
        yolo_config = MODEL_CONFIG['yolo']
        assert 'confidence_threshold' in yolo_config, "Should have confidence threshold"
        assert isinstance(yolo_config['confidence_threshold'], (int, float)), "Confidence should be numeric"
        assert 0 <= yolo_config['confidence_threshold'] <= 1, "Confidence should be between 0 and 1"
        
        # Test visualization config
        assert 'colors' in VISUALIZATION_CONFIG, "Should have color config"
        assert 'player_a' in VISUALIZATION_CONFIG['colors'], "Should have player A color"
        
        # Test config summary
        summary = get_config_summary()
        assert 'project_root' in summary, "Summary should have project root"
        assert 'model_config' in summary, "Summary should have model config"
        
        # Test directory creation (should not fail)
        ensure_directories()
        
        print("  ✅ Configuration system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration functionality failed: {e}")
        return False


def test_utilities_functionality():
    """Test utilities functionality."""
    print("🧪 Testing Utilities Functionality...")
    
    try:
        from badminton.utilities.utilities import find_bboxes_closest_to_center, resize_frame
        
        # Test find_bboxes_closest_to_center
        # Create test bounding boxes [x1, y1, x2, y2]
        test_bboxes = np.array([
            [10, 10, 50, 50],   # Left box
            [100, 10, 140, 50], # Center box  
            [200, 10, 240, 50]  # Right box
        ])
        
        center_x = 120  # Closer to center box
        closest = find_bboxes_closest_to_center(test_bboxes, center_x, num_bboxes=2)
        
        assert len(closest) == 2, "Should return 2 closest boxes"
        assert isinstance(closest[0], list), "Should return lists"
        assert len(closest[0]) == 4, "Each box should have 4 coordinates"
        
        # The center box (100-140) should be closest to center_x=120
        center_box = [100, 10, 140, 50]
        assert center_box in closest, "Center box should be in closest results"
        
        # Test with empty input
        empty_result = find_bboxes_closest_to_center(np.array([]), center_x)
        assert empty_result == [], "Should return empty list for empty input"
        
        # Test resize_frame
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test scaling
        scaled_frame = resize_frame(test_frame, scale_factor=0.5)
        assert scaled_frame.shape[:2] == (50, 50), "Should scale correctly"
        
        # Test target size
        resized_frame = resize_frame(test_frame, target_size=(80, 60))
        assert resized_frame.shape[:2] == (60, 80), "Should resize to target size"
        
        print("  ✅ Utilities functionality working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Utilities functionality failed: {e}")
        return False


def test_legacy_compatibility_functionality():
    """Test that legacy code produces same results as new system."""
    print("🧪 Testing Legacy Compatibility Functionality...")
    
    try:
        # Suppress deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            
            # Test legacy prompt import
            from badminton.llm_analysis.prompt import prompt, get_prompt
            from badminton.llm_analysis.bd_prompt import PROMPT_SMALL, PROMPT_FULL
            
            # Test that legacy prompts have expected content
            assert "helpful assistant" in prompt.lower(), "Legacy prompt should have assistant"
            assert "P_90.0_90.0" in prompt, "Legacy prompt should have pose mappings"
            
            # Test legacy function
            custom_desc = "Test description for legacy compatibility"
            custom_prompt = get_prompt(custom_desc)
            
            assert custom_desc in custom_prompt, "Custom description should be in prompt"
            assert len(custom_prompt) > 500, "Custom prompt should be substantial"
            
            # Compare with new system
            from badminton.llm_analysis.prompt_manager import get_pose_classification_prompt
            new_prompt = get_pose_classification_prompt(custom_desc)
            
            # Should contain similar key elements (though exact format may differ)
            assert "helpful assistant" in new_prompt.lower(), "New prompt should have assistant"
            assert custom_desc in new_prompt, "New prompt should have description"
            
            # Test PROMPT_SMALL vs PROMPT_FULL - they should be different
            # Note: Both use same description but different templates
            assert len(PROMPT_SMALL) != len(PROMPT_FULL) or PROMPT_SMALL != PROMPT_FULL, "PROMPT_SMALL and PROMPT_FULL should be different"
            
            # Test new system simplified vs full
            from badminton.llm_analysis.prompt_manager import PromptManager, PromptConfig
            manager = PromptManager()
            
            new_simple = manager.get_pose_classification_prompt(
                "Test", PromptConfig(template_type="simplified")
            )
            new_full = manager.get_pose_classification_prompt(
                "Test", PromptConfig(template_type="full")
            )
            
            # New system should definitely have different sizes
            assert len(new_simple) < len(new_full), "New simple should be shorter than new full"
            
        print("  ✅ Legacy compatibility functionality working")
        return True
        
    except Exception as e:
        print(f"  ❌ Legacy compatibility functionality failed: {e}")
        return False


def test_data_integrity():
    """Test that data files contain expected content."""
    print("🧪 Testing Data Integrity...")
    
    try:
        import json
        from pathlib import Path
        
        # Test pose descriptions
        pose_desc_path = Path("badminton/data/pose_descriptions.json")
        with open(pose_desc_path, 'r') as f:
            pose_data = json.load(f)
        
        pose_mappings = pose_data['pose_mappings']
        simplified_mappings = pose_data['simplified_mappings']
        
        # Test specific pose mappings
        expected_poses = {
            'P_0.0_0.0': 'right (east)',
            'P_90.0_90.0': 'up (north)',
            'P_180.0_180.0': 'left (west)',
            'P_270.0_270.0': 'down (south)'
        }
        
        for pose_key, expected_content in expected_poses.items():
            assert pose_key in pose_mappings, f"Should have {pose_key} in full mappings"
            assert expected_content in pose_mappings[pose_key].lower(), f"{pose_key} should contain '{expected_content}'"
            
            if pose_key in simplified_mappings:
                assert expected_content in simplified_mappings[pose_key].lower(), f"Simplified {pose_key} should contain '{expected_content}'"
        
        # Test that we have comprehensive coverage
        assert len(pose_mappings) > 200, "Should have comprehensive pose mappings"
        assert len(simplified_mappings) == 4, "Should have exactly 4 simplified mappings"
        
        # Test prompt templates
        templates_path = Path("badminton/data/prompt_templates.json")
        with open(templates_path, 'r') as f:
            template_data = json.load(f)
        
        # Test shot descriptions
        shot_descriptions = template_data['shot_classification']['shot_descriptions']
        expected_shots = ['00_Short_Serve', '13_Long_Serve', '05_Drop_Shot', '14_Smash']
        
        for shot in expected_shots:
            assert shot in shot_descriptions, f"Should have {shot} description"
            shot_desc = shot_descriptions[shot]
            
            # Each shot should have position and notes
            assert 'position' in shot_desc, f"{shot} should have position"
            assert 'notes' in shot_desc, f"{shot} should have notes"
        
        print("  ✅ Data integrity verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Data integrity test failed: {e}")
        return False


def main():
    """Run functionality tests."""
    print("🚀 Starting Phase 1 Functionality Tests\n")
    
    tests = [
        ("Prompt Generation", test_prompt_generation),
        ("Configuration", test_config_functionality),
        ("Utilities", test_utilities_functionality),
        ("Legacy Compatibility", test_legacy_compatibility_functionality),
        ("Data Integrity", test_data_integrity),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  💥 {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 Functionality Test Results:")
    print("=" * 45)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if success:
            passed += 1
    
    print("=" * 45)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All functionality tests passed!")
        return True
    else:
        print("⚠️  Some functionality tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)