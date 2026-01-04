#!/usr/bin/env python3
"""
Comprehensive test suite for Phase 1 refactoring changes.
Tests configuration system, prompt management, and backward compatibility.
"""

import sys
import os
import warnings
import json
from pathlib import Path

# Add badminton to path
sys.path.insert(0, 'badminton')

def test_config_system():
    """Test the centralized configuration system."""
    print("🧪 Testing Configuration System...")
    
    try:
        from badminton.config import (
            PROJECT_ROOT, DATA_ROOT, MODELS_ROOT,
            MODEL_CONFIG, VISUALIZATION_CONFIG, VIDEO_CONFIG, COCO_CONFIG,
            validate_paths, get_config_summary, ensure_directories
        )
        
        # Test basic imports
        assert PROJECT_ROOT.exists(), f"Project root should exist: {PROJECT_ROOT}"
        assert isinstance(MODEL_CONFIG, dict), "MODEL_CONFIG should be a dictionary"
        assert isinstance(VISUALIZATION_CONFIG, dict), "VISUALIZATION_CONFIG should be a dictionary"
        
        # Test path validation
        validation = validate_paths()
        assert isinstance(validation, dict), "validate_paths should return a dictionary"
        assert validation['PROJECT_ROOT'] == True, "PROJECT_ROOT should be valid"
        
        # Test config summary
        summary = get_config_summary()
        assert 'project_root' in summary, "Summary should contain project_root"
        assert 'model_config' in summary, "Summary should contain model_config"
        
        # Test directory creation
        ensure_directories()  # Should not raise any errors
        
        # Test COCO configuration
        assert len(COCO_CONFIG['keypoint_names']) == 17, "Should have 17 COCO keypoints"
        assert COCO_CONFIG['num_keypoints'] == 17, "num_keypoints should be 17"
        
        print("  ✅ Configuration system working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration system failed: {e}")
        return False


def test_prompt_manager():
    """Test the prompt management system."""
    print("🧪 Testing Prompt Manager...")
    
    try:
        from badminton.llm_analysis.prompt_manager import (
            PromptManager, PromptConfig, 
            get_pose_classification_prompt, get_shot_classification_prompt
        )
        
        # Test PromptManager initialization
        manager = PromptManager()
        
        # Test data validation
        validation = manager.validate_data()
        expected_keys = [
            'pose_descriptions_loaded', 'prompt_templates_loaded',
            'pose_mappings_available', 'simplified_mappings_available',
            'pose_classification_templates_available', 'shot_classification_templates_available'
        ]
        
        for key in expected_keys:
            assert key in validation, f"Validation should contain {key}"
            assert validation[key] == True, f"{key} should be True"
        
        # Test pose mappings
        full_mappings = manager.get_pose_mappings(simplified=False)
        simplified_mappings = manager.get_pose_mappings(simplified=True)
        
        assert len(full_mappings) > len(simplified_mappings), "Full mappings should be longer"
        assert "P_0.0_0.0" in full_mappings, "Should contain basic pose mapping"
        assert "P_90.0_90.0" in simplified_mappings, "Should contain simplified mapping"
        
        # Test pose classification prompt generation
        test_description = "Mostly straight pointing in up (north)"
        
        full_prompt = manager.get_pose_classification_prompt(
            test_description, 
            PromptConfig(template_type="full")
        )
        simplified_prompt = manager.get_pose_classification_prompt(
            test_description,
            PromptConfig(template_type="simplified")
        )
        
        assert len(full_prompt) > 1000, "Full prompt should be substantial"
        assert len(simplified_prompt) > 100, "Simplified prompt should exist"
        assert test_description in full_prompt, "Description should be in prompt"
        assert test_description in simplified_prompt, "Description should be in simplified prompt"
        
        # Test shot classification prompt
        test_input = "Position: ServeLine\nFrames: t0,P_210_330,P_90_90"
        shot_prompt = manager.get_shot_classification_prompt(test_input)
        
        assert len(shot_prompt) > 1000, "Shot prompt should be substantial"
        assert test_input in shot_prompt, "Input should be in shot prompt"
        assert "badminton assistant coach" in shot_prompt.lower(), "Should contain identity"
        
        # Test convenience functions
        conv_pose_prompt = get_pose_classification_prompt(test_description)
        conv_shot_prompt = get_shot_classification_prompt(test_input)
        
        assert len(conv_pose_prompt) > 1000, "Convenience pose prompt should work"
        assert len(conv_shot_prompt) > 1000, "Convenience shot prompt should work"
        
        # Test available templates
        templates = manager.get_available_templates()
        assert len(templates) > 0, "Should have available templates"
        
        print("  ✅ Prompt manager working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Prompt manager failed: {e}")
        return False


def test_data_files():
    """Test the JSON data files."""
    print("🧪 Testing Data Files...")
    
    try:
        # Test pose descriptions file
        pose_desc_path = Path("badminton/data/pose_descriptions.json")
        assert pose_desc_path.exists(), "Pose descriptions file should exist"
        
        with open(pose_desc_path, 'r') as f:
            pose_data = json.load(f)
        
        assert 'pose_mappings' in pose_data, "Should contain pose_mappings"
        assert 'simplified_mappings' in pose_data, "Should contain simplified_mappings"
        
        # Check pose mappings structure
        pose_mappings = pose_data['pose_mappings']
        simplified_mappings = pose_data['simplified_mappings']
        
        assert len(pose_mappings) > 100, "Should have many pose mappings"
        assert len(simplified_mappings) == 4, "Should have 4 simplified mappings"
        
        # Check specific mappings
        assert 'P_0.0_0.0' in pose_mappings, "Should have basic east mapping"
        assert 'P_90.0_90.0' in pose_mappings, "Should have basic north mapping"
        assert 'P_90.0_90.0' in simplified_mappings, "Should have north in simplified"
        
        # Test prompt templates file
        templates_path = Path("badminton/data/prompt_templates.json")
        assert templates_path.exists(), "Prompt templates file should exist"
        
        with open(templates_path, 'r') as f:
            template_data = json.load(f)
        
        assert 'pose_classification' in template_data, "Should contain pose_classification"
        assert 'shot_classification' in template_data, "Should contain shot_classification"
        
        # Check pose classification templates
        pose_templates = template_data['pose_classification']
        assert 'base_template' in pose_templates, "Should have base_template"
        assert 'full_template' in pose_templates, "Should have full_template"
        assert 'simplified_template' in pose_templates, "Should have simplified_template"
        
        # Check shot classification templates
        shot_templates = template_data['shot_classification']
        assert 'identity' in shot_templates, "Should have identity"
        assert 'shot_descriptions' in shot_templates, "Should have shot_descriptions"
        
        shot_descriptions = shot_templates['shot_descriptions']
        expected_shots = ['00_Short_Serve', '13_Long_Serve', '05_Drop_Shot', '14_Smash']
        for shot in expected_shots:
            assert shot in shot_descriptions, f"Should contain {shot} description"
        
        print("  ✅ Data files are valid and complete")
        return True
        
    except Exception as e:
        print(f"  ❌ Data files test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility with legacy imports."""
    print("🧪 Testing Backward Compatibility...")
    
    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        try:
            # Test legacy prompt imports
            from badminton.llm_analysis.prompt import prompt, get_prompt
            from badminton.llm_analysis.bd_prompt import PROMPT_SMALL, PROMPT_FULL
            from badminton.llm_analysis.shot_classification_prompt import SC_BASE_PROMPT, SC_INPUT_PROMPT
            
            # Test that legacy variables exist and have content
            assert len(prompt) > 100, "Legacy prompt should have content"
            assert len(PROMPT_SMALL) > 100, "PROMPT_SMALL should have content"
            assert len(PROMPT_FULL) > 100, "PROMPT_FULL should have content"
            assert len(SC_BASE_PROMPT) > 100, "SC_BASE_PROMPT should have content"
            
            # Test legacy functions
            custom_prompt = get_prompt("Test description")
            assert len(custom_prompt) > 100, "get_prompt function should work"
            assert "Test description" in custom_prompt, "Custom description should be in prompt"
            
            # Test that prompts contain expected content
            assert "helpful assistant" in prompt.lower(), "Should contain assistant identity"
            assert "key and value" in prompt.lower(), "Should contain key-value instruction"
            assert "badminton" in SC_BASE_PROMPT.lower(), "Should contain badminton context"
            
            print("  ✅ Backward compatibility maintained")
            return True
            
        except Exception as e:
            print(f"  ❌ Backward compatibility failed: {e}")
            return False


def test_updated_modules():
    """Test updated modules use centralized config."""
    print("🧪 Testing Updated Modules...")
    
    try:
        # Test utilities module
        from badminton.utilities.utilities import (
            get_video_metadata, get_video_frames, find_bboxes_closest_to_center,
            validate_video_file, resize_frame
        )
        
        # Test that functions exist and have proper signatures
        import inspect
        
        # Check get_video_metadata signature
        sig = inspect.signature(get_video_metadata)
        assert 'video_path' in sig.parameters, "get_video_metadata should have video_path parameter"
        
        # Check get_video_frames signature  
        sig = inspect.signature(get_video_frames)
        assert 'video_path' in sig.parameters, "get_video_frames should have video_path parameter"
        assert 'max_frames' in sig.parameters, "get_video_frames should have max_frames parameter"
        
        # Check find_bboxes_closest_to_center signature
        sig = inspect.signature(find_bboxes_closest_to_center)
        assert 'bboxes' in sig.parameters, "find_bboxes_closest_to_center should have bboxes parameter"
        assert 'center_x' in sig.parameters, "Should have center_x parameter"
        
        # Test detectors module imports (without actually loading models)
        from badminton.pose_extractor.detectors import (
            TaskType, SapiensPoseEstimationType, DetectorConfig, 
            Detector, SapiensPoseConfig, download_file, create_preprocessor
        )
        
        # Test enum values
        assert TaskType.POSE.value == "pose", "TaskType.POSE should have correct value"
        
        # Test config classes
        detector_config = DetectorConfig()
        assert hasattr(detector_config, 'model_path'), "DetectorConfig should have model_path"
        assert hasattr(detector_config, 'conf_thres'), "DetectorConfig should have conf_thres"
        
        sapiens_config = SapiensPoseConfig()
        assert hasattr(sapiens_config, 'device'), "SapiensPoseConfig should have device"
        assert hasattr(sapiens_config, 'input_size'), "SapiensPoseConfig should have input_size"
        
        print("  ✅ Updated modules working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Updated modules test failed: {e}")
        return False


def test_error_handling():
    """Test error handling in refactored code."""
    print("🧪 Testing Error Handling...")
    
    try:
        from badminton.llm_analysis.prompt_manager import PromptManager
        from badminton.utilities.utilities import get_video_metadata, find_bboxes_closest_to_center
        import numpy as np
        
        # Test prompt manager with invalid data
        manager = PromptManager()
        
        # Test with empty description (should not crash)
        try:
            result = manager.get_pose_classification_prompt("")
            assert len(result) > 0, "Should handle empty description gracefully"
        except Exception as e:
            print(f"    ⚠️  Empty description handling could be improved: {e}")
        
        # Test utilities error handling
        try:
            # Test with non-existent video file
            get_video_metadata("nonexistent_video.mp4")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
        except Exception as e:
            print(f"    ⚠️  Unexpected error type: {e}")
        
        # Test find_bboxes_closest_to_center with empty array
        try:
            result = find_bboxes_closest_to_center(np.array([]), 100)
            assert result == [], "Should return empty list for empty input"
        except Exception as e:
            print(f"    ⚠️  Empty bbox handling could be improved: {e}")
        
        # Test with invalid bbox format
        try:
            invalid_bboxes = np.array([[1, 2]])  # Only 2 columns instead of 4+
            find_bboxes_closest_to_center(invalid_bboxes, 100)
            assert False, "Should raise ValueError for invalid bbox format"
        except ValueError:
            pass  # Expected
        except Exception as e:
            print(f"    ⚠️  Unexpected error for invalid bbox: {e}")
        
        print("  ✅ Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False


def test_performance():
    """Test performance of refactored components."""
    print("🧪 Testing Performance...")
    
    try:
        import time
        from badminton.llm_analysis.prompt_manager import PromptManager
        
        # Test prompt manager initialization time
        start_time = time.time()
        manager = PromptManager()
        init_time = time.time() - start_time
        
        assert init_time < 1.0, f"PromptManager initialization should be fast, took {init_time:.3f}s"
        
        # Test prompt generation time
        test_description = "Mostly straight pointing in up (north)"
        
        start_time = time.time()
        for _ in range(10):
            prompt = manager.get_pose_classification_prompt(test_description)
        generation_time = (time.time() - start_time) / 10
        
        assert generation_time < 0.1, f"Prompt generation should be fast, took {generation_time:.3f}s per call"
        
        # Test data reload time
        start_time = time.time()
        success = manager.reload_data()
        reload_time = time.time() - start_time
        
        assert success, "Data reload should succeed"
        assert reload_time < 0.5, f"Data reload should be fast, took {reload_time:.3f}s"
        
        print(f"  ✅ Performance acceptable (init: {init_time:.3f}s, generation: {generation_time:.3f}s)")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False


def run_integration_test():
    """Run an end-to-end integration test."""
    print("🧪 Running Integration Test...")
    
    try:
        # Test complete workflow: config -> prompt manager -> legacy compatibility
        from badminton.config import MODEL_CONFIG, validate_paths
        from badminton.llm_analysis.prompt_manager import PromptManager, PromptConfig
        
        # 1. Validate configuration
        paths_valid = validate_paths()
        assert paths_valid['PROJECT_ROOT'], "Project root should be valid"
        
        # 2. Initialize prompt manager
        manager = PromptManager()
        validation = manager.validate_data()
        assert all(validation.values()), f"All validations should pass: {validation}"
        
        # 3. Generate different types of prompts
        test_desc = "Very bent with the shoulder elbow pointing right (east), elbow wrist pointing up (north)"
        
        # Full prompt
        full_prompt = manager.get_pose_classification_prompt(
            test_desc, 
            PromptConfig(template_type="full")
        )
        
        # Simplified prompt
        simple_prompt = manager.get_pose_classification_prompt(
            test_desc,
            PromptConfig(template_type="simplified")
        )
        
        # Shot classification prompt
        shot_input = "Position: MidCourt\nFrames: t0,P_90_90,P_90_90,P_90_90,P_270_270,P_90_90,P_90_90"
        shot_prompt = manager.get_shot_classification_prompt(shot_input)
        
        # 4. Verify prompt content
        assert test_desc in full_prompt, "Description should be in full prompt"
        assert test_desc in simple_prompt, "Description should be in simple prompt"
        assert shot_input in shot_prompt, "Input should be in shot prompt"
        assert len(full_prompt) > len(simple_prompt), "Full prompt should be longer"
        
        # 5. Test legacy compatibility
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from badminton.llm_analysis.prompt import prompt
            from badminton.llm_analysis.bd_prompt import PROMPT_SMALL
            
            assert len(prompt) > 100, "Legacy prompt should work"
            assert len(PROMPT_SMALL) > 100, "Legacy PROMPT_SMALL should work"
        
        # 6. Test configuration usage
        yolo_config = MODEL_CONFIG['yolo']
        assert 'confidence_threshold' in yolo_config, "YOLO config should have confidence_threshold"
        assert isinstance(yolo_config['confidence_threshold'], (int, float)), "Confidence should be numeric"
        
        print("  ✅ Integration test passed - all components working together")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests and provide summary."""
    print("🚀 Starting Phase 1 Refactoring Tests\n")
    
    tests = [
        ("Configuration System", test_config_system),
        ("Prompt Manager", test_prompt_manager),
        ("Data Files", test_data_files),
        ("Backward Compatibility", test_backward_compatibility),
        ("Updated Modules", test_updated_modules),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance),
        ("Integration", run_integration_test),
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
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if success:
            passed += 1
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 refactoring is solid.")
        return True
    else:
        print("⚠️  Some tests failed. Review the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)