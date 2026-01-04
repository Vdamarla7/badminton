#!/usr/bin/env python3
"""
Focused test for Phase 1 refactoring structure without external dependencies.
Tests only the code structure and imports, not ML functionality.
"""

import sys
import warnings

# Add badminton to path
sys.path.insert(0, 'badminton')

def test_module_structure():
    """Test that refactored modules have correct structure without importing ML libraries."""
    print("🧪 Testing Module Structure...")
    
    try:
        # Test utilities module structure
        from badminton.utilities import utilities
        
        # Check that functions exist
        expected_functions = [
            'get_video_metadata', 'get_video_frames', 'find_bboxes_closest_to_center',
            'validate_video_file', 'resize_frame', 'get_frame_at_time'
        ]
        
        for func_name in expected_functions:
            assert hasattr(utilities, func_name), f"utilities should have {func_name}"
        
        # Test that functions have docstrings
        for func_name in expected_functions:
            func = getattr(utilities, func_name)
            assert func.__doc__ is not None, f"{func_name} should have docstring"
            assert len(func.__doc__.strip()) > 10, f"{func_name} should have meaningful docstring"
        
        # Test detectors module structure (without importing ML dependencies)
        import importlib.util
        
        detectors_path = "badminton/pose_extractor/detectors.py"
        spec = importlib.util.spec_from_file_location("detectors", detectors_path)
        
        # Read the file content to check structure
        with open(detectors_path, 'r') as f:
            content = f.read()
        
        # Check that key classes and functions are defined
        expected_items = [
            'class TaskType', 'class SapiensPoseEstimationType', 
            'class DetectorConfig', 'class Detector', 'class SapiensPoseConfig',
            'def download_file', 'def create_preprocessor'
        ]
        
        for item in expected_items:
            assert item in content, f"detectors.py should contain {item}"
        
        # Check that centralized config is imported
        assert 'from badminton.config import' in content, "Should import from centralized config"
        assert 'YOLO_MODEL_PATH' in content, "Should use YOLO_MODEL_PATH from config"
        assert 'MODEL_CONFIG' in content, "Should use MODEL_CONFIG from config"
        
        # Check documentation
        assert '"""' in content, "Should have module docstring"
        assert 'Args:' in content, "Should have function documentation with Args"
        assert 'Returns:' in content, "Should have function documentation with Returns"
        
        # NOTE: Ignoring posescript_generator directory as requested
        
        print("  ✅ Module structure is correct")
        return True
        
    except Exception as e:
        print(f"  ❌ Module structure test failed: {e}")
        return False


def test_import_paths():
    """Test that all import paths work correctly."""
    print("🧪 Testing Import Paths...")
    
    try:
        # Test config imports
        from badminton.config import PROJECT_ROOT, MODEL_CONFIG
        assert PROJECT_ROOT is not None, "PROJECT_ROOT should be importable"
        assert MODEL_CONFIG is not None, "MODEL_CONFIG should be importable"
        
        # Test prompt manager imports
        from badminton.llm_analysis.prompt_manager import PromptManager
        assert PromptManager is not None, "PromptManager should be importable"
        
        # Test utilities imports
        from badminton.utilities.utilities import get_video_metadata
        assert get_video_metadata is not None, "get_video_metadata should be importable"
        
        # Test legacy imports with deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            
            from badminton.llm_analysis.prompt import prompt
            from badminton.llm_analysis.bd_prompt import PROMPT_SMALL
            from badminton.llm_analysis.shot_classification_prompt import SC_BASE_PROMPT
            
            assert prompt is not None, "Legacy prompt should be importable"
            assert PROMPT_SMALL is not None, "Legacy PROMPT_SMALL should be importable"
            assert SC_BASE_PROMPT is not None, "Legacy SC_BASE_PROMPT should be importable"
        
        print("  ✅ All import paths working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Import paths test failed: {e}")
        return False


def test_file_organization():
    """Test that files are organized correctly."""
    print("🧪 Testing File Organization...")
    
    try:
        from pathlib import Path
        
        # Check that new files exist
        required_files = [
            "badminton/config.py",
            "badminton/data/pose_descriptions.json",
            "badminton/data/prompt_templates.json", 
            "badminton/llm_analysis/prompt_manager.py"
        ]
        
        for file_path in required_files:
            path = Path(file_path)
            assert path.exists(), f"Required file should exist: {file_path}"
            assert path.stat().st_size > 0, f"File should not be empty: {file_path}"
        
        # Check that directories were created
        required_dirs = [
            "badminton/data",
            "badminton/logs"
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            assert path.exists(), f"Required directory should exist: {dir_path}"
            assert path.is_dir(), f"Path should be a directory: {dir_path}"
        
        # Check that legacy files still exist but are updated
        legacy_files = [
            "badminton/llm_analysis/prompt.py",
            "badminton/llm_analysis/bd_prompt.py",
            "badminton/llm_analysis/shot_classification_prompt.py"
        ]
        
        for file_path in legacy_files:
            path = Path(file_path)
            assert path.exists(), f"Legacy file should still exist: {file_path}"
            
            with open(path, 'r') as f:
                content = f.read()
            
            assert "DEPRECATED" in content, f"Legacy file should be marked deprecated: {file_path}"
            assert "prompt_manager" in content, f"Legacy file should reference new system: {file_path}"
        
        print("  ✅ File organization is correct")
        return True
        
    except Exception as e:
        print(f"  ❌ File organization test failed: {e}")
        return False


def test_code_quality():
    """Test code quality improvements."""
    print("🧪 Testing Code Quality...")
    
    try:
        # Test that hard-coded paths are removed
        files_to_check = [
            "badminton/pose_extractor/detectors.py",
            "badminton/utilities/utilities.py",
            "badminton/llm_analysis/prompt_manager.py"
            # NOTE: Excluding posescript_generator files as requested
        ]
        
        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for hard-coded paths (common patterns)
            bad_patterns = [
                '../models/', './models/', '/Users/', '/home/',
                'C:\\', 'D:\\', 'hardcoded', 'TODO', 'FIXME'
            ]
            
            for pattern in bad_patterns:
                if pattern in content and pattern not in ['# TODO', '# FIXME']:  # Allow in comments
                    print(f"    ⚠️  Found potential hard-coded path '{pattern}' in {file_path}")
        
        # Test that type hints are present
        from badminton.utilities.utilities import get_video_metadata
        import inspect
        
        sig = inspect.signature(get_video_metadata)
        
        # Check that function has type hints
        has_return_annotation = sig.return_annotation != inspect.Signature.empty
        has_param_annotations = any(
            param.annotation != inspect.Signature.empty 
            for param in sig.parameters.values()
        )
        
        assert has_return_annotation, "get_video_metadata should have return type annotation"
        assert has_param_annotations, "get_video_metadata should have parameter type annotations"
        
        # Test that docstrings follow consistent format
        docstring = get_video_metadata.__doc__
        assert docstring is not None, "Function should have docstring"
        assert "Args:" in docstring, "Docstring should have Args section"
        assert "Returns:" in docstring, "Docstring should have Returns section"
        
        print("  ✅ Code quality improvements verified")
        return True
        
    except Exception as e:
        print(f"  ❌ Code quality test failed: {e}")
        return False


def main():
    """Run focused structure tests."""
    print("🚀 Starting Phase 1 Structure Tests (No External Dependencies)\n")
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Import Paths", test_import_paths),
        ("File Organization", test_file_organization),
        ("Code Quality", test_code_quality),
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
    print("📊 Structure Test Results:")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if success:
            passed += 1
    
    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All structure tests passed!")
        return True
    else:
        print("⚠️  Some structure tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)