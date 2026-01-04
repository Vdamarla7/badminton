#!/usr/bin/env python3
"""
Test script to validate naming conventions across the badminton codebase.
This ensures all functions, classes, constants, and variables follow the style guide.
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Add badminton to path for imports
sys.path.insert(0, 'badminton')

class NamingConventionChecker:
    """Checks Python files for naming convention compliance."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
        # Naming patterns
        self.function_pattern = re.compile(r'^[a-z_][a-z0-9_]*$')
        self.class_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
        self.constant_pattern = re.compile(r'^[A-Z_][A-Z0-9_]*$')
        self.variable_pattern = re.compile(r'^[a-z_][a-z0-9_]*$')
        
        # Exceptions for common patterns
        self.function_exceptions = {
            '__init__', '__str__', '__repr__', '__len__', '__getitem__',
            '__setitem__', '__delitem__', '__iter__', '__next__', '__enter__',
            '__exit__', '__call__', '__eq__', '__ne__', '__lt__', '__le__',
            '__gt__', '__ge__', '__hash__', '__bool__'
        }
        
        self.constant_exceptions = {
            'logger',  # logging.getLogger(__name__)
            'warnings',  # warnings module
        }
    
    def check_file(self, file_path: Path) -> Dict[str, List[str]]:
        """Check a single Python file for naming convention violations."""
        file_errors = []
        file_warnings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Check each node in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._check_function_name(node.name, file_path, node.lineno, file_errors)
                
                elif isinstance(node, ast.ClassDef):
                    self._check_class_name(node.name, file_path, node.lineno, file_errors)
                
                elif isinstance(node, ast.Assign):
                    # Check for constants (uppercase assignments at module level)
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if self._is_module_level_constant(target.id, content):
                                self._check_constant_name(target.id, file_path, node.lineno, file_errors)
                            else:
                                self._check_variable_name(target.id, file_path, node.lineno, file_warnings)
        
        except SyntaxError as e:
            file_errors.append(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            file_warnings.append(f"Could not parse {file_path}: {e}")
        
        return {
            'errors': file_errors,
            'warnings': file_warnings
        }
    
    def _check_function_name(self, name: str, file_path: Path, line_no: int, errors: List[str]):
        """Check if function name follows snake_case convention."""
        if name in self.function_exceptions:
            return
        
        if not self.function_pattern.match(name):
            errors.append(
                f"{file_path}:{line_no} - Function '{name}' should use snake_case"
            )
    
    def _check_class_name(self, name: str, file_path: Path, line_no: int, errors: List[str]):
        """Check if class name follows PascalCase convention."""
        if not self.class_pattern.match(name):
            errors.append(
                f"{file_path}:{line_no} - Class '{name}' should use PascalCase"
            )
    
    def _check_constant_name(self, name: str, file_path: Path, line_no: int, errors: List[str]):
        """Check if constant name follows SCREAMING_SNAKE_CASE convention."""
        if name in self.constant_exceptions:
            return
        
        if not self.constant_pattern.match(name):
            errors.append(
                f"{file_path}:{line_no} - Constant '{name}' should use SCREAMING_SNAKE_CASE"
            )
    
    def _check_variable_name(self, name: str, file_path: Path, line_no: int, warnings: List[str]):
        """Check if variable name follows snake_case convention."""
        # Skip private variables and special cases
        if name.startswith('_') or name in self.constant_exceptions:
            return
        
        if not self.variable_pattern.match(name):
            warnings.append(
                f"{file_path}:{line_no} - Variable '{name}' should use snake_case"
            )
    
    def _is_module_level_constant(self, name: str, content: str) -> bool:
        """Determine if a variable is likely a module-level constant."""
        # Simple heuristic: if it's all uppercase, it's probably a constant
        return name.isupper() and '_' in name or len(name) > 1 and name.isupper()
    
    def check_directory(self, directory: Path) -> Dict[str, Any]:
        """Check all Python files in a directory recursively."""
        all_errors = []
        all_warnings = []
        files_checked = 0
        
        for py_file in directory.rglob('*.py'):
            # Skip __pycache__ and other generated files
            if '__pycache__' in str(py_file) or '.pyc' in str(py_file):
                continue
            
            # Skip test files for now (they have different conventions)
            if py_file.name.startswith('test_'):
                continue
            
            files_checked += 1
            result = self.check_file(py_file)
            
            if result['errors']:
                all_errors.extend(result['errors'])
            
            if result['warnings']:
                all_warnings.extend(result['warnings'])
        
        return {
            'files_checked': files_checked,
            'errors': all_errors,
            'warnings': all_warnings
        }


def test_naming_conventions():
    """Test naming conventions across the badminton codebase."""
    print("🧪 Testing Naming Conventions...")
    
    checker = NamingConventionChecker()
    badminton_dir = Path('badminton')
    
    if not badminton_dir.exists():
        print("❌ Badminton directory not found")
        return False
    
    results = checker.check_directory(badminton_dir)
    
    print(f"📁 Files checked: {results['files_checked']}")
    
    # Report errors
    if results['errors']:
        print(f"\n❌ Found {len(results['errors'])} naming convention errors:")
        for error in results['errors']:
            print(f"  {error}")
    else:
        print("✅ No naming convention errors found!")
    
    # Report warnings
    if results['warnings']:
        print(f"\n⚠️  Found {len(results['warnings'])} naming convention warnings:")
        for warning in results['warnings'][:10]:  # Limit to first 10
            print(f"  {warning}")
        if len(results['warnings']) > 10:
            print(f"  ... and {len(results['warnings']) - 10} more warnings")
    else:
        print("✅ No naming convention warnings found!")
    
    # Summary
    total_issues = len(results['errors']) + len(results['warnings'])
    if total_issues == 0:
        print("\n🎉 All naming conventions are perfect!")
        return True
    else:
        print(f"\n📊 Summary: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        return len(results['errors']) == 0  # Pass if no errors, even with warnings


def test_specific_patterns():
    """Test specific naming patterns we care about."""
    print("\n🧪 Testing Specific Naming Patterns...")
    
    patterns_to_check = [
        # (pattern, example_good, example_bad, description)
        (r'^[a-z_][a-z0-9_]*$', 'get_pose_data', 'getPoseData', 'Function names (snake_case)'),
        (r'^[A-Z][a-zA-Z0-9]*$', 'VideoPoseDataset', 'videoPoseDataset', 'Class names (PascalCase)'),
        (r'^[A-Z_][A-Z0-9_]*$', 'YOLO_MODEL_PATH', 'yolo_model_path', 'Constants (SCREAMING_SNAKE_CASE)'),
        (r'^[a-z_][a-z0-9_]*$', 'pose_keypoints', 'poseKeypoints', 'Variables (snake_case)'),
    ]
    
    all_good = True
    
    for pattern, good_example, bad_example, description in patterns_to_check:
        regex = re.compile(pattern)
        
        good_match = regex.match(good_example)
        bad_match = regex.match(bad_example)
        
        if good_match and not bad_match:
            print(f"✅ {description}: Pattern works correctly")
        else:
            print(f"❌ {description}: Pattern validation failed")
            all_good = False
    
    return all_good


def main():
    """Run all naming convention tests."""
    print("🚀 Starting Naming Convention Tests\n")
    
    # Test the patterns themselves
    patterns_ok = test_specific_patterns()
    
    # Test the actual codebase
    codebase_ok = test_naming_conventions()
    
    # Overall result
    print("\n" + "="*60)
    print("NAMING CONVENTION TEST SUMMARY")
    print("="*60)
    
    if patterns_ok and codebase_ok:
        print("🎉 All naming convention tests passed!")
        print("✅ The codebase follows excellent naming conventions")
        return True
    else:
        if not patterns_ok:
            print("❌ Pattern validation failed")
        if not codebase_ok:
            print("❌ Codebase has naming convention issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)