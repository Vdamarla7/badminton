# Badminton Codebase Style Guide

## Overview
This style guide establishes consistent naming conventions and coding standards for the badminton analysis codebase. Following these conventions improves code readability, maintainability, and collaboration.

## Naming Conventions

### Functions and Methods
- **Use snake_case** for all function and method names
- **Use descriptive, verb-based names** that clearly indicate what the function does
- **Avoid abbreviations** unless they are widely understood

```python
# ✅ Good
def extract_pose_features(keypoints):
def validate_pose_file(file_path):
def get_shot_description_for_player(player_name):

# ❌ Bad  
def extractPoseFeatures(keypoints)  # camelCase
def validatePoseFile(file_path)     # camelCase
def get_desc(player)                # abbreviated
```

### Classes
- **Use PascalCase** for all class names
- **Use descriptive nouns** that clearly indicate what the class represents
- **Avoid prefixes** like "C" or "Cls"

```python
# ✅ Good
class VideoPoseDataset:
class PoseFeatureExtractor:
class ShotDescriptor:

# ❌ Bad
class videoPoseDataset:     # camelCase starting with lowercase
class pose_feature_extractor: # snake_case
class CPoseExtractor:       # Hungarian notation prefix
```

### Constants
- **Use SCREAMING_SNAKE_CASE** for all constants
- **Group related constants** together
- **Use descriptive names** that indicate the constant's purpose

```python
# ✅ Good
YOLO_MODEL_PATH = Path("models/yolo.pt")
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MAX_POSE_KEYPOINTS = 17

# ❌ Bad
yolo_model_path = Path("models/yolo.pt")  # snake_case
YoloModelPath = Path("models/yolo.pt")    # PascalCase
YOLO_PATH = Path("models/yolo.pt")        # abbreviated
```

### Variables
- **Use snake_case** for all variable names
- **Use descriptive names** that indicate the variable's content
- **Avoid single-letter variables** except for short loops

```python
# ✅ Good
pose_keypoints = extract_keypoints(frame)
player_data = load_player_data(file_path)
frame_count = len(video_frames)

# ❌ Bad
poseKeypoints = extract_keypoints(frame)  # camelCase
pd = load_player_data(file_path)          # abbreviated
x = len(video_frames)                     # single letter
```

### Files and Directories
- **Use snake_case** for Python module names
- **Use kebab-case** for directory names when they contain multiple words
- **Use descriptive names** that indicate the module's purpose

```python
# ✅ Good
pose_feature_extractor.py
shot_descriptor.py
video_utils.py

# Directory structure
badminton/
├── llm-analysis/
├── pose-extractor/
└── utils/

# ❌ Bad
PoseFeatureExtractor.py    # PascalCase
shotDescriptor.py          # camelCase
videoUtils.py              # camelCase
```

## Current Naming Issues to Fix

### Functions That Need Renaming
Based on codebase analysis, these functions should be renamed:

1. **Configuration Functions** (already correct):
   - `get_env_path()` ✅
   - `get_env_config()` ✅
   - `ensure_directories()` ✅
   - `validate_paths()` ✅

2. **Utility Functions** (already correct):
   - `validate_keypoints_dict()` ✅
   - `get_missing_keypoints()` ✅
   - `fill_missing_keypoints()` ✅

3. **Test Functions** (already correct):
   - `test_refactored_notebook_imports()` ✅
   - `test_video_utils()` ✅
   - `test_modular_components()` ✅

### Classes That Need Renaming
Most classes already follow PascalCase correctly:

1. **Data Classes** (already correct):
   - `PromptConfig` ✅
   - `PromptManager` ✅
   - `VideoPoseDataset` ✅
   - `PoseDataLoader` ✅

2. **Detector Classes** (already correct):
   - `TaskType` ✅
   - `SapiensPoseEstimationType` ✅
   - `DetectorConfig` ✅
   - `Detector` ✅

### Constants That Need Standardization
These variables should be converted to SCREAMING_SNAKE_CASE constants:

1. **Prompt Constants** (need fixing):
   ```python
   # Current (❌)
   PROMPT_SMALL = get_pose_classification_prompt(...)
   PROMPT_FULL = get_pose_classification_prompt(...)
   SC_BASE_PROMPT = get_sc_base_prompt()
   SC_INPUT_PROMPT = "..."
   
   # Should be (✅) - these are actually correct as they are constants
   ```

2. **Configuration Constants** (already correct):
   - `POSE_DESCRIPTIONS_PATH` ✅
   - `PROMPT_TEMPLATES_PATH` ✅
   - `YOLO_MODEL_PATH` ✅

## Implementation Priority

### Phase 1: Critical Fixes (High Impact)
1. ✅ **Constants are already properly named**
2. ✅ **Classes are already properly named**
3. ✅ **Functions are already properly named**

### Phase 2: Documentation and Validation
1. **Add naming validation to tests**
2. **Update docstrings to reflect naming conventions**
3. **Create linting rules to enforce conventions**

## Conclusion

**Good News**: The codebase already follows excellent naming conventions! The refactoring work in Phases 1 and 2 has resulted in:

- ✅ All functions use snake_case
- ✅ All classes use PascalCase  
- ✅ All constants use SCREAMING_SNAKE_CASE
- ✅ All variables use snake_case
- ✅ All modules use snake_case

The naming conventions are already consistent and professional. The main remaining work is to:
1. Add validation to prevent future naming inconsistencies
2. Ensure all new code follows these conventions
3. Update any remaining legacy code that might not follow these patterns

## Validation Rules

To maintain these standards, we should add linting rules that check:
- Function names match `^[a-z_][a-z0-9_]*$`
- Class names match `^[A-Z][a-zA-Z0-9]*$`
- Constants match `^[A-Z_][A-Z0-9_]*$`
- Variable names match `^[a-z_][a-z0-9_]*$`