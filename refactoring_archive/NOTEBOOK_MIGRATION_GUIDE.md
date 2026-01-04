# Notebook Migration Guide: Using Refactored Badminton Analysis System

## Overview

This guide explains how to migrate from the original `openAI.ipynb` and `openAI_v1.ipynb` notebooks to use the refactored badminton analysis system. The refactored system provides the same functionality with improved architecture, better maintainability, and additional features.

## Quick Start

### For Existing Users (Zero Changes Required)

Your existing notebooks will continue to work without any changes! The refactored system maintains full backward compatibility.

```python
# This still works exactly the same
from badminton.utilities.visualization_utilities import VideoPoseDataset
vpd = VideoPoseDataset(poses_path=pose_file, video_path=video_file)
result = vpd.get_shot_description_for_player(player='green')
```

### For New Users (Recommended Approach)

Use the new modular imports for better performance and features:

```python
# Recommended: Use the new refactored VideoPoseDataset
from badminton.data.video_pose_dataset import VideoPoseDataset
vpd = VideoPoseDataset(poses_path=pose_file, video_path=video_file)
result = vpd.get_shot_description_for_player(player='green')
```

## Migration Options

### Option 1: No Migration (Easiest)

Keep using your existing notebooks as-is. They will work with deprecation warnings that you can ignore.

**Pros:**
- Zero effort required
- No code changes needed
- Immediate compatibility

**Cons:**
- Missing new features
- Deprecation warnings
- Not using optimized architecture

### Option 2: Simple Import Update (Recommended)

Update only the import statement for better performance:

```python
# OLD
from badminton.utilities.visualization_utilities import VideoPoseDataset

# NEW
from badminton.data.video_pose_dataset import VideoPoseDataset
```

**Pros:**
- Minimal changes (1 line)
- Access to new features
- Better performance
- No deprecation warnings

**Cons:**
- Requires small code change

### Option 3: Full Refactored Approach (Advanced)

Use individual modular components for maximum flexibility:

```python
from badminton.data.pose_data_loader import PoseDataLoader
from badminton.features.pose_feature_extractor import PoseFeatureExtractor
from badminton.analysis.shot_descriptor import ShotDescriptor
from badminton.visualization.pose_visualizer import PoseVisualizer

# Use components individually for custom workflows
loader = PoseDataLoader(pose_file)
extractor = PoseFeatureExtractor()
analyzer = ShotDescriptor()
```

**Pros:**
- Maximum flexibility
- Best performance
- Access to all new features
- Modular architecture benefits

**Cons:**
- Requires more code changes
- Need to understand new architecture

## New Features Available

### Enhanced Analysis

```python
# New: Analyze shot patterns
shot_analysis = vpd.analyze_shot_pattern(player='green')
print(f"Shot phases: {shot_analysis['shot_phases']}")
print(f"Most common poselets: {shot_analysis['poselet_summary']['most_common_poselets']}")

# New: Get dataset summary
summary = vpd.get_dataset_summary()
print(f"Video info: {summary['frame_count']} frames, {summary['duration']:.2f}s")
```

### Improved Utilities

```python
from badminton.utils.video_utils import list_video_files, replace_mp4_with_csv

# Better file discovery
video_files = list_video_files("VB_DATA/poses/")

# Utility functions
csv_file = replace_mp4_with_csv("video.mp4")  # Returns "video.csv"
```

### Individual Component Access

```python
# Access individual components for custom workflows
data_loader = vpd.data_loader
visualizer = vpd.visualizer
feature_extractor = vpd.feature_extractor
shot_descriptor = vpd.shot_descriptor

# Use components directly
green_data = data_loader.get_player_data('green')
features = feature_extractor.extract_poselets_for_player(green_data)
```

## Updated Notebook Examples

### Updated openAI.ipynb

See `badminton/openAI_refactored.ipynb` for a complete example that:
- Uses the new import (1 line change)
- Demonstrates new features
- Shows backward compatibility
- Includes migration notes

### Updated openAI_v1.ipynb

See `badminton/openAI_v1_refactored.ipynb` for a batch processing example that:
- Uses improved utilities for file discovery
- Demonstrates enhanced error handling
- Shows advanced analysis features
- Includes performance improvements

## Compatibility Matrix

| Feature | Original | Refactored | Notes |
|---------|----------|------------|-------|
| `VideoPoseDataset()` | ✅ | ✅ | Same interface |
| `get_shot_description_for_player()` | ✅ | ✅ | Same interface |
| `get_poselets_for_player()` | ✅ | ✅ | Same interface |
| `annotate_video_with_poses()` | ✅ | ✅ | Same interface |
| `vpd.playera` / `vpd.playerb` | ✅ | ✅ | Same interface |
| `analyze_shot_pattern()` | ❌ | ✅ | New feature |
| `get_dataset_summary()` | ❌ | ✅ | New feature |
| Individual components | ❌ | ✅ | New feature |
| Enhanced utilities | ❌ | ✅ | New feature |

## Performance Improvements

### Memory Usage
- **Lazy loading**: Video frames loaded only when needed
- **Optimized data structures**: Reduced memory footprint
- **Better garbage collection**: Improved cleanup

### Processing Speed
- **Modular architecture**: Faster component initialization
- **Cached computations**: Avoid redundant calculations
- **Optimized algorithms**: Improved feature extraction

### Batch Processing
- **Better error handling**: Graceful failure recovery
- **Progress tracking**: Enhanced monitoring
- **Resource management**: Improved file handling

## Testing Your Migration

Use the provided test script to verify compatibility:

```bash
python test_refactored_notebooks.py
```

This will test:
- Import compatibility
- Basic functionality
- Backward compatibility
- New features
- Error handling

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # If you get import errors, try:
   import sys
   sys.path.insert(0, '/path/to/badminton')
   ```

2. **Deprecation Warnings**
   ```python
   # To suppress warnings (not recommended):
   import warnings
   warnings.filterwarnings("ignore", category=DeprecationWarning)
   ```

3. **Missing sklearn**
   ```python
   # The system gracefully handles missing sklearn
   # Install if needed: pip install scikit-learn
   ```

### Getting Help

1. **Check the test script**: `python test_refactored_notebooks.py`
2. **Review example notebooks**: `openAI_refactored.ipynb` and `openAI_v1_refactored.ipynb`
3. **Check module documentation**: All modules have comprehensive docstrings
4. **Verify file paths**: Ensure your data files exist and are accessible

## Migration Timeline

### Immediate (Now)
- ✅ All existing notebooks work without changes
- ✅ New refactored notebooks available
- ✅ Full backward compatibility maintained

### Short Term (Next Few Weeks)
- Update import statements for better performance
- Explore new analysis features
- Migrate to refactored notebooks

### Long Term (Future)
- Original imports may be deprecated (with advance notice)
- New features will be added to refactored system only
- Performance optimizations will focus on refactored architecture

## Benefits Summary

### For Developers
- **Better maintainability**: Modular, well-documented code
- **Enhanced testability**: Individual components can be tested
- **Improved extensibility**: Easy to add new features
- **Code reusability**: Components can be used independently

### For Users
- **Same interface**: No learning curve for existing functionality
- **New features**: Enhanced analysis and utilities
- **Better performance**: Optimized algorithms and data structures
- **Improved reliability**: Better error handling and validation

### For Research
- **Reproducibility**: Consistent, well-documented interfaces
- **Flexibility**: Mix and match components for custom workflows
- **Extensibility**: Easy to add new analysis methods
- **Collaboration**: Clear module boundaries for team development

## Conclusion

The refactored badminton analysis system provides significant improvements while maintaining full backward compatibility. You can:

1. **Continue using existing notebooks** without any changes
2. **Gradually migrate** by updating import statements
3. **Leverage new features** for enhanced analysis
4. **Use modular components** for custom workflows

The choice is yours, and all approaches are fully supported!