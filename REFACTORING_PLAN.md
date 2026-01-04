# Badminton Codebase Refactoring Plan

## Overview
This plan addresses critical code readability issues identified in the badminton analysis codebase. The refactoring will be executed in 4 phases over 4 weeks, prioritizing the most impactful changes first.

## Phase 1: Critical Infrastructure (Week 1)
**Goal:** Fix fundamental issues that block maintainability

### 1.1 Configuration Management
- [x] Create `badminton/config.py` with all paths and settings
- [x] Remove all hard-coded paths from codebase
- [x] Add environment variable support
- [x] Update all modules to use centralized config

### 1.2 Prompt Consolidation
- [x] Create `badminton/data/pose_descriptions.json` 
- [x] Merge content from `prompt.py`, `bd_prompt.py`, `shot_classification_prompt.py`
- [x] Create `badminton/llm_analysis/prompt_manager.py` for template handling
- [x] Update all LLM modules to use new prompt system
- [x] Remove duplicate prompt files

### 1.3 Documentation Foundation
- [x] Add comprehensive docstrings to all public functions in:
  - [x] `utilities/utilities.py`
  - [x] `pose_extractor/detectors.py` 
  - [ ] `utilities/visualization_utilities.py`
- [x] Add type hints to all function signatures
- [x] Create module-level docstrings explaining purpose

## Phase 2: Structural Refactoring (Week 2) - COMPLETED
**Goal:** Break down large classes and eliminate duplication

### 2.1 Text Generator Unification
- [ ] ~~Analyze all 4 text generators and identify the canonical one~~
- [ ] ~~Create unified `badminton/text_generation/generator.py`~~
- [ ] ~~Archive unused generators to `badminton/archive/`~~
- [ ] ~~Update all references to use unified generator~~
- **SKIPPED**: Ignoring posescript_generator directory as requested

### 2.2 VideoPoseDataset Refactoring
- [x] Create `badminton/data/pose_data_loader.py`
- [x] Create `badminton/visualization/pose_visualizer.py`
- [x] Create `badminton/features/pose_feature_extractor.py`
- [x] Create `badminton/analysis/shot_descriptor.py`
- [x] Migrate functionality from `VideoPoseDataset` to new classes
- [x] Update all imports and usage

### 2.3 Common Utilities Extraction
- [x] Create `badminton/utils/pose_utils.py` for shared pose operations
- [x] Create `badminton/utils/video_utils.py` for video processing
- [x] Create `badminton/utils/keypoint_utils.py` for COCO keypoint handling
- [x] Move duplicated code to appropriate utility modules
- [x] Update imports across codebase

## Phase 3: Code Quality Improvements (Week 3)
**Goal:** Standardize conventions and improve robustness

### 3.1 Naming Convention Standardization
- [x] Create `STYLE_GUIDE.md` with naming conventions
- [x] Rename all functions to consistent snake_case
- [x] Rename all classes to consistent PascalCase
- [x] Rename all constants to SCREAMING_SNAKE_CASE
- [x] Update all references
- [x] Create validation script to maintain standards

### 3.2 Error Handling Enhancement
- [ ] Add input validation to all public functions
- [ ] Create custom exception classes in `badminton/exceptions.py`
- [ ] Add meaningful error messages with context
- [ ] Add file existence checks for all file operations
- [ ] Add model loading validation

### 3.3 Magic Number Elimination
- [ ] Create `badminton/constants.py` for all magic numbers
- [ ] Replace hardcoded values with named constants
- [ ] Add comments explaining the meaning of constants
- [ ] Group related constants logically

## Phase 4: Polish and Testing (Week 4)
**Goal:** Final cleanup and validation

### 4.1 Logging Implementation
- [ ] Create `badminton/utils/logging_config.py`
- [ ] Replace all print statements with proper logging
- [ ] Add configurable log levels
- [ ] Add file logging option

### 4.2 Code Completion
- [ ] Complete or remove all TODO/incomplete functions
- [ ] Remove commented-out code blocks
- [ ] Clean up unused imports
- [ ] Remove unused variables

### 4.3 Integration Testing
- [ ] Create basic integration tests for each module
- [ ] Test configuration loading
- [ ] Test prompt system
- [ ] Test refactored classes
- [ ] Validate all imports work correctly

### 4.4 Documentation Updates
- [ ] Update main `README.md` with new structure
- [ ] Create module-specific README files
- [ ] Document new configuration system
- [ ] Create usage examples

## Success Metrics
- [ ] 30-40% reduction in code duplication
- [ ] 100% of public functions have docstrings
- [ ] 0 hard-coded paths remaining
- [ ] All modules can be imported without errors
- [ ] Consistent naming conventions throughout
- [ ] Comprehensive error handling

## File Structure After Refactoring
```
badminton/
├── config.py                          # Centralized configuration
├── constants.py                       # All magic numbers and constants
├── exceptions.py                      # Custom exception classes
├── data/
│   ├── pose_descriptions.json         # Consolidated prompt data
│   └── pose_data_loader.py           # Data loading logic
├── utils/
│   ├── pose_utils.py                 # Shared pose operations
│   ├── video_utils.py                # Video processing utilities
│   ├── keypoint_utils.py             # COCO keypoint handling
│   └── logging_config.py             # Logging configuration
├── visualization/
│   └── pose_visualizer.py            # Pose visualization
├── features/
│   └── pose_feature_extractor.py     # Feature extraction
├── analysis/
│   └── shot_descriptor.py            # Shot description logic
├── llm_analysis/
│   └── prompt_manager.py             # Prompt template system
├── posescript_generator/             # IGNORED - not part of refactoring
└── archive/                          # Deprecated/unused code
```

## Execution Notes
- Each phase should be completed before moving to the next
- Run tests after each major change
- Keep git commits small and focused
- Document any breaking changes
- Maintain backward compatibility where possible during transition