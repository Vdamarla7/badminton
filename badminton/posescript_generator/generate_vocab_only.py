#!/usr/bin/env python3
"""
Generate vocabulary files for PoseScript.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add PoseScript to path
posescript_path = Path(__file__).parent.parent / "posescript" / "src"
if posescript_path.exists():
    sys.path.insert(0, str(posescript_path))


def generate_vocabulary_files():
    """Generate the vocabulary files using PoseScript's vocab.py script."""
    print("📖 Generating vocabulary files...")
    
    # Change to PoseScript source directory
    posescript_src_dir = Path(__file__).parent.parent / "posescript" / "src" / "text2pose"
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(posescript_src_dir))
        
        # Generate PoseScript vocabulary
        print("   Generating PoseScript vocabulary...")
        posescript_vocab_cmd = [
            sys.executable, "vocab.py",
            "--dataset", "posescript",
            "--caption_files", "posescript_human_6293.json", "posescript_auto_100k.json",
            "--vocab_filename", "vocab_posescript_6293_auto100k.pkl",
            "--make_compatible_to_side_flip"
        ]
        
        # Add a smaller word list for testing
        word_list = [
            'a', 'person', 'standing', 'sitting', 'walking', 'running', 'jumping',
            'arm', 'arms', 'leg', 'legs', 'hand', 'hands', 'head', 'body',
            'left', 'right', 'up', 'down', 'forward', 'backward',
            'raised', 'lowered', 'extended', 'bent', 'straight',
            'position', 'pose', 'stance', 'balance', 'movement'
        ]
        
        posescript_vocab_cmd.extend(["--new_word_list"] + word_list)
        
        print(f"   Running command: {' '.join(posescript_vocab_cmd)}")
        result = subprocess.run(posescript_vocab_cmd, capture_output=True, text=True)
        
        print(f"   Return code: {result.returncode}")
        print(f"   stdout: {result.stdout}")
        if result.stderr:
            print(f"   stderr: {result.stderr}")
        
        if result.returncode == 0:
            print("   ✅ PoseScript vocabulary generated successfully!")
        else:
            print(f"   ⚠️  PoseScript vocabulary generation had issues but may have worked")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate vocabulary files: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_cwd)


def main():
    """Main function."""
    print("🚀 Generating PoseScript Vocabulary Files")
    print("=" * 50)
    
    success = generate_vocabulary_files()
    
    # Check if files were created
    import text2pose.config as config
    posescript_vocab = Path(config.POSESCRIPT_LOCATION) / "vocab_posescript_6293_auto100k.pkl"
    
    if posescript_vocab.exists():
        print(f"✅ Vocabulary file created: {posescript_vocab}")
    else:
        print(f"❌ Vocabulary file not found: {posescript_vocab}")
    
    return success


if __name__ == "__main__":
    main()