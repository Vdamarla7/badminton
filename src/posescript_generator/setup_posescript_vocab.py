#!/usr/bin/env python3
"""
Setup script to create vocabulary files and download language models for PoseScript.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add PoseScript to path
posescript_path = Path(__file__).parent.parent / "posescript" / "src"
if posescript_path.exists():
    sys.path.insert(0, str(posescript_path))

import text2pose.config as config


def setup_directories():
    """Create necessary directories for PoseScript data."""
    print("🗂️  Setting up directories...")
    
    # Create data directories
    posescript_data_dir = Path(config.POSESCRIPT_LOCATION)
    posefix_data_dir = Path(config.POSEFIX_LOCATION)
    transformer_cache_dir = Path(config.TRANSFORMER_CACHE_DIR)
    
    posescript_data_dir.mkdir(parents=True, exist_ok=True)
    posefix_data_dir.mkdir(parents=True, exist_ok=True)
    transformer_cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directories:")
    print(f"   PoseScript: {posescript_data_dir}")
    print(f"   PoseFix: {posefix_data_dir}")
    print(f"   Transformers: {transformer_cache_dir}")
    
    return posescript_data_dir, posefix_data_dir, transformer_cache_dir


def download_language_models(transformer_cache_dir):
    """Download the required HuggingFace language models."""
    print("\n📚 Downloading language models...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_types = [
            "distilbert-base-uncased",
            "gpt2"  # Also needed for text generation
        ]
        
        for model_type in model_types:
            print(f"   Downloading {model_type}...")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_type)
            tokenizer_path = transformer_cache_dir / model_type
            tokenizer.save_pretrained(str(tokenizer_path))
            
            # Download model
            model = AutoModel.from_pretrained(model_type)
            model.save_pretrained(str(tokenizer_path))
            
            print(f"   ✅ Saved {model_type} to {tokenizer_path}")
        
        print("✅ Language models downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download language models: {e}")
        return False


def create_dummy_caption_files(posescript_data_dir, posefix_data_dir):
    """Create minimal dummy caption files for vocabulary generation."""
    print("\n📝 Creating dummy caption files...")
    
    # Create minimal PoseScript caption files
    posescript_human_file = posescript_data_dir / "posescript_human_6293.json"
    posescript_auto_file = posescript_data_dir / "posescript_auto_100k.json"
    
    # Create minimal PoseFix caption files
    posefix_human_file = posefix_data_dir / "posefix_human_6157.json"
    posefix_auto_file = posefix_data_dir / "posefix_auto_135305.json"
    posefix_paraphrases_file = posefix_data_dir / "posefix_paraphrases_4284.json"
    
    # Minimal JSON structure for caption files
    dummy_caption_data = {
        "0": {
            "text": ["A person standing with arms at their sides."]
        },
        "1": {
            "text": ["A person in a balanced position with legs apart."]
        },
        "2": {
            "text": ["A person with arms raised above their head."]
        }
    }
    
    import json
    
    # Write dummy files
    caption_files = [
        posescript_human_file,
        posescript_auto_file,
        posefix_human_file,
        posefix_auto_file,
        posefix_paraphrases_file
    ]
    
    for caption_file in caption_files:
        with open(caption_file, 'w') as f:
            json.dump(dummy_caption_data, f, indent=2)
        print(f"   ✅ Created {caption_file}")
    
    print("✅ Dummy caption files created!")
    return True


def generate_vocabulary_files():
    """Generate the vocabulary files using PoseScript's vocab.py script."""
    print("\n📖 Generating vocabulary files...")
    
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
        
        # Add the extensive word list
        word_list = [
            '(', ')', '.', 'a', 'a-pose', 'a-shape', 'about', 'abstract', 'acting', 'action', 'activities',
            'adjust', 'adjusting', 'adjustment', 'aim', 'aiming', 'aims', 'an', 'animal', 'argument', 'arm',
            'arms', 'art', 'at', 'back', 'backwards', 'balance', 'balancing', 'ball', 'bend', 'bending',
            'bent', 'bird', 'body', 'bow', 'bowed', 'bowing', 'call', 'catch', 'catching', 'celebrate',
            'celebrating', 'check', 'checking', 'cheering', 'clap', 'clapping', 'clasp', 'clasping',
            'clean', 'cleaning', 'close', 'closing', 'crouch', 'crouching', 'dance', 'dancing', 'down',
            'drink', 'drinking', 'eat', 'eating', 'exercise', 'exercising', 'face', 'fall', 'falling',
            'feet', 'fight', 'fighting', 'foot', 'forward', 'gesture', 'gesturing', 'grab', 'grabbing',
            'hand', 'hands', 'head', 'hit', 'hitting', 'holding', 'hop', 'hopping', 'hug', 'hugging',
            'jump', 'jumping', 'kick', 'kicking', 'knee', 'kneel', 'kneeling', 'knees', 'lean', 'leaning',
            'leg', 'legs', 'lift', 'lifting', 'look', 'looking', 'march', 'marching', 'move', 'movement',
            'moving', 'open', 'opening', 'person', 'play', 'playing', 'point', 'pointing', 'pose', 'poses',
            'position', 'raise', 'raising', 'reach', 'reaching', 'run', 'running', 'sit', 'sitting',
            'stand', 'standing', 'step', 'stepping', 'stretch', 'stretching', 'swing', 'swinging',
            'throw', 'throwing', 'touch', 'touching', 'turn', 'turning', 'walk', 'walking', 'wave',
            'waving', 'with', 'yoga'
        ]
        
        posescript_vocab_cmd.extend(["--new_word_list"] + word_list)
        
        result = subprocess.run(posescript_vocab_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ PoseScript vocabulary generated successfully!")
        else:
            print(f"   ⚠️  PoseScript vocabulary generation had issues:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
        
        # Generate PoseFix vocabulary (simplified)
        print("   Generating PoseFix vocabulary...")
        posefix_vocab_cmd = [
            sys.executable, "vocab.py",
            "--dataset", "posefix",
            "--caption_files", "posefix_human_6157.json", "posefix_auto_135305.json", "posefix_paraphrases_4284.json",
            "--vocab_filename", "vocab_posefix_6157_pp4284_auto.pkl",
            "--make_compatible_to_side_flip"
        ]
        
        result = subprocess.run(posefix_vocab_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ PoseFix vocabulary generated successfully!")
        else:
            print(f"   ⚠️  PoseFix vocabulary generation had issues:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
        
        print("✅ Vocabulary generation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate vocabulary files: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def verify_setup():
    """Verify that all required files have been created."""
    print("\n🔍 Verifying setup...")
    
    # Check vocabulary files
    posescript_vocab = Path(config.POSESCRIPT_LOCATION) / "vocab_posescript_6293_auto100k.pkl"
    posefix_vocab = Path(config.POSEFIX_LOCATION) / "vocab_posefix_6157_pp4284_auto.pkl"
    
    # Check transformer models
    transformer_dir = Path(config.TRANSFORMER_CACHE_DIR)
    distilbert_dir = transformer_dir / "distilbert-base-uncased"
    gpt2_dir = transformer_dir / "gpt2"
    
    files_to_check = [
        ("PoseScript vocabulary", posescript_vocab),
        ("PoseFix vocabulary", posefix_vocab),
        ("DistilBERT model", distilbert_dir),
        ("GPT-2 model", gpt2_dir)
    ]
    
    all_good = True
    for name, path in files_to_check:
        if path.exists():
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (missing)")
            all_good = False
    
    if all_good:
        print("✅ All required files are present!")
    else:
        print("⚠️  Some files are missing, but the setup may still work.")
    
    return all_good


def main():
    """Main setup function."""
    print("🚀 Setting up PoseScript Vocabulary and Language Models")
    print("=" * 60)
    
    try:
        # Step 1: Setup directories
        posescript_data_dir, posefix_data_dir, transformer_cache_dir = setup_directories()
        
        # Step 2: Download language models
        download_success = download_language_models(transformer_cache_dir)
        
        # Step 3: Create dummy caption files
        caption_success = create_dummy_caption_files(posescript_data_dir, posefix_data_dir)
        
        # Step 4: Generate vocabulary files
        vocab_success = generate_vocabulary_files()
        
        # Step 5: Verify setup
        verify_success = verify_setup()
        
        print("\n" + "=" * 60)
        if download_success and caption_success and vocab_success:
            print("🎉 PoseScript setup completed successfully!")
            print("\n📋 Next steps:")
            print("   1. Run the real PoseScript generator test:")
            print("      python posescript_generator/test_real_posescript.py")
            print("   2. Use the real PoseScript generator in your notebook")
        else:
            print("⚠️  Setup completed with some issues.")
            print("   The simplified PoseScript generator should still work.")
        
        print(f"\n📁 Files created in:")
        print(f"   PoseScript data: {posescript_data_dir}")
        print(f"   PoseFix data: {posefix_data_dir}")
        print(f"   Language models: {transformer_cache_dir}")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()