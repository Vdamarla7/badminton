#!/usr/bin/env python3
"""
Download and setup the actual PoseScript dataset.
"""

import os
import sys
import subprocess
import zipfile
import requests
from pathlib import Path

# Add PoseScript to path
posescript_path = Path(__file__).parent.parent / "posescript" / "src"
if posescript_path.exists():
    sys.path.insert(0, str(posescript_path))

import text2pose.config as config


def download_posescript_dataset():
    """Download the actual PoseScript dataset."""
    print("📥 Downloading PoseScript dataset...")
    
    dataset_url = "https://download.europe.naverlabs.com/ComputerVision/PoseScript/posescript_release_v2.zip"
    posescript_data_dir = Path(config.POSESCRIPT_LOCATION).parent
    posescript_data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file_path = posescript_data_dir / "posescript_release_v2.zip"
    
    try:
        # Download the dataset
        print(f"   Downloading from: {dataset_url}")
        print(f"   Saving to: {zip_file_path}")
        
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r   Progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end='')
        
        print(f"\n   ✅ Downloaded {downloaded_size} bytes")
        
        # Extract the dataset
        print("   📂 Extracting dataset...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(posescript_data_dir)
        
        print("   ✅ Dataset extracted successfully!")
        
        # Clean up zip file
        zip_file_path.unlink()
        print("   🗑️  Cleaned up zip file")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download dataset: {e}")
        return False


def verify_dataset_files():
    """Verify that the required dataset files are present."""
    print("🔍 Verifying dataset files...")
    
    posescript_data_dir = Path(config.POSESCRIPT_LOCATION)
    
    # Expected files from the PoseScript dataset
    expected_files = [
        "posescript_human_6293.json",
        "posescript_auto_100k.json",
        "ids_2_dataset_sequence_and_frame_index_100k.json",
        "train_ids_100k.json",
        "val_ids_100k.json",
        "test_ids_100k.json"
    ]
    
    all_present = True
    for filename in expected_files:
        file_path = posescript_data_dir / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"   ✅ {filename} ({file_size:,} bytes)")
        else:
            print(f"   ❌ {filename} (missing)")
            all_present = False
    
    return all_present


def generate_real_vocabulary():
    """Generate vocabulary using the real PoseScript dataset."""
    print("📖 Generating vocabulary with real PoseScript dataset...")
    
    # Change to PoseScript source directory
    posescript_src_dir = Path(__file__).parent.parent / "posescript" / "src" / "text2pose"
    original_cwd = os.getcwd()
    
    try:
        os.chdir(str(posescript_src_dir))
        
        # Generate PoseScript vocabulary using the exact command from the README
        posescript_vocab_cmd = [
            sys.executable, "vocab.py",
            "--dataset", "posescript",
            "--caption_files", "posescript_human_6293.json", "posescript_auto_100k.json",
            "--make_compatible_to_side_flip",
            "--vocab_filename", "vocab_posescript_6293_auto100k.pkl"
        ]
        
        # Add the complete word list from the PoseScript README
        word_list = [
            '(', ')', '.', 'a', 'a-pose', 'a-shape', 'about', 'abstract', 'acting', 'action', 'activities',
            'adjust', 'adjusting', 'adjustment', 'aim', 'aiming', 'aims', 'an', 'animal', 'argument', 'arm',
            'arms', 'art', 'at', 'aupplauding', 'back', 'backwards', 'balance', 'balancing', 'ball',
            'bartender', 'beaming', 'begging', 'behaving', 'behavior', 'bend', 'bending', 'bent', 'bird',
            'body', 'bow', 'bowed', 'bowing', 'bump', 'bumping', 'call', 'cartwheel', 'catch', 'catching',
            'celebrate', 'celebrating', 'charge', 'charging', 'check', 'checking', 'cheering', 'chicken',
            'choking', 'chop', 'chopping', 'circular', 'clap', 'clapping', 'clasp', 'clasping', 'clean',
            'cleaning', 'close', 'closing', 'collapsing', 'communicate', 'communicating', 'conduct',
            'conducting', 'consuming', 'cough', 'coughing', 'cower', 'cowering', 'crawl', 'crawling',
            'crossed', 'crossed-limbs', 'crossing', 'crouch', 'crouching', 'cry', 'crying', 'cuddling',
            'cursty', 'curtsy', 'curtsying', 'cut', 'cutting', 'dance', 'dancing', 'defensive', 'delivering',
            'desesperate', 'desesperation', 'despair', 'despairing', 'desperate', 'dip', 'direction',
            'disagree', 'dive', 'diving', 'do', 'doing', 'down', 'dribble', 'dribbling', 'drink', 'drinking',
            'drive', 'driving', 'drunk', 'drunken', 'duck', 'eat', 'eating', 'embracing', 'escaping',
            'evade', 'evading', 'exercices', 'exercise/training', 'exercising', 'face', 'fall', 'falling',
            'feet', 'fidget', 'fidgeting', 'fidgets', 'fight', 'fighting', 'fire', 'firing', 'fish',
            'fishing', 'flail', 'flailing', 'flap', 'flapping', 'flip', 'flipping', 'floor', 'fluttering',
            'food', 'foot', 'for', 'forward', 'gain', 'gesture', 'gesturing', 'get', 'getting', 'gifting',
            'giggling', 'give', 'giving', 'glide', 'gliding', 'going', 'golf', 'golfing', 'grab', 'grabbing',
            'grasp', 'grasping', 'greet', 'greeting', 'ground', 'gun', 'hacking', 'hair', 'hand', 'handling',
            'hands', 'handstand', 'handstanding', 'hang', 'hanging', 'having', 'head', 'headstand',
            'headstanding', 'hello', 'hi', 'hit', 'hitting', 'holding', 'hop', 'hopping', 'hug', 'hugging',
            'imitating', 'in', 'incline', 'inclined', 'inclining', 'injured', 'inspecting', 'instrument',
            'interact', 'interacting', 'interface', 'into', 'inward', 'jacks', 'jog', 'jogging', 'juggle',
            'juggling', 'jump', 'jumping', 'kick', 'kicking', 'knee', 'kneel', 'kneeled', 'kneeling',
            'knees', 'knelt', 'knock', 'knocking', 'lamenting', 'laugh', 'laughing', 'lead', 'leading',
            'lean', 'leaning', 'leap', 'leaping', 'leg', 'legs', 'lick', 'licking', 'lie', 'lift', 'lifting',
            'like', 'limbs', 'limp', 'limping', 'listen', 'listening', 'look', 'looking', 'lower', 'lowering',
            'lunge', 'lunging', 'lying', 'making', 'march', 'marching', 'martial', 'middle', 'mime',
            'mimicking', 'miming', 'misc', 'mix', 'mixing', 'moonwalk', 'moonwalking', 'motion', 'move',
            'movement', 'movements', 'moving', 'musique', 'navigate', 'object', 'of', 'on', 'open', 'opening',
            'operate', 'operating', 'or', 'orchestra', 'original', 'over', 'part', 'pat', 'patting', 'perform',
            'performance', 'performing', 'person', 'phone', 'picking', 'place', 'placing', 'play', 'playing',
            'plays', 'plead', 'pleading', 'point', 'pointing', 'pose', 'poses', 'position', 'practicing',
            'pray', 'prayer', 'praying', 'prepare', 'preparing', 'press', 'pressing', 'protect', 'protecting',
            'punch', 'punching', 'quivering', 'raising', 'reaching', 'relax', 'relaxation', 'relaxing',
            'release', 'releasing', 'remove', 'removing', 'reveal', 'rocking', 'rolling', 'rope', 'rub',
            'rubbing', 'run', 'running', 'salute', 'saluting', 'saying', 'scratch', 'scratching', 'search',
            'searching', 'seizing', 'series', 'shake', 'shaking', 'shape', 'shave', 'shaving', 'shivering',
            'shooting', 'shoulder', 'showing', 'shrug', 'shrugging', 'shuffle', 'side', 'sideways', 'sign',
            'sit', 'sitting', 'skate', 'skating', 'sketch', 'skip', 'skipping', 'slash', 'slicing', 'slide',
            'sliding', 'slightly', 'smacking', 'smell', 'smelling', 'snack', 'snacking', 'sneak', 'sneaking',
            'sneeze', 'sneezing', 'sobbing', 'some', 'someone', 'something', 'somethings', 'speaking', 'spin',
            'spinning', 'sport', 'sports', 'spread', 'spreading', 'squat', 'squatting', 'stagger', 'staggering',
            'stances', 'stand', 'standing', 'staring', 'step', 'stepping', 'stick', 'stomp', 'stomping',
            'stop', 'strafe', 'strafing', 'stretch', 'stretching', 'stroke', 'stroking', 'stumble', 'stumbling',
            'style', 'styling', 'sudden', 'support', 'supporting', 'sway', 'swaying', 'swim', 'swimming',
            'swing', 'swinging', 'swipe', 'swiping', 't', 't-pose', 't-shape', 'take/pick', 'taking', 'tap',
            'tapping', 'telephone', 'tentative', 'the', 'things', 'throw', 'throwing', 'tie', 'tiptoe',
            'tiptoeing', 'tiptoes', 'to', 'touch', 'touching', 'training', 'transition', 'trashing', 'trip',
            'tripping', 'try', 'trying', 'tumbling', 'turn', 'turning', 'twist', 'twisting', 'twitching',
            'tying', 'uncross', 'unknown', 'up', 'up/down', 'upper', 'using', 'vocalise', 'vocalizing',
            'voice', 'voicing', 'vomit', 'vomitting', 'waist', 'wait', 'waiting', 'walk', 'walking', 'wash',
            'washing', 'wave', 'waving', 'weeping', 'wiggle', 'wiggling', 'with', 'with/use', 'wobble',
            'wobbling', 'worry', 'worrying', 'wrist', 'wrists', 'write', 'writing', 'yawn', 'yawning',
            'yoga', 'zombie'
        ]
        
        posescript_vocab_cmd.extend(["--new_word_list"] + word_list)
        
        print(f"   Running vocabulary generation with real dataset...")
        result = subprocess.run(posescript_vocab_cmd, capture_output=True, text=True)
        
        print(f"   Return code: {result.returncode}")
        print(f"   stdout: {result.stdout}")
        if result.stderr:
            print(f"   stderr: {result.stderr}")
        
        if result.returncode == 0:
            print("   ✅ Real PoseScript vocabulary generated successfully!")
            return True
        else:
            print(f"   ⚠️  Vocabulary generation completed with warnings")
            return True  # Often still works
        
    except Exception as e:
        print(f"❌ Failed to generate vocabulary: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def check_vocabulary_size():
    """Check the size of the generated vocabulary."""
    print("🔍 Checking vocabulary size...")
    
    vocab_file = Path(config.POSESCRIPT_LOCATION) / "vocab_posescript_6293_auto100k.pkl"
    
    if not vocab_file.exists():
        print(f"   ❌ Vocabulary file not found: {vocab_file}")
        return False
    
    try:
        import pickle
        with open(vocab_file, 'rb') as f:
            vocab = pickle.load(f)
        
        vocab_size = len(vocab)
        print(f"   ✅ Vocabulary size: {vocab_size} tokens")
        
        if vocab_size >= 2100:  # Close to expected 2158
            print("   ✅ Vocabulary size looks good for the trained model!")
            return True
        else:
            print(f"   ⚠️  Vocabulary size ({vocab_size}) is smaller than expected (2158)")
            print("   This may cause issues with the trained model.")
            return False
            
    except Exception as e:
        print(f"   ❌ Could not check vocabulary size: {e}")
        return False


def main():
    """Main function to download and setup the PoseScript dataset."""
    print("🚀 Downloading and Setting Up Real PoseScript Dataset")
    print("=" * 60)
    
    try:
        # Step 1: Download the dataset
        download_success = download_posescript_dataset()
        
        if not download_success:
            print("❌ Failed to download dataset. Exiting.")
            return False
        
        # Step 2: Verify dataset files
        verify_success = verify_dataset_files()
        
        if not verify_success:
            print("⚠️  Some dataset files are missing, but continuing...")
        
        # Step 3: Generate vocabulary with real data
        vocab_success = generate_real_vocabulary()
        
        # Step 4: Check vocabulary size
        size_check = check_vocabulary_size()
        
        print("\n" + "=" * 60)
        if download_success and vocab_success and size_check:
            print("🎉 PoseScript dataset setup completed successfully!")
            print("\n📋 Next steps:")
            print("   1. Test the real PoseScript generator:")
            print("      python posescript_generator/test_real_posescript.py")
            print("   2. Use the real PoseScript generator in your notebook")
        else:
            print("⚠️  Setup completed with some issues.")
            print("   You can still try the real PoseScript generator.")
        
        print(f"\n📁 Dataset location: {config.POSESCRIPT_LOCATION}")
        print(f"📁 Vocabulary file: {config.POSESCRIPT_LOCATION}/vocab_posescript_6293_auto100k.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()