"""
Script to organize BreakHis dataset into train/val folders for PyTorch ImageFolder
Run this script ONCE before training
"""
import os
import shutil
import random
from pathlib import Path

# CONFIGURATION - UPDATE THESE PATHS!
BREAKHIS_ROOT = r'C:\Users\DELL\Desktop\Breast_Cancer_prediction\breast-cancer-detection\data\BreaKHis_v1\BreaKHis_v1\histology_slides\breast'

OUTPUT_DIR = r'C:\Users\DELL\Desktop\Breast_Cancer_prediction\breast-cancer-detection\data\organized'
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

def collect_images(base_path, class_name):
    """Collect all image paths from BreakHis subfolders"""
    images = []
    sob_path = os.path.join(base_path, class_name, 'SOB')
    
    if not os.path.exists(sob_path):
        print(f"Warning: {sob_path} does not exist!")
        return images
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(sob_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                images.append(os.path.join(root, file))
    
    return images

def organize_dataset():
    """Organize BreakHis dataset into train/val structure"""
    
    print("=" * 60)
    print("BreakHis Dataset Organizer")
    print("=" * 60)
    
    # Create output directories
    for split in ['train', 'val']:
        for cls in ['benign', 'malignant']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
    
    # Collect images for each class
    print("\n1. Collecting benign images...")
    benign_images = collect_images(BREAKHIS_ROOT, 'benign')
    print(f"   Found {len(benign_images)} benign images")
    
    print("\n2. Collecting malignant images...")
    malignant_images = collect_images(BREAKHIS_ROOT, 'malignant')
    print(f"   Found {len(malignant_images)} malignant images")
    
    if len(benign_images) == 0 or len(malignant_images) == 0:
        print("\nERROR: No images found! Please check your BREAKHIS_ROOT path.")
        print(f"Current path: {BREAKHIS_ROOT}")
        return
    
    # Shuffle and split
    print("\n3. Shuffling and splitting dataset...")
    random.seed(42)  # For reproducibility
    random.shuffle(benign_images)
    random.shuffle(malignant_images)
    
    benign_split = int(len(benign_images) * TRAIN_SPLIT)
    malignant_split = int(len(malignant_images) * TRAIN_SPLIT)
    
    splits = {
        'train': {
            'benign': benign_images[:benign_split],
            'malignant': malignant_images[:malignant_split]
        },
        'val': {
            'benign': benign_images[benign_split:],
            'malignant': malignant_images[malignant_split:]
        }
    }
    
    # Copy files
    print("\n4. Copying files to organized structure...")
    for split_name, classes in splits.items():
        for class_name, images in classes.items():
            dest_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            print(f"   Copying {len(images)} {class_name} images to {split_name}/")
            
            for img_path in images:
                filename = os.path.basename(img_path)
                dest_path = os.path.join(dest_dir, filename)
                shutil.copy2(img_path, dest_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET ORGANIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nDataset split:")
    print(f"  Training:")
    print(f"    - Benign: {len(splits['train']['benign'])} images")
    print(f"    - Malignant: {len(splits['train']['malignant'])} images")
    print(f"  Validation:")
    print(f"    - Benign: {len(splits['val']['benign'])} images")
    print(f"    - Malignant: {len(splits['val']['malignant'])} images")
    print(f"\nTotal images: {len(benign_images) + len(malignant_images)}")
    print("\nYou can now run train.py with --data_dir pointing to:")
    print(f"  data/organized")

if __name__ == '__main__':
    organize_dataset()
