"""
Deep exploration of BreakHis folder structure
"""
import os

def explore_directory(path, max_depth=4, current_depth=0, prefix=""):
    """Recursively explore directory structure"""
    if current_depth >= max_depth:
        return
    
    try:
        items = os.listdir(path)
        for item in items[:20]:  # Limit to first 20 items per folder
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"{prefix}📁 {item}/")
                explore_directory(item_path, max_depth, current_depth + 1, prefix + "  ")
            else:
                # Show file extension
                print(f"{prefix}📄 {item}")
    except PermissionError:
        print(f"{prefix}❌ Permission denied")

# Start exploring from data folder
base = r'C:\Users\DELL\Desktop\Breast_Cancer_prediction\breast-cancer-detection\data'

print("=" * 70)
print("COMPLETE FOLDER STRUCTURE EXPLORATION")
print("=" * 70)
print(f"\nExploring: {base}\n")

explore_directory(base, max_depth=5)

print("\n" + "=" * 70)
print("\nLooking for image files (.png, .jpg, .tif)...")
print("=" * 70)

# Search for actual image files
image_count = 0
sample_paths = []

for root, dirs, files in os.walk(base):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_count += 1
            if len(sample_paths) < 5:  # Collect first 5 image paths
                sample_paths.append(os.path.join(root, file))

print(f"\nTotal images found: {image_count}")
print("\nSample image paths:")
for path in sample_paths:
    print(f"  {path}")

if sample_paths:
    # Extract the correct base path
    first_image = sample_paths[0]
    print(f"\n" + "=" * 70)
    print("RECOMMENDED PATH FOR organize_data.py:")
    print("=" * 70)
    
    # Try to find where benign/malignant split happens
    parts = first_image.split(os.sep)
    
    # Find 'benign' or 'malignant' in path
    for i, part in enumerate(parts):
        if part in ['benign', 'malignant']:
            recommended_path = os.sep.join(parts[:i])
            print(f"\nBREAKHIS_ROOT = r'{recommended_path}'")
            break
