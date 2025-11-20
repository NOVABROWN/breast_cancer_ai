"""
Script to find your BreakHis dataset location
"""
import os

# Base directory where we'll search
base_dir = r'C:\Users\DELL\Desktop\Breast_Cancer_prediction\breast-cancer-detection\data'

print("Searching for BreakHis dataset...")
print(f"Base directory: {base_dir}\n")

if os.path.exists(base_dir):
    print("Contents of data folder:")
    print("=" * 60)
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            print(f"\n📁 Folder: {item}")
            print(f"   Full path: {item_path}")
            
            # Check what's inside
            try:
                contents = os.listdir(item_path)
                print(f"   Contains: {contents[:10]}")  # Show first 10 items
                
                # Check for benign/malignant folders
                if 'benign' in contents and 'malignant' in contents:
                    print(f"   ✅ Found benign and malignant folders!")
                    print(f"\n   USE THIS PATH IN organize_data.py:")
                    print(f"   BREAKHIS_ROOT = r'{item_path}'")
                    
                    # Check deeper structure
                    benign_path = os.path.join(item_path, 'benign')
                    print(f"\n   Benign folder contents:")
                    print(f"   {os.listdir(benign_path)[:5]}")
                    
            except PermissionError:
                print(f"   (Cannot access)")
        else:
            print(f"📄 File: {item}")
else:
    print(f"ERROR: Directory does not exist: {base_dir}")
    print("\nPlease update base_dir in this script to match your actual data location.")
