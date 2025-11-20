import os

val_dir = r'C:\Users\DELL\Desktop\Breast_Cancer_prediction\breast-cancer-detection\data\organized\val'

print("="*70)
print("AVAILABLE VALIDATION IMAGES")
print("="*70)

for class_name in ['benign', 'malignant']:
    class_dir = os.path.join(val_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    print(f"\n{class_name.upper()} images ({len(images)} total):")
    print("-"*70)
    
    # Show first 5
    for i, img in enumerate(images[:5]):
        full_path = os.path.join(class_dir, img)
        print(f"{i+1}. {img}")
        print(f"   Path: data/organized/val/{class_name}/{img}")
    
    print(f"\n... and {len(images)-5} more images\n")
