"""
Single Image Prediction Script for Breast Cancer Detection
This is the main script for your minor project demo!

Usage: python predict.py --image path_to_image.png --model best_model.pth
"""
import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


def load_model(model_path, device):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    class_names = checkpoint.get('class_names', ['benign', 'malignant'])
    
    return model, class_names


def preprocess_image(image_path):
    """Preprocess image for prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def predict(model, image_tensor, class_names, device):
    """Make prediction on a single image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        pred_class = class_names[predicted.item()]
        conf_score = confidence.item() * 100
        
        # FIX: Extract first row and convert to list
        return pred_class, conf_score, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Breast Cancer Detection - Single Image Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth file)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("BREAST CANCER DETECTION SYSTEM")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    
    # Load model
    print("\nLoading model...")
    model, class_names = load_model(args.model, device)
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    
    # Preprocess image
    print("\nPreprocessing image...")
    image_tensor = preprocess_image(args.image)
    
    if image_tensor is None:
        print("Failed to load image!")
        return
    
    # Make prediction
    print("Running inference...")
    pred_class, confidence, probabilities = predict(model, image_tensor, class_names, device)
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"\nPrediction: {pred_class.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nClass Probabilities:")
    # FIX: probabilities is now a 1D array, so direct indexing works
    for i, cls in enumerate(class_names):
        print(f"  {cls.capitalize():>10s}: {float(probabilities[i])*100:.2f}%")
    
    print("\n" + "=" * 70)
    
    # Medical disclaimer
    print("\nNOTE: This is an AI-based screening tool for educational purposes.")
    print("Always consult with medical professionals for accurate diagnosis.")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
