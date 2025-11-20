"""
Streamlit Web Interface for Breast Cancer Detection System
A beautiful, user-friendly frontend for your minor project

Run with: streamlit run app.py
"""

import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .benign {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .malignant {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .result-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path='best_model.pth'):
    """Load the trained model (cached for efficiency)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        class_names = checkpoint.get('class_names', ['benign', 'malignant'])
        
        return model, class_names, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def preprocess_image(image):
    """Preprocess image for prediction"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def predict(model, image_tensor, class_names, device):
    """Make prediction on image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        pred_class = class_names[predicted.item()]
        conf_score = confidence.item() * 100
        probs = probabilities[0].cpu().numpy()
        
        return pred_class, conf_score, probs

def main():
    # Header
    st.markdown('<p class="main-header">🔬 Breast Cancer Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Histopathological Image Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/microscope.png", width=100)
        st.title("About")
        st.info("""
        This AI system classifies breast cancer histopathological images as:
        - **Benign** (non-cancerous)
        - **Malignant** (cancerous)
        
        **Model Performance:**
        - Accuracy: 97.66%
        - Dataset: BreakHis (7,909 images)
        - Architecture: ResNet-18
        """)
        
        st.warning("""
        ⚠️ **Disclaimer:**
        This is an educational tool. Always consult medical professionals for diagnosis.
        """)
    
    # Load model
    model, class_names, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'best_model.pth' exists in the current directory.")
        return
    
    st.success(f"✅ Model loaded successfully! Using device: {device}")
    
    # File uploader
    st.markdown("---")
    st.subheader("📤 Upload Histopathological Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image (PNG, JPG, JPEG, TIF)",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a breast tissue histopathological image for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
            st.caption(f"Filename: {uploaded_file.name}")
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Predict button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess and predict
                    image_tensor = preprocess_image(image)
                    pred_class, confidence, probabilities = predict(model, image_tensor, class_names, device)
                    
                    # Display results
                    if pred_class.lower() == 'benign':
                        st.markdown(f"""
                        <div class="prediction-box benign">
                            <p class="result-text">✅ BENIGN</p>
                            <p class="confidence-text">Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box malignant">
                            <p class="result-text">⚠️ MALIGNANT</p>
                            <p class="confidence-text">Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability chart
                    st.subheader("📊 Class Probabilities")
                    prob_data = {
                        'Class': [cls.capitalize() for cls in class_names],
                        'Probability (%)': [float(p)*100 for p in probabilities]
                    }
                    
                    st.bar_chart(prob_data, x='Class', y='Probability (%)', use_container_width=True)
                    
                    # Detailed probabilities
                    st.markdown("**Detailed Probabilities:**")
                    for i, cls in enumerate(class_names):
                        st.metric(
                            label=cls.capitalize(),
                            value=f"{float(probabilities[i])*100:.2f}%"
                        )
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a histopathological image to begin analysis")
        
        # Sample images section
        st.markdown("---")
        st.subheader("📝 Sample Images for Testing")
        st.markdown("""
        You can test the system with images from:
        - `data/organized/val/benign/` - For benign samples
        - `data/organized/val/malignant/` - For malignant samples
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🎓 Minor Project - Breast Cancer Detection using Deep Learning</p>
        <p>Powered by PyTorch & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
