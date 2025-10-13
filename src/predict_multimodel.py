import numpy as np
from tensorflow.keras.models import load_model
from image_preprocessing import load_and_preprocess_image
import pandas as pd

def predict(image_path, tabular_features, model_path="../models/multimodal_model.h5"):
    model = load_model(model_path)
    
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Batch dimension
    
    tabular_features = np.array(tabular_features).reshape(1, -1)
    
    pred = model.predict([img, tabular_features])
    result = "Malignant (Cancerous)" if pred[0][0] >= 0.5 else "Benign (Non-Cancerous)"
    print(f"🔍 Prediction: {result}")
    return result

# Example usage
if __name__ == "__main__":
    # Replace with your image path and tabular row (30 features)
    image_path = "../data/raw/images/sample1.jpg"
    tabular_features = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                        25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    predict(image_path, tabular_features)

