import os
import numpy as np
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=(224,224)):
    """
    Load an image, resize and normalize it
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to 0-1
    return img_array

def load_images_from_folder(folder_path):
    """
    Load all images in a folder
    """
    images = []
    filenames = []
    for fname in os.listdir(folder_path):
        if fname.endswith(('.jpg', '.png')):
            img = load_and_preprocess_image(os.path.join(folder_path, fname))
            images.append(img)
            filenames.append(fname)
    return np.array(images), filenames
