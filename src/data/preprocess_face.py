import cv2
import numpy as np

def preprocess_face_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (48, 48))  # Standard size for face recognition
    image = image / 255.0  # Normalize pixel values
    return image

def preprocess_face_data(data):
    features = []
    for _, row in data.iterrows():
        image_path = row['image_path']
        image = preprocess_face_image(image_path)
        features.append(image)
    return np.array(features)
