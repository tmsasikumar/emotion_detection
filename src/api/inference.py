import numpy as np
from tensorflow.keras.models import load_model
from src.data.preprocess_voice import extract_mfcc
from src.data.preprocess_face import preprocess_face_image

voice_model = load_model('../models/voice_model.h5')
face_model = load_model('../models/face_model.h5')

def predict_mood(voice_data, face_data):
    # Preprocess inputs
    voice_mfcc = extract_mfcc(voice_data)
    face_image = preprocess_face_image(face_data)

    # Reshape for model
    voice_mfcc = voice_mfcc.reshape(1, -1, 1)
    face_image = face_image.reshape(1, 48, 48, 3)

    # Predict
    voice_prediction = voice_model.predict(voice_mfcc)
    face_prediction = face_model.predict(face_image)

    # Combine predictions
    mood = np.argmax((voice_prediction + face_prediction) / 2)

    return mood
