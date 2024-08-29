import librosa
import numpy as np

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def preprocess_voice_data(data):
    features = []
    for file_path in data:
        mfcc = extract_mfcc(file_path)
        features.append(mfcc)
    return np.array(features)
