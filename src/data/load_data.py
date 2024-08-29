import os
import pandas as pd

def load_voice_data(data_dir):
    """Load raw voice data from the specified directory."""
    data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(data_dir, file_name)
            data.append(file_path)
    return data

def load_face_data(data_dir):
    """Load raw face data from the specified directory."""
    return pd.read_csv(os.path.join(data_dir, 'face_data.csv'))
