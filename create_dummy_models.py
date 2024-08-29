import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, InputLayer

# Create and save a dummy voice model
def create_and_save_voice_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(13,)))  # Example input shape for MFCC features
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # Example output for 5 mood classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save('../models/voice_model.h5')
    print("Voice model saved as voice_model.h5")


# Create and save a dummy face model
def create_and_save_face_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(64, 64, 3)))  # Example input shape for a 64x64 RGB image
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # Example output for 5 mood classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save('../models/face_model.h5')
    print("Face model saved as face_model.h5")


# Create and save a dummy fusion model
def create_and_save_fusion_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(10,)))  # Example input shape for concatenated features
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # Example output for 5 mood classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.save('../models/fusion_model.h5')
    print("Fusion model saved as fusion_model.h5")


# Make sure the directory exists
import os
os.makedirs('../models', exist_ok=True)

# Create and save the models
create_and_save_voice_model()
create_and_save_face_model()
create_and_save_fusion_model()
