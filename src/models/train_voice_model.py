import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

def create_voice_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_voice_model(X_train, y_train):
    model = create_voice_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    model.save('../models/voice_model.h5')
    return model
