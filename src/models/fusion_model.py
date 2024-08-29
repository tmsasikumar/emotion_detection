from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense, Input

def create_fusion_model(voice_model, face_model):
    # Get the output of each model
    voice_output = voice_model.output
    face_output = face_model.output

    # Concatenate the outputs
    combined = Concatenate()([voice_output, face_output])

    # Add a dense layer on top
    z = Dense(100, activation='relu')(combined)
    z = Dense(10, activation='softmax')(z)

    # Create the final model
    fusion_model = Model(inputs=[voice_model.input, face_model.input], outputs=z)
    fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return fusion_model
