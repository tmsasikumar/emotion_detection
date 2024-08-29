from flask import Flask, request, jsonify
from src.api.inference import predict_mood

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        voice_data = request.files.get('voice')
        face_data = request.files.get('face')
        if voice_data and face_data:
            prediction = predict_mood(voice_data, face_data)
            return jsonify({'mood': prediction})
        else:
            return jsonify({'error': 'Voice and face data are required'}), 400

if __name__ == '__main__':
    app.run(debug=True)
