"""
Flask Web App for Alzheimer's Risk Prediction
Allows upload of audio, triggers feature extraction and inference, displays results.
"""
import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))
from audio_feature_extraction import extract_audio_features
import joblib
import subprocess
import time

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'webm'}
MODEL_PATH = 'D:/Docathon/models/alzheimers_model.joblib'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('../frontend_web', 'index.html')


# Function to load the trained model

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Function to predict risk using the model and audio features

def predict_risk(audio_features, model):
    try:
        import numpy as np
        # Ensure audio_features is a flat list/array, not a dict
        if isinstance(audio_features, dict):
            audio_features = list(audio_features.values())
        # Flatten nested lists/arrays if present
        flat_features = []
        for f in audio_features:
            if isinstance(f, (list, np.ndarray)):
                flat_features.extend(np.array(f).flatten().tolist())
            else:
                flat_features.append(f)
        # Ensure all elements are numeric
        flat_features = [float(x) for x in flat_features]
        risk_score = model.predict([flat_features])
        return risk_score[0]
    except Exception as e:
        raise RuntimeError(f"Failed to predict risk: {str(e)}")
        
@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files.get('audio')
    if not audio_file or not allowed_file(audio_file.filename):
        return 'Invalid or missing audio file', 400
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f"{int(time.time())}_" + audio_file.filename))
    audio_file.save(audio_path)
    # Convert webm to wav using FFmpeg
    if audio_path.endswith('.webm'):
        wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        subprocess.run(['C:/Program Files/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe', '-i', audio_path, wav_path], check=True)
        audio_path = wav_path
    try:
        audio_features = extract_audio_features(audio_path)
    except Exception as e:
        return f'Error during audio feature extraction: {str(e)}', 500
    try:
        model = load_model(MODEL_PATH)
        risk_score = predict_risk(audio_features, model)
        # Convert risk_score to binary classification
        if isinstance(risk_score, str):
            classification = risk_score
        else:
            classification = 'Alzheimer' if risk_score > 0.5 else 'Not Alzheimer'
    except Exception as e:
        return f'Error during inference: {str(e)}', 500
    return jsonify({'classification': classification})

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('../frontend_web', filename)

if __name__ == '__main__':
    app.run(debug=True)


