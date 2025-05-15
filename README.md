# Alzheimer's Risk Prediction Prototype

## Project Structure

- `backend/` - Backend processing modules (audio, eye tracking, ML inference)
- `models/` - Trained machine learning models
- `frontend/` - User interface (web app)
- `dataset/` - Audio and (future) eye tracking data
- `audio_feature_extraction.py` - Legacy script (to be modularized)

## Getting Started

1. Install dependencies for backend and frontend (see respective folders).
2. Run backend services for feature extraction and inference.
3. Start the frontend web app for user interaction.

## Modules

- **Audio Feature Extraction:** Extracts speech features for risk analysis.
- **Eye Movement Tracking:** (To be implemented) Processes video for gaze/eye metrics.
- **ML Inference:** Loads models and predicts Alzheimer's risk.
- **Frontend:** Uploads data, displays results, and provides explanations.

## Usage

- Place audio files in `dataset/alzehiemer` or `dataset/normal`.
- Run backend scripts to extract features and predict risk.
- Use the web interface for uploading and viewing results.

---

This is a prototype for research and demonstration purposes only.