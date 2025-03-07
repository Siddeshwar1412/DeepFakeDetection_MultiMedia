import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Constants
MODEL_PATH = "E:\Minor_project\FakeVideodetector\deepfake_video_model.h5"
FRAME_SIZE = (64, 64)
MAX_FRAMES = 30

# Load Trained Model
def load_video_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Loading trained video deepfake model...")
        return load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("🔥 Model file not found. Train the model first!")

video_model = load_video_model()

# Flask App Setup
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"])  # Allow frontend

# Process Video for Prediction
def process_video(video_file, model):
    cap = cv2.VideoCapture(video_file)
    frames = []

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE) / 255.0  # Normalize
        frames.append(img_to_array(frame))

    cap.release()

    if len(frames) < MAX_FRAMES:
        return {"error": "Not enough frames detected in video."}, 400

    frames = np.array(frames).reshape(1, MAX_FRAMES, *FRAME_SIZE, 3)
    prediction = model.predict(frames)
    confidence = float(prediction[0][0]) * 100
    result = "Fake" if prediction[0][0] > 0.5 else "Real"

    return {"prediction": result, "confidence": f"{confidence:.2f}%"}, 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = "temp_video.mp4"
    file.save(file_path)

    response, status_code = process_video(file_path, video_model)
    os.remove(file_path)  # Cleanup
    return jsonify(response), status_code

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "API is running successfully."}), 200

if __name__ == "__main__":
    app.run(debug=True, port=8080)