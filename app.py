import os
import tensorflow as tf
import numpy as np
import cv2
import librosa
import time
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:5500"])  # Allow frontend

# ----------------- Configure Flask-Mail -----------------
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME", "internship1412@gmail.com")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD", "siddeshwar14#")
app.config["MAIL_DEFAULT_SENDER"] = app.config["MAIL_USERNAME"]
mail = Mail(app)

# ----------------- Load Deepfake Detection Models -----------------
try:
    print("✅ Loading models...")
    image_model = tf.keras.models.load_model(
        r"E:\Minor_project\FakeImageDetector\deepfake_detector.h5"
    )
    audio_model = tf.keras.models.load_model(
        r"E:\Minor_project\FakeAudioDetector\fake_audio_model.h5"
    )
    print("✅ All models loaded successfully!")

except Exception as e:
    print(f"🔥 Model loading error: {str(e)}")

# ----------------- Preprocessing Functions -----------------
def process_image(image_bytes):
    """Preprocess image for deepfake detection model."""
    try:
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224)) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = image_model.predict(img)
        result = "Fake" if prediction > 0.5 else "Real"
        print(f"✅ Image prediction: {result}")

        return jsonify({"prediction": result}), 200

    except Exception as e:
        print(f"❌ Image processing error: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500

def process_audio(file):
    """Preprocess and predict deepfake audio."""
    try:
        print(f"🔹 Processing audio file: {file.filename}")

        # Load audio and extract MFCCs
        audio, sr = librosa.load(file, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Ensure MFCC shape (40, 174)
        target_shape = (40, 174)

        # Pad or trim MFCCs
        if mfccs.shape[1] < target_shape[1]:  
            pad_width = target_shape[1] - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :target_shape[1]]  

        print(f"📊 Adjusted MFCCs shape: {mfccs.shape}")  # Should print (40, 174)

        # Reshape for model input
        mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
        mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension

        # Make prediction
        prediction = audio_model.predict(mfccs)

        # Extract the first value from the array
        pred_value = float(prediction[0][0])  

        # Classify based on threshold
        result = "Fake" if pred_value > 0.5 else "Real"
        print(f"✅ Audio prediction: {result}")

        return jsonify({"prediction": result}), 200

    except Exception as e:
        print(f"❌ Audio processing error: {str(e)}")
        return jsonify({"error": "Failed to process audio"}), 500

# ----------------- API Route for Deepfake Detection -----------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_type = request.form.get('fileType')  # Get the file type from the form data

    if file_type == "image":
        return process_image(file.read())
    elif file_type == "audio":
        return process_audio(file)
    else:
        return jsonify({"error": "Invalid file type"}), 400
    
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test successful"}), 200

# ----------------- API Route for Sending Email -----------------
@app.route("/send-email", methods=["POST"])
def send_email():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    message_content = data.get("message")

    if not name or not email or not message_content:
        return jsonify({"message": "All fields are required"}), 400

    try:
        msg = Message(
            "Contact Form Submission", recipients=["internship1412@gmail.com"]
        )
        msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message_content}"
        mail.send(msg)
        return jsonify({"message": "Email sent successfully!"}), 200
    except Exception as e:
        return jsonify({"message": "Failed to send email", "error": str(e)}), 500

# ----------------- Run Flask Server -----------------
if __name__ == "__main__":
    app.run(debug=True, port=8080)  # Ensure Flask is running on 8080