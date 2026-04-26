from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import math
from datetime import datetime
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

app = Flask(__name__)

# Load biomarkers model + scaler + label encoder
biomarker_model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

CNN_CLASSES = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
IMAGE_SIZE = (128, 128)

cnn_model = load_model("wbc_classification_model.keras") if load_model else None

def prepare_input(wbc, neut, lymph):
    wbc_log = math.log1p(wbc)
    nlr = neut / lymph if lymph != 0 else 0

    data = np.array([[wbc_log, neut, lymph, nlr]])
    data_scaled = scaler.transform(data)

    return data_scaled

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict-biomarkers', methods=['POST'])
def predict_biomarkers():
    data = request.json

    try:
        wbc = float(data['WBC'])
        neut = float(data['Neutrophils'])
        lymph = float(data['Lymphocytes'])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please enter valid biomarker values."}), 400

    if wbc < 0 or neut < 0 or lymph < 0:
        return jsonify({"error": "Values must be zero or greater."}), 400

    X = prepare_input(wbc, neut, lymph)

    pred = biomarker_model.predict(X)[0]
    probs = biomarker_model.predict_proba(X)[0]

    label = le.inverse_transform([pred])[0]
    class_probabilities = [
        {
            "label": class_name,
            "probability": float(prob)
        }
        for class_name, prob in zip(le.classes_, probs)
    ]

    # Logging
    with open("logs.txt", "a") as f:
        f.write(f"{datetime.now()} | WBC:{wbc}, N:{neut}, L:{lymph} -> {label}\n")

    return jsonify({
        "prediction": label,
        "probabilities": class_probabilities
    })

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if cnn_model is None:
        return jsonify({
            "error": "TensorFlow is not installed, so the CNN image classifier is unavailable."
        }), 500

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "Please choose an image file."}), 400

    try:
        image = Image.open(file.stream).convert("RGB").resize(IMAGE_SIZE)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception:
        return jsonify({"error": "The uploaded file could not be read as an image."}), 400

    probs = cnn_model.predict(img_array, verbose=0)[0]
    pred_index = int(np.argmax(probs))
    prediction = CNN_CLASSES[pred_index]

    class_probabilities = [
        {
            "label": class_name,
            "probability": float(prob)
        }
        for class_name, prob in zip(CNN_CLASSES, probs)
    ]

    with open("logs.txt", "a") as f:
        f.write(f"{datetime.now()} | image:{file.filename} -> {prediction}\n")

    return jsonify({
        "prediction": prediction,
        "probabilities": class_probabilities
    })

if __name__ == '__main__':
    app.run(debug=True)
