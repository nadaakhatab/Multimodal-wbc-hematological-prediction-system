"""
WBC Classifier - Unified App
Run: python app.py
"""

import base64
import io
import math
import os
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, jsonify, render_template, request
from PIL import Image

# ── TensorFlow (CNN classifier) ──────────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

# ── PyTorch + OpenCV (UNet segmentation) ─────────────────────────────────────
try:
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch_ok = True
except ImportError:
    cv2 = torch = nn = F = None
    _torch_ok = False

# ═════════════════════════════════════════════════════════════════════════════
#  UNet definition
# ═════════════════════════════════════════════════════════════════════════════
if _torch_ok:
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.double_conv(x)

    class UNet(nn.Module):
        def __init__(self, n_channels=3, n_classes=1):
            super().__init__()
            self.inc   = DoubleConv(n_channels, 64)
            self.down1 = nn.MaxPool2d(2);  self.conv1 = DoubleConv(64, 128)
            self.down2 = nn.MaxPool2d(2);  self.conv2 = DoubleConv(128, 256)
            self.down3 = nn.MaxPool2d(2);  self.conv3 = DoubleConv(256, 512)
            self.down4 = nn.MaxPool2d(2);  self.conv4 = DoubleConv(512, 1024)
            self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.conv5 = DoubleConv(1024, 512)
            self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2);  self.conv6 = DoubleConv(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2);  self.conv7 = DoubleConv(256, 128)
            self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2);   self.conv8 = DoubleConv(128, 64)
            self.outc = nn.Conv2d(64, n_classes, 1)

        def _up(self, x, skip, conv_t, conv_block):
            x = conv_t(x)
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return conv_block(x)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.conv1(self.down1(x1))
            x3 = self.conv2(self.down2(x2))
            x4 = self.conv3(self.down3(x3))
            x5 = self.conv4(self.down4(x4))
            x  = self._up(x5, x4, self.up1, self.conv5)
            x  = self._up(x,  x3, self.up2, self.conv6)
            x  = self._up(x,  x2, self.up3, self.conv7)
            x  = self._up(x,  x1, self.up4, self.conv8)
            return self.outc(x)

# ═════════════════════════════════════════════════════════════════════════════
#  Load models
# ═════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)

# -- Biomarkers --
biomarker_model = joblib.load("model.pkl")
scaler          = joblib.load("scaler.pkl")
le              = joblib.load("label_encoder.pkl")

# -- CNN classifier --
CNN_CLASSES  = ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]
IMAGE_SIZE   = (128, 128)
cnn_model    = load_model("wbc_classification_model.keras") if load_model else None

# -- UNet segmentation --
SEG_SIZE          = 256
seg_model         = None
seg_error         = None

if not _torch_ok:
    seg_error = "PyTorch and OpenCV are required for the segmentation feature."
else:
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists("unet_wbc.pth"):
        try:
            seg_model = UNet(3, 1).to(_device)
            seg_model.load_state_dict(torch.load("unet_wbc.pth", map_location=_device))
            seg_model.eval()
        except Exception as e:
            seg_error = f"Segmentation model failed to load: {e}"
    else:
        seg_error = "unet_wbc.pth not found — segmentation unavailable."

# ═════════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ═════════════════════════════════════════════════════════════════════════════
def _log(msg: str):
    with open("logs.txt", "a") as f:
        f.write(f"{datetime.now()} | {msg}\n")

def _encode_cv(array, ext, rgb_to_bgr=False):
    if rgb_to_bgr:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, array)
    if not ok:
        raise RuntimeError(f"Failed to encode {ext}")
    return base64.b64encode(buf.tobytes()).decode()

def _prepare_biomarkers(wbc, neut, lymph):
    wbc_log = math.log1p(wbc)
    nlr     = neut / lymph if lymph != 0 else 0
    return scaler.transform(np.array([[wbc_log, neut, lymph, nlr]]))

# ═════════════════════════════════════════════════════════════════════════════
#  Routes — pages
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return render_template("index.html")

# ═════════════════════════════════════════════════════════════════════════════
#  Routes — API
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/predict-biomarkers", methods=["POST"])
def predict_biomarkers():
    data = request.json
    try:
        wbc  = float(data["WBC"])
        neut = float(data["Neutrophils"])
        lymph = float(data["Lymphocytes"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please enter valid biomarker values."}), 400

    if wbc < 0 or neut < 0 or lymph < 0:
        return jsonify({"error": "Values must be zero or greater."}), 400

    X    = _prepare_biomarkers(wbc, neut, lymph)
    pred = biomarker_model.predict(X)[0]
    probs = biomarker_model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]

    _log(f"biomarkers WBC:{wbc} N:{neut} L:{lymph} -> {label}")
    return jsonify({
        "prediction": label,
        "probabilities": [
            {"label": c, "probability": float(p)}
            for c, p in zip(le.classes_, probs)
        ]
    })


@app.route("/predict-image", methods=["POST"])
def predict_image():
    if cnn_model is None:
        return jsonify({"error": "TensorFlow not installed — CNN classifier unavailable."}), 500

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "Please choose an image file."}), 400

    try:
        img = Image.open(file.stream).convert("RGB").resize(IMAGE_SIZE)
        arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    probs      = cnn_model.predict(arr, verbose=0)[0]
    prediction = CNN_CLASSES[int(np.argmax(probs))]

    _log(f"image:{file.filename} -> {prediction}")
    return jsonify({
        "prediction": prediction,
        "probabilities": [
            {"label": c, "probability": float(p)}
            for c, p in zip(CNN_CLASSES, probs)
        ]
    })


@app.route("/predict-segmentation", methods=["POST"])
def predict_segmentation():
    if seg_model is None:
        return jsonify({"error": seg_error or "Segmentation model not loaded."}), 500

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "Please choose an image file."}), 400

    try:
        raw   = file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        img_np = np.array(image)

        resized = cv2.resize(img_np, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
        tensor  = torch.tensor(resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(_device)

        with torch.no_grad():
            pred = torch.sigmoid(seg_model(tensor)).squeeze().cpu().numpy()

        binary = (pred > 0.5).astype(np.uint8) * 255

        return jsonify({
            "status": "success",
            "original":          _encode_cv(img_np,                         ".jpg", rgb_to_bgr=True),
            "probability_mask":  _encode_cv((pred * 255).astype(np.uint8),  ".png"),
            "binary_mask":       _encode_cv(binary,                         ".png"),
            "size": f"{SEG_SIZE}x{SEG_SIZE}",
        })
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
