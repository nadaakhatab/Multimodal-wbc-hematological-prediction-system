import base64
import io

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

try:
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    cv2 = None
    torch = None
    nn = None
    F = None


class DoubleConv(nn.Module if nn is not None else object):
    def __init__(self, in_channels, out_channels):
        if nn is None:
            raise RuntimeError("PyTorch is not installed.")
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module if nn is not None else object):
    def __init__(self, n_channels=3, n_classes=1):
        if nn is None:
            raise RuntimeError("PyTorch is not installed.")
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv5 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))

        x = self.up1(x5)
        x = F.interpolate(x, size=x4.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x4], dim=1)
        x = self.conv5(x)

        x = self.up2(x)
        x = F.interpolate(x, size=x3.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.conv6(x)

        x = self.up3(x)
        x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv7(x)

        x = self.up4(x)
        x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv8(x)

        return self.outc(x)


IMG_SIZE = 256
device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else None
segmentation_model = None
segmentation_error = None

if torch is None or cv2 is None:
    segmentation_error = "PyTorch and OpenCV are required for the segmentation feature."
else:
    try:
        segmentation_model = UNet(n_channels=3, n_classes=1).to(device)
        segmentation_model.load_state_dict(torch.load("unet_wbc.pth", map_location=device))
        segmentation_model.eval()
    except Exception as exc:
        segmentation_error = f"Segmentation model failed to load: {exc}"
        segmentation_model = None


def _encode_image(array, extension, rgb_to_bgr=False):
    if rgb_to_bgr:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(extension, array)
    if not success:
        raise RuntimeError(f"Failed to encode {extension} image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def predict_segmentation_from_file(file_storage):
    if segmentation_model is None:
        return None, segmentation_error or "Segmentation model not loaded."

    try:
        raw_bytes = file_storage.read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image_np = np.array(image)

        img_resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img_tensor = torch.tensor(img_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = segmentation_model(img_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()

        binary_mask = (pred > 0.5).astype(np.uint8) * 255

        result = {
            "status": "success",
            "original": _encode_image(image_np, ".jpg", rgb_to_bgr=True),
            "probability_mask": _encode_image((pred * 255).astype(np.uint8), ".png"),
            "binary_mask": _encode_image(binary_mask, ".png"),
            "size": f"{IMG_SIZE}x{IMG_SIZE}"
        }
        return result, None
    except Exception as exc:
        return None, f"Processing failed: {exc}"


standalone_app = Flask(__name__)


@standalone_app.route("/")
def home():
    return render_template("segmentation.html")


@standalone_app.route("/segmentation")
def segmentation_page():
    return render_template("segmentation.html")


@standalone_app.route("/predict-segmentation", methods=["POST"])
def predict_segmentation_route():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "Please choose an image file."}), 400

    result, error = predict_segmentation_from_file(file)
    if error:
        return jsonify({"error": error}), 500
    return jsonify(result)


if __name__ == "__main__":
    standalone_app.run(debug=True, host="127.0.0.1", port=5000)
