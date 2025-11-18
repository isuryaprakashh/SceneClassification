from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from PIL import Image
import torch
from torchvision import transforms

from models1 import CNNRegularized, ResNetTransfer

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
CLASSES_PATH = ARTIFACTS_DIR / "classes.json"

app = Flask(__name__)


def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError("classes.json not found. Train the model first.")
    with CLASSES_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data["classes"]


def load_model(num_classes: int, image_size: int):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("best_model.pt not found. Train the model first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    chk_image_size = checkpoint.get("image_size", image_size)
    arch = checkpoint.get("arch", "cnn_regularized")
    
    # Load the correct model architecture
    if arch == "resnet_transfer":
        model = ResNetTransfer(num_classes=num_classes, input_size=chk_image_size)
    else:
        model = CNNRegularized(num_classes=num_classes, input_size=chk_image_size)
    
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, chk_image_size


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# Lazy loading - initialize on first use
CLASSES = None
MODEL = None
TRAIN_IMAGE_SIZE = 224
INFER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = None


def initialize_model():
    """Initialize model and classes on first use."""
    global CLASSES, MODEL, TRAIN_IMAGE_SIZE, TRANSFORM
    if CLASSES is None:
        CLASSES = load_classes()
    if MODEL is None:
        MODEL, TRAIN_IMAGE_SIZE = load_model(len(CLASSES), image_size=224)
        MODEL.to(INFER_DEVICE)
        TRANSFORM = build_transform(TRAIN_IMAGE_SIZE)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Scene Classification</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            margin: 0;
            padding: 1rem;
            background-color: #f4f4f4; 
            min-height: 100vh;
        }
        .container { 
            background: white; 
            padding: 1.5rem;
            border-radius: 8px; 
            max-width: 600px;
            margin: 0 auto;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h2 {
            margin-top: 0;
            font-size: 1.5rem;
        }
        input[type=file] { 
            margin: 1rem 0;
            width: 100%;
            padding: 0.5rem;
            font-size: 1rem;
        }
        button { 
            padding: 0.75rem 1.5rem;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
            width: 100%;
            margin-top: 0.5rem;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result { 
            margin-top: 1.5rem; 
            font-size: 1.1rem; 
            font-weight: bold;
            padding: 1rem;
            background-color: #e7f3ff;
            border-radius: 4px;
            word-wrap: break-word;
        }
        @media (max-width: 480px) {
            body {
                padding: 0.5rem;
            }
            .container {
                padding: 1rem;
            }
            h2 {
                font-size: 1.25rem;
            }
            .result {
                font-size: 1rem;
            }
        }
    </style>
 </head>
 <body>
    <div class="container">
        <h2>Upload an image</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Predict Scene</button>
        </form>
        {% if prediction %}
            <div class="result">Predicted Scene: {{ prediction }}</div>
        {% endif %}
    </div>
 </body>
</html>
"""


def predict_image(image: Image.Image):
    initialize_model()  # Ensure model is loaded
    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(INFER_DEVICE)
    with torch.no_grad():
        outputs = MODEL(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        score, pred_idx = torch.max(probabilities, 1)
    return CLASSES[pred_idx.item()], float(score.item())


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file:
            prediction = "No file provided"
        else:
            try:
                initialize_model()  # Ensure model is loaded before prediction
                image = Image.open(file.stream)
                label, confidence = predict_image(image)
                prediction = f"{label} ({confidence * 100:.2f}%)"
            except FileNotFoundError as exc:
                prediction = f"Model not found. Please train the model first: {exc}"
            except Exception as exc:  # pylint:disable=broad-except
                prediction = f"Error: {exc}"
    return render_template_string(HTML_TEMPLATE, prediction=prediction)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    try:
        initialize_model()  # Ensure model is loaded before prediction
        image = Image.open(file.stream)
        label, confidence = predict_image(image)
        return jsonify({"label": label, "confidence": confidence})
    except FileNotFoundError as exc:
        return jsonify({"error": f"Model not found. Please train the model first: {exc}"}), 404
    except Exception as exc:  # pylint:disable=broad-except
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

