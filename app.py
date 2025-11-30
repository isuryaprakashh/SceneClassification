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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
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
  <meta charset="UTF-8" />
  <title>SceneClassification – Train your model on real-world scenes</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <style>
    :root {
      --bg-gradient: radial-gradient(circle at top left, #ffe6f7 0, #f5f2ff 30%, #e9f5ff 60%, #e3f4ff 100%);
      --card-bg: rgba(255, 255, 255, 0.9);
      --border-soft: #e4e4ec;
      --text-main: #111827;
      --text-soft: #6b7280;
      --pill-dark: #111827;
      --pill-light: #ffffff;
      --pill-border: #d4d4dd;
      --radius-card: 22px;
      --shadow-soft: 0 18px 40px rgba(15, 23, 42, 0.18);
      --accent: #111827;
      --accent-soft: rgba(17, 24, 39, 0.06);
      --error-bg: #ffeaea;
      --error-border: #ff4d4d;
      --success-bg: #e7f1ff;
      --success-border: #2f6bed;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      min-height: 100vh;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
        "Segoe UI", sans-serif;
      background: var(--bg-gradient);
      color: var(--text-main);
    }

    .shell {
      width: 100%;
      margin: 0;
      padding: 32px 56px 40px;
    }

    /* HERO */
    .hero {
      text-align: center;
      margin-bottom: 24px;
      padding-inline: 40px;
    }

    .hero-title {
      font-size: clamp(2.3rem, 3vw + 1rem, 3.2rem);
      font-weight: 700;
      letter-spacing: -0.04em;
      line-height: 1.05;
      margin-bottom: 18px;
    }

    .hero-actions {
      display: inline-flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }

    .pill {
      border-radius: 999px;
      padding: 12px 22px;
      font-size: 0.95rem;
      font-weight: 500;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.15);
    }

    .pill-dark {
      background: var(--pill-dark);
      color: #ffffff;
    }

    /* ANALYZER CARD */
    .analyzer-wrapper {
      margin-top: 18px;
      margin-bottom: 18px;
    }

    .analyzer-card {
      background: var(--card-bg);
      border-radius: var(--radius-card);
      border: 1px solid rgba(255, 255, 255, 0.8);
      box-shadow: var(--shadow-soft);
      padding: 16px 18px 18px;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
      gap: 14px;
      align-items: stretch;
    }

    .analyzer-left-title {
      font-size: 0.9rem;
      font-weight: 600;
      margin-bottom: 4px;
    }

    .analyzer-left-sub {
      font-size: 0.8rem;
      color: var(--text-soft);
      margin-bottom: 10px;
    }

    .drop-zone {
      border-radius: 14px;
      border: 1.8px dashed var(--border-soft);
      background: #fafafa;
      padding: 1.1rem;
      text-align: center;
      cursor: pointer;
      transition: 0.18s ease;
    }

    .drop-zone:hover {
      background: #f3f4f6;
    }

    .drop-zone.drag-over {
      border-color: var(--accent);
      background: var(--accent-soft);
    }

    .drop-main-icon {
      font-size: 1.6rem;
      margin-bottom: 0.2rem;
    }

    .drop-title {
      font-size: 0.9rem;
      font-weight: 500;
      margin-bottom: 2px;
    }

    .drop-subtitle {
      font-size: 0.8rem;
      color: var(--text-soft);
    }

    .file-name {
      margin-top: 6px;
      font-size: 0.78rem;
      color: var(--text-soft);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    input[type=file] {
      display: none;
    }

    .analyzer-actions {
      margin-top: 10px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    /* FIXED preview box size */
    .preview-box {
      width: 100%;
      height: 220px;          /* fixed height */
      border-radius: 14px;
      border: 1px solid var(--border-soft);
      background: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
    }

    .preview-box::before {
      content: "Preview";
      position: absolute;
      top: 8px;
      left: 10px;
      font-size: 0.7rem;
      color: var(--text-soft);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .preview-box img {
      max-width: 100%;
      max-height: 100%;
      width: auto;           /* keep aspect ratio */
      height: auto;
      display: block;
    }

    .preview-empty {
      font-size: 0.85rem;
      color: var(--text-soft);
    }

    .result-pill {
      margin-top: 10px;
      padding: 0.65rem 0.8rem;
      border-radius: 999px;
      font-size: 0.86rem;
      display: none; /* shown by JS */
      align-items: center;
      gap: 6px;
    }

    .result-pill.success {
      background: var(--success-bg);
      border: 1px solid var(--success-border);
    }

    .result-pill.error {
      background: var(--error-bg);
      border: 1px solid var(--error-border);
    }

    .result-pill.loading {
      background: #e5e7eb;
      border: 1px solid #cbd5f5;
      color: #374151;
    }

    .result-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--text-soft);
    }

    .result-value {
      font-weight: 600;
    }

    .result-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #22c55e;
    }

    .result-dot.error {
      background: #ef4444;
    }

    .result-dot.loading {
      background: #9ca3af;
    }

    .submit-btn {
      border-radius: 999px;
      padding: 10px 18px;
      font-size: 0.9rem;
      font-weight: 500;
      border: none;
      cursor: pointer;
      background: var(--accent);
      color: #ffffff;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.24);
    }

    .submit-btn span:last-child {
      font-size: 0.9rem;
    }

    /* PROGRAM CARDS ROW */
    .programs-header {
      font-size: 0.9rem;
      color: var(--text-soft);
      margin-bottom: 8px;
      padding-inline: 2px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .programs-header strong {
      color: var(--text-main);
    }

    .programs-row {
      margin-top: 4px;
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: minmax(220px, 1fr);
      gap: 14px;
      overflow-x: auto;
      padding-bottom: 4px;
    }

    .program-card {
      background: var(--card-bg);
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-soft);
      border: 1px solid rgba(255, 255, 255, 0.6);
      overflow: hidden;
      min-width: 220px;
      display: flex;
      flex-direction: column;
    }

    .program-image {
      height: 150px;
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
    }

    .img-1 {
      background-image: url("https://images.unsplash.com/photo-1559494007-9f5847c49d94?q=80&w=687&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    }
    .img-2 {
      background-image: url("https://images.stockcake.com/public/f/7/c/f7ce4081-c879-4005-a861-a7fb636f38dd_large/sunset-over-river-stockcake.jpg");
    }
    .img-3 {
      background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSHrgOQ5sMNkUlc8xwJHjtIBXbbCzDfL7jQPQ&s");
    }
    .img-4 {
      background-image: url("https://burst.shopifycdn.com/photos/city-landscape-at-night.jpg?width=1000&format=pjpg&exif=0&iptc=0");
    }
    .img-5 {
      background-image: url("https://www.shutterstock.com/image-photo/vertical-background-image-wooden-school-600nw-2143046561.jpg");
    }

    .program-body {
      padding: 12px 14px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .program-level {
      font-size: 0.7rem;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(243, 244, 246, 0.9);
      align-self: flex-start;
      color: var(--text-soft);
    }

    .program-title {
      font-size: 0.92rem;
      font-weight: 600;
    }

    .program-meta {
      font-size: 0.78rem;
      color: var(--text-soft);
    }

    @media (max-width: 900px) {
      .shell {
        padding: 24px 16px 32px;
      }

      .hero {
        padding-inline: 18px;
      }

      .analyzer-card {
        grid-template-columns: minmax(0, 1fr);
      }
    }

    @media (max-width: 640px) {
      .hero-actions {
        flex-direction: column;
        width: 100%;
      }

      .pill {
        width: 100%;
        justify-content: center;
      }

      .programs-row {
        grid-auto-columns: 78%;
      }
    }
  </style>

  <script>
    document.addEventListener("DOMContentLoaded", () => {
      const dropZone = document.getElementById("drop-zone");
      const fileInput = document.getElementById("file-input");
      const preview = document.getElementById("preview");
      const placeholder = document.getElementById("preview-placeholder");
      const fileNameSpan = document.getElementById("file-name");
      const form = document.getElementById("predict-form");
      const resultPill = document.getElementById("result-pill");
      const resultLabel = document.getElementById("result-label");
      const resultValue = document.getElementById("result-value");
      const resultDot = document.getElementById("result-dot");

      function handleFile(file) {
        if (!file) {
          preview.src = "";
          preview.style.display = "none";
          placeholder.style.display = "block";
          fileNameSpan.textContent = "No image selected";
          return;
        }

        fileNameSpan.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
          preview.src = e.target.result;
          preview.style.display = "block";
          placeholder.style.display = "none";
        };
        reader.readAsDataURL(file);
      }

      if (fileInput) {
        fileInput.addEventListener("change", (e) => {
          const file = e.target.files[0];
          handleFile(file);
        });
      }

      if (dropZone) {
        dropZone.addEventListener("click", () => {
          fileInput.click();
        });

        dropZone.addEventListener("dragover", (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropZone.classList.add("drag-over");
        });

        dropZone.addEventListener("dragleave", (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropZone.classList.remove("drag-over");
        });

        dropZone.addEventListener("drop", (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropZone.classList.remove("drag-over");

          const file = e.dataTransfer.files[0];
          if (!file) return;

          const dt = new DataTransfer();
          dt.items.add(file);
          fileInput.files = dt.files;

          handleFile(file);
        });
      }

      // AJAX prediction so the page doesn't reload and preview stays
      form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
          resultPill.style.display = "inline-flex";
          resultPill.className = "result-pill error";
          resultDot.className = "result-dot error";
          resultLabel.textContent = "Status";
          resultValue.textContent = "Please select an image first.";
          return;
        }

        // show loading state
        resultPill.style.display = "inline-flex";
        resultPill.className = "result-pill loading";
        resultDot.className = "result-dot loading";
        resultLabel.textContent = "Status";
        resultValue.textContent = "Predicting scene...";

        const formData = new FormData();
        formData.append("image", file);

        try {
          const res = await fetch("/api/predict", {
            method: "POST",
            body: formData
          });

          const data = await res.json();

          if (!res.ok) {
            resultPill.className = "result-pill error";
            resultDot.className = "result-dot error";
            resultLabel.textContent = "Error";
            resultValue.textContent = data.error || "Something went wrong.";
          } else {
            const conf = (data.confidence * 100).toFixed(2);
            resultPill.className = "result-pill success";
            resultDot.className = "result-dot";
            resultLabel.textContent = "Prediction";
            resultValue.textContent = data.label + " (" + conf + "%)";
          }
        } catch (err) {
          resultPill.className = "result-pill error";
          resultDot.className = "result-dot error";
          resultLabel.textContent = "Error";
          resultValue.textContent = "Request failed.";
        }
      });
    });
  </script>
</head>

<body>
  <div class="shell">
    <!-- HERO -->
    <section class="hero">
      <h1 class="hero-title">
        Understand any scene.<br />
        Instantly. Anywhere.
      </h1>

      <div class="hero-actions">
        <button class="pill pill-dark" type="button" onclick="document.getElementById('file-input').click()">
          Upload an image
          <span>➜</span>
        </button>
      </div>
    </section>

    <!-- ANALYZER CARD -->
    <section class="analyzer-wrapper">
      <form id="predict-form" enctype="multipart/form-data">
        <div class="analyzer-card">
          <div>
            <div class="analyzer-left-title">Try your own image</div>
            <div class="analyzer-left-sub">
              Drag & drop or tap to upload a scene. Then click predict to see what the model sees.
            </div>

            <div id="drop-zone" class="drop-zone">
              <div class="drop-main-icon">📷</div>
              <div class="drop-title">Drag & Drop or Tap to Select</div>
              <div class="drop-subtitle">PNG, JPG up to ~10MB</div>
              <div class="file-name" id="file-name">No image selected</div>
            </div>

            <input id="file-input" name="image" type="file" accept="image/*" required>

            <div class="analyzer-actions">
              <button class="submit-btn" type="submit">
                <span>Predict scene</span>
                <span>⏵</span>
              </button>
            </div>
          </div>

          <div>
            <div class="preview-box">
              <img id="preview" style="display:none;" />
              <div id="preview-placeholder" class="preview-empty">
                Image preview will appear here after selection.
              </div>
            </div>

            <div id="result-pill" class="result-pill">
              <span id="result-dot" class="result-dot loading"></span>
              <span id="result-label" class="result-label"></span>
              <span id="result-value" class="result-value"></span>
            </div>
          </div>
        </div>
      </form>
    </section>

    <!-- SAMPLE SCENES -->
    <section>
      <div class="programs-header">
        <span><strong>Sample scenes</strong> · Try images similar to these for best results.</span>
        <span style="font-size:0.8rem; color: var(--text-soft);">Powered by your trained CNN / ResNet model.</span>
      </div>

      <div class="programs-row">
        <article class="program-card">
          <div class="program-image img-1"></div>
          <div class="program-body">
            <span class="program-level">Outdoor · Nature</span>
            <div class="program-title">Mountain lake landscape</div>
            <div class="program-meta">Great for natural outdoor scene detection.</div>
          </div>
        </article>

        <article class="program-card">
          <div class="program-image img-2"></div>
          <div class="program-body">
            <span class="program-level">Sunset · Water</span>
            <div class="program-title">Golden river sunset</div>
            <div class="program-meta">Tests lighting and horizon understanding.</div>
          </div>
        </article>

        <article class="program-card">
          <div class="program-image img-3"></div>
          <div class="program-body">
            <span class="program-level">Urban · Daytime</span>
            <div class="program-title">City streets & skyline</div>
            <div class="program-meta">Urban environment classification.</div>
          </div>
        </article>

        <article class="program-card">
          <div class="program-image img-4"></div>
          <div class="program-body">
            <span class="program-level">City · Night</span>
            <div class="program-title">Night city landscape</div>
            <div class="program-meta">Low-light and night-time scenes.</div>
          </div>
        </article>

        <article class="program-card">
          <div class="program-image img-5"></div>
          <div class="program-body">
            <span class="program-level">Indoor · Objects</span>
            <div class="program-title">Textured wooden desk</div>
            <div class="program-meta">Good for indoor / object-focused scenes.</div>
          </div>
        </article>
      </div>
    </section>
  </div>
</body>
</html>
"""


def predict_image(image: Image.Image):
    initialize_model()
    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(INFER_DEVICE)
    with torch.no_grad():
        outputs = MODEL(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        score, pred_idx = torch.max(probabilities, 1)
    return CLASSES[pred_idx.item()], float(score.item())


@app.route("/", methods=["GET"])
def index():
    # Only render the page; prediction happens via /api/predict
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    try:
        initialize_model()
        image = Image.open(file.stream)
        label, confidence = predict_image(image)
        return jsonify({"label": label, "confidence": confidence})
    except FileNotFoundError as exc:
        return jsonify(
            {"error": f"Model not found. Please train the model first: {exc}"}
        ), 404
    except Exception as exc:  # pylint:disable=broad-except
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)