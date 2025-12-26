import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

# Config
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["benign", "malignant"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model():
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
    
    pred_idx = probs.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100
    
    results = {
        "prediction": pred_class,
        "confidence": f"{confidence:.2f}%",
        "probabilities": {name: f"{probs[i].item() * 100:.2f}%" for i, name in enumerate(CLASS_NAMES)}
    }
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    try:
        image = Image.open(file.stream).convert("RGB")
        results = predict(image)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Model loaded on {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    app.run(debug=True, host="0.0.0.0", port=1412)
