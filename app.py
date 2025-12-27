import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "best_model_ham_weighted.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

# mismas transforms que usaste al entrenar
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# FLASK
# ==============================
app = Flask(__name__)
CORS(app)

# ==============================
# CARGAR RESNET18
# ==============================
def load_model():
    print("🔄 Cargando modelo ResNet18...")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print(f"✅ Modelo cargado desde {MODEL_PATH} en {DEVICE}")
    return model


model = load_model()


# ==============================
# FUNCIÓN DE PREDICCIÓN (1 imagen)
# ==============================
def predict_pil_image(img: Image.Image):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).item()

    label_int = 1 if prob >= 0.5 else 0
    return label_int, prob


# ==============================
# ENDPOINTS
# ==============================

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200


# =====================================================
# /predict  (TU FRONTEND LO USA – SE MANTIENE IGUAL)
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se encontró el archivo 'file' en la petición"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    try:
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"No se pudo leer la imagen: {str(e)}"}), 400

    label_int, prob = predict_pil_image(img)
    label_str = "Maligna" if label_int == 1 else "Benigna"

    return jsonify({
        "prediction": label_str,
        "probability_malignant": prob,
        "probability_benign": 1 - prob
    }), 200


# =====================================================
# /predict_batch  (MULTIPLES IMÁGENES)
# =====================================================
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if "files" not in request.files:
        return jsonify({"error": "Falta el campo 'files' con las imágenes"}), 400

    files = request.files.getlist("files")

    if len(files) == 0:
        return jsonify({"error": "No se enviaron imágenes"}), 400

    results = []
    errors = []

    for file in files:
        if file.filename == "":
            continue

        try:
            img = Image.open(file.stream)
            label_int, prob = predict_pil_image(img)
            label_str = "Maligna" if label_int == 1 else "Benigna"

            results.append({
                "filename": file.filename,
                "prediction": label_str,
                "probability_malignant": prob,
                "probability_benign": 1 - prob
            })
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify({
        "total_images": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }), 200


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("🚀 Servidor Flask corriendo en puerto 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
