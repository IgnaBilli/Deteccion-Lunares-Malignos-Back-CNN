import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "best_model_ham_v3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

# mismas normalizaciones usadas en el modelo v3
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__)
CORS(app)


# ==============================
# CARGAR MODELO EfficientNet-B0 v3
# ==============================
def load_model():
    print("🔄 Cargando modelo EfficientNet-B0 v3...")

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # reemplazar capa final (1000 → 1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    # cargar pesos entrenados
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print(f"✅ Modelo cargado desde {MODEL_PATH} en {DEVICE}")
    return model


model = load_model()  # se carga una sola vez al iniciar el servidor


# ==============================
# FUNCIÓN DE PREDICCIÓN
# ==============================
def predict_pil_image(img: Image.Image):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast("cuda" if DEVICE.type == "cuda" else "cpu"):
            output = model(tensor)
            prob = torch.sigmoid(output).item()

    label_int = 1 if prob >= 0.5 else 0
    return label_int, prob


# ==============================
# ENDPOINTS
# ==============================

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"}), 200


@app.route("/predict", methods=["POST"])
def predict2():
    """Recibe una imagen como file (multipart/form-data)."""

    if "file" not in request.files:
        return jsonify({"error": "Falta el archivo 'file' en la petición"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    # leer imagen
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


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Recibe múltiples imágenes y devuelve predicciones para todas."""

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
# MAIN SERVER
# ==============================
if __name__ == "__main__":
    print("🚀 Servidor Flask levantando en puerto 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
