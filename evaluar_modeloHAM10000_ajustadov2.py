import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================
# CONFIGURACIÓN
# ==============================
IMG_DIR = "HAM10000_images"
CSV_PATH = "HAM10000_metadata.csv"
MODEL_PATH = "best_model_ham_v3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

print("🧠 Evaluando modelo HAM10000 v3 (EfficientNet-B0)")
print(f"📍 Dispositivo detectado: {DEVICE}")

# ==============================
# DATASET
# ==============================
class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = torch.tensor(row["binary_label"], dtype=torch.float32)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label.unsqueeze(0)

# ==============================
# PREPARAR DATOS
# ==============================
def prepare_data():
    df = pd.read_csv(CSV_PATH)

    benignas = ["nv", "bkl", "df", "vasc"]
    df["binary_label"] = df["dx"].apply(lambda x: 0 if x in benignas else 1)
    df["path"] = df["image_id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))

    _, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["binary_label"],
        random_state=42
    )

    print(f"🧪 Test set: {len(test_df)} imágenes")
    return test_df

# ==============================
# TRANSFORMACIONES
# ==============================
def get_transforms():
    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return test_tf

# ==============================
# MODELO EfficientNet-B0 v3
# ==============================
def create_model():
    print("🔄 Cargando modelo EfficientNet-B0 v3...")

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    # cargar tus mejores pesos
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    print(f"✅ Modelo cargado desde {MODEL_PATH}")
    return model

# ==============================
# EVALUACIÓN
# ==============================
def evaluate_model(model, test_loader):
    y_true, y_pred = [], []

    use_amp = DEVICE.type == "cuda"

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluando"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(imgs)
            else:
                outputs = model(imgs)

            preds = (torch.sigmoid(outputs) > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n📊 REPORTE DE CLASIFICACIÓN - HAM10000 v3")
    print(classification_report(y_true, y_pred, target_names=["Benigna", "Maligna"]))

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benigna", "Maligna"],
        yticklabels=["Benigna", "Maligna"],
    )
    plt.title("Matriz de Confusión - HAM10000 v3")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

# ==============================
# MAIN
# ==============================
def main():
    test_df = prepare_data()
    test_tf = get_transforms()

    test_loader = DataLoader(
        SkinDataset(test_df, test_tf),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = create_model()
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
