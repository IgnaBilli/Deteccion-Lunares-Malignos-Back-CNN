import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
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
MODEL_PATH = "best_model_ham_weighted.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 15       # 🔼 Subimos a 15 para permitir mejor convergencia
BATCH_SIZE = 32
LR = 0.0003
IMG_SIZE = (224, 224)

print(f"🧠 Entrenando modelo HAM10000 (ajustado para detectar malignas)...")
print(f"📍 Dispositivo: {DEVICE}")


# ==============================
# DATASET PERSONALIZADO
# ==============================
class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        label = torch.tensor(self.df.iloc[idx]["binary_label"], dtype=torch.float32)
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

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["binary_label"], random_state=42)

    print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ==============================
# TRANSFORMACIONES
# ==============================
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_tf, test_tf


# ==============================
# MODELO
# ==============================
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)


# ==============================
# ENTRENAMIENTO
# ==============================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=100*correct/total)

        # === VALIDACIÓN ===
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"📈 Época {epoch+1}/{epochs} | Train Acc: {correct/total:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n🏁 Entrenamiento finalizado. Mejor Val Accuracy: {best_val_acc:.3f}")


# ==============================
# EVALUACIÓN
# ==============================
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluando"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n📊 REPORTE DE CLASIFICACIÓN - HAM10000 (Ajustado)")
    print(classification_report(y_true, y_pred, target_names=["Benigna", "Maligna"]))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benigna", "Maligna"],
                yticklabels=["Benigna", "Maligna"])
    plt.title("Matriz de Confusión - HAM10000 Ajustado")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()


# ==============================
# MAIN
# ==============================
def main():
    train_df, val_df, test_df = prepare_data()
    train_tf, test_tf = get_transforms()

    train_loader = DataLoader(SkinDataset(train_df, train_tf), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SkinDataset(val_df, test_tf), batch_size=BATCH_SIZE)
    test_loader = DataLoader(SkinDataset(test_df, test_tf), batch_size=BATCH_SIZE)

    model = create_model()

    # ⚖️ Ponderamos más las malignas (≈4 veces)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n🚀 Entrenando por {EPOCHS} épocas con pos_weight=4.0...\n")
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
