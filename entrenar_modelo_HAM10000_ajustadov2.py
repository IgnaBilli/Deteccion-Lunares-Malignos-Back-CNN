import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    recall_score,
)

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

EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4           # 🔻 LR más bajo para EfficientNet
IMG_SIZE = (224, 224)
PATIENCE = 8        # early stopping más relajado
SMOOTHING = 0.05    # label smoothing

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

print("🧠 Entrenando modelo HAM10000 v3 (GPU + AMP, OneCycle, smoothing)")
print(f"📍 Dispositivo detectado: {DEVICE}")


# ==============================
# DATASET PERSONALIZADO
# ==============================
class SkinDataset(Dataset):
    def __init__(self, df, transform_benign=None, transform_malign=None):
        self.df = df.reset_index(drop=True)
        self.tf_b = transform_benign
        self.tf_m = transform_malign

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = torch.tensor(row["binary_label"], dtype=torch.float32)

        image = Image.open(img_path).convert("RGB")

        if label.item() == 0 and self.tf_b:
            image = self.tf_b(image)
        elif label.item() == 1 and self.tf_m:
            image = self.tf_m(image)

        return image, label.unsqueeze(0)


# ==============================
# PREPARAR DATOS
# ==============================
def prepare_data():
    df = pd.read_csv(CSV_PATH)

    benignas = ["nv", "bkl", "df", "vasc"]
    df["binary_label"] = df["dx"].apply(lambda x: 0 if x in benignas else 1)
    df["path"] = df["image_id"].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))

    print("\n📊 Distribución binaria (0=Benigna, 1=Maligna):")
    print(df["binary_label"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["binary_label"], random_state=42
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["binary_label"], random_state=42
    )

    print(f"\n📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Train distribution:")
    print(train_df["binary_label"].value_counts())

    return train_df, val_df, test_df


# ==============================
# TRANSFORMACIONES
# ==============================
def get_transforms():
    augment_benign = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Fuerte pero no tan agresivo
    augment_malign = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return augment_benign, augment_malign, test_tf


# ==============================
# MODELO
# ==============================
def create_model():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    return model.to(DEVICE)


# ==============================
# ENTRENAMIENTO (GPU + AMP + OneCycle)
# ==============================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, patience, smoothing):

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_f2 = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}", leave=False)

        for imgs, labels in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # label smoothing: 0 -> eps/2, 1 -> 1 - eps/2
            labels_smooth = labels * (1.0 - smoothing) + 0.5 * smoothing

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels_smooth)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels_smooth)
                loss.backward()
                optimizer.step()

            scheduler.step()  # OneCycleLR: step por batch

            preds = (torch.sigmoid(outputs) > 0.5).float()

            running_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- VALIDACIÓN ----
        model.eval()
        y_true, y_pred = [], []
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)  # sin smoothing en val
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                preds = (torch.sigmoid(outputs) > 0.5).float()

                val_loss += loss.item() * labels.size(0)
                y_true.extend(labels.cpu().numpy().ravel())
                y_pred.extend(preds.cpu().numpy().ravel())

        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        val_loss /= len(y_true)
        val_f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=1)
        val_recall_malign = recall_score(y_true, y_pred, pos_label=1)

        print(
            f"Época {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} | "
            f"F2(malignas): {val_f2:.3f} | Recall malignas: {val_recall_malign:.3f}"
        )

        if val_f2 > best_f2:
            best_f2 = val_f2
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Mejor modelo guardado (F2={best_f2:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping activado (Mejor F2={best_f2:.3f})")
                break

    print(f"\n🏁 Entrenamiento finalizado. Mejor F2 validación: {best_f2:.3f}")


# ==============================
# EVALUACIÓN
# ==============================
def evaluate_model(model, test_loader):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    use_amp = DEVICE.type == "cuda"

    y_true, y_pred = [], []

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

            y_true.extend(labels.cpu().numpy().ravel())
            y_pred.extend(preds.cpu().numpy().ravel())

    print("\n📊 REPORTE DE CLASIFICACIÓN - TEST")
    print(classification_report(y_true, y_pred, target_names=["Benigna", "Maligna"]))

    f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=1)
    recall_malign = recall_score(y_true, y_pred, pos_label=1)
    print(f"📌 F2 (malignas): {f2:.3f}")
    print(f"📌 Recall malignas: {recall_malign:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
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
    plt.tight_layout()
    plt.show()


# ==============================
# MAIN
# ==============================
def main():
    train_df, val_df, test_df = prepare_data()
    augment_benign, augment_malign, test_tf = get_transforms()

    train_dataset = SkinDataset(train_df, augment_benign, augment_malign)
    val_dataset = SkinDataset(val_df, test_tf, test_tf)
    test_dataset = SkinDataset(test_df, test_tf, test_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = create_model()

    num_benign = (train_df["binary_label"] == 0).sum()
    num_malign = (train_df["binary_label"] == 1).sum()
    pos_weight_value = num_benign / num_malign
    pos_weight_value = min(pos_weight_value, 2.0)  # 🔻 limitamos
    print(f"\n⚖️ pos_weight calculado: {num_benign / num_malign:.3f} → usado: {pos_weight_value:.3f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=DEVICE)
    )

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # OneCycleLR: scheduler moderno, step por batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    print("\n🚀 Entrenando v3 (GPU + AMP + OneCycle + smoothing)...\n")
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        EPOCHS,
        PATIENCE,
        SMOOTHING,
    )

    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
