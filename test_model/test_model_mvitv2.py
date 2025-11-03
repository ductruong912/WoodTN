import os
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cpu"
pth_path = r"C:\Users\xuant\Documents\GitHub\WoodTN\checkpoints\mvitv2_best_acc_0.9996.pth"
test_dir = r"C:\Users\xuant\Documents\GitHub\WoodTN\Go_that"
num_classes = 16
batch_size = 8   
img_size = 300
save_csv = "confusion_matrix_cpu.csv"

# =========================
# DATA TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# LOAD TEST DATA
# =========================
print("🔹 Loading dataset...")
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes
print(f"🧱 Test samples: {len(test_dataset)} | Classes: {len(class_names)}")

# =========================
# LOAD MODEL
# =========================
print("🔹 Loading MViTv2 model on CPU...")

model = timm.create_model("mvitv2_small", pretrained=False)

# ✅ Fix lỗi Sequential head
if isinstance(model.head, nn.Sequential):
    in_features = model.head[-1].in_features  # phần Linear cuối cùng
    model.head[-1] = nn.Linear(in_features, num_classes)
else:
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

# Load checkpoint lên CPU
ckpt = torch.load(pth_path, map_location=device)
model.load_state_dict(ckpt, strict=True)
model.to(device)
model.eval()

print(f"✅ Model loaded successfully from {pth_path}")
print(f"⚙️ Running inference on CPU only.")

# =========================
# EVALUATION
# =========================
y_true, y_pred = [], []
correct, total = 0, 0

print("🔹 Evaluating...")
with torch.no_grad():
    for imgs, labels in tqdm(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"\n✅ Accuracy on test set (CPU): {acc*100:.2f}%")

# =========================
# CONFUSION MATRIX + REPORT
# =========================
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
df_cm.to_csv(save_csv, encoding='utf-8-sig')
print(f"\n📄 Confusion matrix saved to: {save_csv}")

print("\n🔹 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix (CPU)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
