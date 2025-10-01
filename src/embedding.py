import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ==== CONFIG ====
model_path = r"C:\WoodAI\checkpoint_phase2.pth"   # đường dẫn model
test_dir = r"C:\Test quay anh thuc te 1"  # thư mục test
batch_size = 82
num_classes = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load model EfficientNetV2-S ====
model = models.efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
checkpoint = torch.load(model_path, map_location=device)
if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint
# Load weight đã train
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ==== Biến model thành feature extractor ====
feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
feature_extractor.eval()

# ==== DataLoader test ====
test_tf = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_data = datasets.ImageFolder(test_dir, transform=test_tf)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# ==== Lấy mapping idx -> class name ====
idx2class = {v: k for k, v in test_data.class_to_idx.items()}

# ==== Extract embeddings ====
all_features = []
all_classnames = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        feats = feature_extractor(inputs)   # [B, 1280, 1, 1]
        feats = feats.view(feats.size(0), -1)  # Flatten -> [B, 1280]
        all_features.append(feats.cpu())
        # convert labels -> class name
        all_classnames.extend([idx2class[l.item()] for l in labels])

# Gộp lại
all_features = torch.cat(all_features, dim=0).numpy()

print("Embedding shape:", all_features.shape)  # [N, 1280]
print("Labels:", len(all_classnames))

# ==== Lưu ra Excel chỉ với tên class ====
df = pd.DataFrame(all_features, columns=[f"feat_{i}" for i in range(all_features.shape[1])])
df["label_name"] = all_classnames
df.to_excel("embeddings.xlsx", index=False)
print("Đã lưu embeddings.xlsx (chỉ có label_name)")

# ==== Visualize bằng t-SNE ====
print("Đang chạy t-SNE (mất vài phút nếu data to)...")
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(all_features)

plt.figure(figsize=(8,6))

# ánh xạ tên class -> số để tô màu
unique_classes = sorted(list(set(all_classnames)))
class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
colors = [class_to_idx[c] for c in all_classnames]

scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, cmap="tab20", s=10)

# tạo legend với tên class
handles, _ = scatter.legend_elements(num=len(unique_classes))
plt.legend(handles, unique_classes, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title("t-SNE Visualization of Test Embeddings")
plt.show()

