import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch.nn as nn
# ==== CONFIGURATION ====
num_classes = 16
input_size = 300  # Phải giống lúc train
batch_size = 32
model_path = r'C:\WoodAI\save_model\Go_v4_epoch_14.pth'
test_dir = r'C:\WoodAI\Go_that'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== LOAD MODEL ====
model = models.efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
# Load weight đã train
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ==== TRANSFORM ====
test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])  # ImageNet stats
])

# ==== DATA ====
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes

# ==== INFERENCE ====
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ==== EVALUATION ====
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names)

print("\n📋 Classification Report:\n")
print(report)

# ==== CONFUSION MATRIX PLOT ====
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
