import onnxruntime as ort
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# === CONFIG ===
onnx_path = r"C:\WoodAI\save_model\model_goai_v2_epcoh_10.onnx"
test_dir = r"C:\WoodAI\data\data_split\Test_real_resize"
batch_size = 16
class_names = [
 'Bach dan (Lang Son)',
 'Bach dan (Thai Nguyen)',
 'Gao',
 'Gioi',
 'Keo lai',
 'Keo tai tuong',
 'Lat hoa',
 'Lim xanh',
 'Mo (Thai Nguyen)',
 'Mo Tuyen Quang',
 'Que',
 'Tech',
 'Thong',
 'Xoan ta (Dai Tu)',
 'Xoan ta (Dinh Hoa)',
 'other',
 'test']
# === Transform giống lúc train ===
transform = transforms.Compose([ 
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load test data ===
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes

# === Load ONNX model ===
ort_session = ort.InferenceSession(onnx_path)

# === Run inference ===
y_true, y_pred = [], []

for images, labels in test_loader:
    for img, label in zip(images, labels):
        ort_inputs = {"input": img.unsqueeze(0).numpy()} 
        ort_outs = ort_session.run(None, ort_inputs)
        pred = np.argmax(ort_outs[0], axis=1)[0]

        y_true.append(label.item())
        y_pred.append(pred)


# === Kết quả ===
# print("\nClassification Report:")
# print(classification_report(y_true, y_pred, target_names=class_names))

# tạo confusion matrix
cm = confusion_matrix(y_true, y_pred)

# tạo DataFrame với tên hàng + cột
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

# in ra console tóm tắt
print("\n Confusion Matrix:")
print(df_cm)

# lưu ra file csv để đọc rõ ràng bằng Excel
df_cm.to_csv("test.csv", encoding="utf-8-sig")
print("\nConfusion Matrix đã được lưu ra file")
