import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# =========================
# 1. Load model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNetV2-S pretrained
model = models.efficientnet_v2_s(weights=None)
num_classes = 16
model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
model = model.to(device)
# Load weight đã train
model.load_state_dict(torch.load("C:\WoodAI\save_model\Go_v4_epoch_14.pth", map_location=device))
model.eval()

# =========================
# 2. Transform cho input
# =========================
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop((300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Danh sách class (thứ tự phải trùng lúc train)
class_names = [
 'Bach dan Lang Son',
 'Bach dan Thai Nguyen',
 'Gao',
 'Gioi',
 'Keo lai',
 'Keo tai tuong',
 'Lat hoa',
 'Lim xanh',
 'Mo Thai Nguyen',
 'Mo Tuyen Quang',
 'Que',
 'Tech',
 'Thong',
 'Xoan ta Dai Tu',
 'Xoan ta Dinh Hoa',
 'other']

# =========================
# 3. Hàm dự đoán 1 frame
# =========================
def predict(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    img = transform(transforms.ToPILImage()(img)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    return class_names[pred.item()], conf.item() * 100

# =========================
# 4. Mở camera
# =========================
cap = cv2.VideoCapture(0)  # 0 = webcam mặc định
import cv2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize nhỏ để hiển thị nhanh hơn
    display = cv2.resize(frame, (640, 480))

    # Predict
    label, conf = predict(display)

    # Vẽ kết quả
    text = f"{label} ({conf:.1f}%)"
    cv2.putText(display, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Wood Classification - EfficientNetV2S", display)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
