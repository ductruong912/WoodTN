import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ====== Config ======
MODEL_PATH = r'C:\WoodAI\save_model\Go_v1_epoch_7.pth'
NUM_CLASSES = 16
class_names = [
    'Bach dan (Lang Son)', 'Bach dan (Thai Nguyen)', 'Gioi', 'Gao',
    'Keo lai', 'Keo tai tuong', 'Lim xanh', 'Lat hoa',
    'Mo (Thai Nguyen)', 'Mo Tuyen Quang', 'Que', 'Thong',
    'Tech', 'Xoan ta (Dai Tu)', 'Xoan ta (Dinh Hoa)', 'other'
]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== Model & Transform ======
model = models.efficientnet_v2_s(pretrained=False)
model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== Grad-CAM setup ======
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# ====== Webcam ======
cap = cv2.VideoCapture(0)
print("🎥 Đang mở webcam... Bấm ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc webcam.")
        break

    # ===== Resize sao cho max 1 chiều = 400 =====
    h, w = frame.shape[:2]
    if max(h, w) != 400:
        scale = 400 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ===== Center crop 300x300 =====
    h, w = frame.shape[:2]
    x_start = max((w - 300) // 2, 0)
    y_start = max((h - 300) // 2, 0)
    cropped = frame[y_start:y_start + 300, x_start:x_start + 300]

    # Bỏ qua frame nếu chưa đủ lớn để crop
    if cropped.shape[0] < 300 or cropped.shape[1] < 300:
        continue

    # ===== Chuẩn bị input cho model =====
    rgb_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # ===== Predict + Grad-CAM =====
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        label = class_names[pred.item()]
        targets = [ClassifierOutputTarget(pred.item())]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    rgb_np = np.array(pil_img).astype(np.float32) / 255.0
    cam_img = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    # ===== Hiển thị kết quả =====
    cam_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
    cv2.putText(cam_bgr, f'Predict: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Grad-CAM", cam_bgr)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
