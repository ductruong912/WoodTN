import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import timm
import os
# ======= Config =======
MODEL_PATH = r'C:\WoodAI\save_model\Go_v4_epoch_14.pth'
NUM_CLASSES = 16

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
    'other'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Load model =======
model = models.efficientnet_v2_s(weights=None)
model.classifier[1] = nn.Linear(in_features=1280, out_features=NUM_CLASSES)
# Load weight đã train
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ======= Transform giống lúc train =======
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]),
])

# ======= Webcam Prediction =======
cap = cv2.VideoCapture(0)  # 0: webcam mặc định
save_dir = r'C:\WoodAI\data_test_camera'
os.makedirs(save_dir, exist_ok=True)

print("🎥 Đang mở webcam... Bấm ESC để thoát. Bấm S để lưu ảnh.")

img_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không kết nối được webcam.")
        break

    # Convert sang RGB và PIL
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        label = class_names[pred.item()]

    # Hiển thị nhãn dự đoán
    cv2.putText(frame, f'Predict: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Live Prediction', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC để thoát
        break
    elif key == ord('s'):  # Nhấn S để lưu ảnh
        save_path = os.path.join(save_dir, f'img_{img_id}_{label}.jpg')
        cv2.imwrite(save_path, frame)
        print(f'💾 Đã lưu ảnh: {save_path}')
        img_id += 1

cap.release()
cv2.destroyAllWindows()
