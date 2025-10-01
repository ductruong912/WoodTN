import cv2
import torch
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image

# ====== Config ======
MODEL_PATH = "C:\WoodAI\model_goai_v2_epcoh_3.onnx"
CLASS_NAMES = [
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
 'other']

# ====== Load ONNX model ======
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# ====== Preprocessing giống lúc train ======
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.03),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]),
])

# ====== Webcam loop ======
cap = cv2.VideoCapture(0) 
print("🎥 Đang mở webcam... Bấm ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Lỗi không đọc được frame từ webcam.")
        break

    # Chuyển ảnh sang RGB rồi transform
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).numpy()

    # Dự đoán với ONNX
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    pred = np.argmax(outputs[0])
    label = CLASS_NAMES[pred]

    # Hiển thị kết quả
    cv2.putText(frame, f'Predict: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ONNX Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
