import os
from tqdm import tqdm
import cv2

ROOT_INPUT = r"C:\WoodAI\data\other_resize"
ROOT_OUTPUT = r"C:\WoodAI\data\data_crop"
CROP_SIZE = 224

os.makedirs(ROOT_OUTPUT, exist_ok=True)

def crop_center(image, size=CROP_SIZE):
    h, w = image.shape[:2]
    if h < size or w < size:
        return None
    x = (w - size) // 2
    y = (h - size) // 2
    return image[y:y+size, x:x+size]

for class_name in sorted(os.listdir(ROOT_INPUT)):
    input_class_path = os.path.join(ROOT_INPUT, class_name)
    output_class_path = os.path.join(ROOT_OUTPUT, class_name)

    if not os.path.isdir(input_class_path):
        continue

    os.makedirs(output_class_path, exist_ok=True)
    print(f"Đang xử lý class: {class_name}")

    for filename in tqdm(os.listdir(input_class_path), desc=f" {class_name}", leave=False):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        in_path = os.path.join(input_class_path, filename)
        out_path = os.path.join(output_class_path, filename)

        image = cv2.imread(in_path)
        if image is None:
            continue

        cropped = crop_center(image)
        if cropped is not None:
            cv2.imwrite(out_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
print("Đã crop toàn bộ ảnh và lưu vào:", ROOT_OUTPUT)
