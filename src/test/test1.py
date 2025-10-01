# Duyệt toàn bộ ảnh trong thư mục.
# Tính độ nét (blur score) với 1 ngưỡng duy nhất (THRESHOLD, mặc định 32).
# Chuyển ảnh rõ (score ≥ THRESHOLD) sang thư mục mới data_netro/<tên class> (copy ảnh).
# In danh sách ảnh mờ ra file blurry_images.txt

import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

CLASS_FOLDER = r"C:\WoodAI\data\data_train\Gioi"
TARGET_ROOT = r"C:\WoodAI\data\data_blur"
THRESHOLD = 7
CROP_SIZE = 300

def get_blur_score(image, crop_size=CROP_SIZE):
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return 0.0
    x = (w - crop_size) // 2
    y = (h - crop_size) // 2
    crop = image[y:y+crop_size, x:x+crop_size]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            bytes_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
    except:
        return None

def check_and_copy_clear_images(folder_path, threshold=THRESHOLD):
    mo_anh = []
    net_anh = []

    class_name = os.path.basename(folder_path)

    # folder chứa ảnh nét
    target_clear = os.path.join(TARGET_ROOT, class_name, "clear")
    os.makedirs(target_clear, exist_ok=True)

    # folder chứa ảnh mờ
    target_blur = os.path.join(TARGET_ROOT, class_name, "blur")
    os.makedirs(target_blur, exist_ok=True)

    print(f"\nĐang kiểm tra ảnh trong: {folder_path}")
    for filename in tqdm(os.listdir(folder_path), desc="Kiểm tra", leave=False):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder_path, filename)
        if not os.path.isfile(path):
            continue

        image = imread_unicode(path)
        if image is None:
            print(f"Không đọc được ảnh: {filename}")
            continue

        score = get_blur_score(image)

        try:
            if score >= threshold:
                shutil.copy2(path, os.path.join(target_clear, filename))
                net_anh.append((filename, score))
            else:
                shutil.copy2(path, os.path.join(target_blur, filename))
                mo_anh.append((filename, score))
        except Exception as e:
            print(f"Lỗi copy {filename}: {e}")

    print(f"\nẢnh rõ (≥ {threshold}): {len(net_anh)} copy vào {target_clear}")
    print(f"Ảnh mờ (< {threshold}): {len(mo_anh)} copy vào {target_blur}")

    with open("blurry_images.txt", "a", encoding="utf-8") as f:
        f.write(f"\nClass: {class_name}")
        f.write(f"\nẢnh mờ (score < {threshold}): {len(mo_anh)} ảnh\n")
        for filename, score in sorted(mo_anh, key=lambda x: x[1]):
            f.write(f"{filename}\t{score:.2f}\n")

if __name__ == "__main__":
    if not os.path.exists(CLASS_FOLDER):
        print(f"Không tồn tại folder: {CLASS_FOLDER}")
    else:
        check_and_copy_clear_images(CLASS_FOLDER)
