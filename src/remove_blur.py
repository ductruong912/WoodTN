# kiem tra anh mo o cropcenter 300x300 va di chuyen anh mo sang class khac, bang tung class
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

CLASS_FOLDER = r"C:\WoodAI\data\data_jpg\Bach dan (Lang Son)"
BLUR_THRESHOLD = 31
CROP_SIZE = 300 
BLURRY_ROOT = r"C:\WoodAI\data\data_blurry"

def is_blurry_center(image, threshold=BLUR_THRESHOLD, crop_size=CROP_SIZE):
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return True, 0.0  

    # Lấy vùng giữa ảnh
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    center_crop = image[start_y:start_y + crop_size, start_x:start_x + crop_size]

    gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def process_folder(folder_path):
    total = 0
    moved = 0
    class_name = os.path.basename(folder_path)
    blurry_folder = os.path.join(BLURRY_ROOT, class_name)
    os.makedirs(blurry_folder, exist_ok=True)

    print(f"\n📂 Đang kiểm tra: {folder_path}")

    for filename in tqdm(os.listdir(folder_path), desc="🧹 Đang xử lý", leave=False):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src_path = os.path.join(folder_path, filename)

        if not os.path.isfile(src_path):
            continue

        image = cv2.imread(src_path)
        if image is None:
            print(f"⚠️ Không đọc được ảnh: {filename}")
            continue

        total += 1
        is_blur, score = is_blurry_center(image)

        if is_blur:
            try:
                dst_path = os.path.join(blurry_folder, filename)
                shutil.move(src_path, dst_path)
                moved += 1
                print(f"📦 Di chuyển ảnh mờ (center): {filename} | score = {score:.2f}")
            except Exception as e:
                print(f"❌ Lỗi khi di chuyển {filename}: {e}")

    print(f"\n✅ Tổng ảnh: {total} | Đã di chuyển: {moved} | Còn lại: {total - moved}")

if __name__ == "__main__":
    if not os.path.exists(CLASS_FOLDER):
        print(f"❌ Folder không tồn tại: {CLASS_FOLDER}")
    else:
        print(f"📁 Đang kiểm tra độ mờ (center crop) trong: {CLASS_FOLDER}")
        print(f"🎯 Ngưỡng độ nét: {BLUR_THRESHOLD} | Crop: {CROP_SIZE}x{CROP_SIZE}")
        process_folder(CLASS_FOLDER)