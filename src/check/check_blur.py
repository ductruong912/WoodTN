# Duyệt toàn bộ ảnh trong thư mục.
# Tính độ nét (blur score) với 1 ngưỡng duy nhất (THRESHOLD, mặc định 32).
# Chuyển ảnh rõ (score ≥ THRESHOLD) sang thư mục mới data_netro/<tên class> (copy ảnh).
# In danh sách ảnh mờ ra file blurry_images.txt

import cv2
import os
import shutil
import numpy as np

# ========== CONFIG ==========
# Đặt đường dẫn folder ảnh gốc ở đây
CLASS_FOLDER = "/Users/ductruong/Documents/GitHub/WoodTN/data/Bạch đàn Lạng Sơn"

BLUR_THRESHOLD = 20

# Thư mục để lưu ảnh rõ
OUTPUT_FOLDER = os.path.join(CLASS_FOLDER, "clear_images")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# =============================

def variance_of_laplacian(image):
    """Đo độ nét bằng phương pháp Laplacian Variance"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def check_and_copy_clear_images(folder_path, threshold=BLUR_THRESHOLD):
    total = 0
    kept = 0
    removed = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # bỏ qua folder clear_images
        if os.path.isdir(file_path) or "clear_images" in file_path:
            continue

        # đọc ảnh
        image = cv2.imread(file_path)
        if image is None:
            print(f"⚠️ Không đọc được ảnh: {filename}")
            continue

        total += 1

        # chuyển grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # tính sharpness
        fm = variance_of_laplacian(gray)

        if fm >= threshold:
            kept += 1
            shutil.copy(file_path, os.path.join(OUTPUT_FOLDER, filename))
        else:
            removed += 1
            # Nếu muốn lưu ảnh mờ riêng thì mở comment dòng dưới
            shutil.copy(file_path, os.path.join(folder_path, "blur_images", filename))

        print(f"{filename}: sharpness={fm:.2f} --> {'OK' if fm >= threshold else 'BLUR'}")

    print("\n===== SUMMARY =====")
    print(f"Tổng ảnh: {total}")
    print(f"Giữ lại (rõ): {kept}")
    print(f"Lọc bỏ (mờ): {removed}")
    print(f"Ảnh rõ được lưu trong: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    print(f"Đang kiểm tra ảnh trong: {CLASS_FOLDER}")
    check_and_copy_clear_images(CLASS_FOLDER)


