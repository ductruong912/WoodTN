import os
import cv2
from tqdm import tqdm

SOURCE_DIR = r"C:\WoodAI\data\data_split\test"
DEST_DIR = r"C:\WoodAI\data\data_split\test_resize_crop_384"
TARGET_SIZE = 384

def resize_and_center_crop(img, target_size= TARGET_SIZE):
    h, w = img.shape[:2]
    short_edge = min(h, w)

    # Tính scale sao cho cạnh ngắn = target_size
    scale = target_size / short_edge
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize giữ tỉ lệ
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop target_size x target_size
    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    cropped = resized[top:top + target_size, left:left + target_size]
    return cropped

def resize_and_save_all(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    class_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in class_folders:
        src_class_path = os.path.join(source_dir, class_name)
        dst_class_path = os.path.join(dest_dir, class_name)
        os.makedirs(dst_class_path, exist_ok=True)

        image_files = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\nĐang xử lý class: {class_name} ({len(image_files)} ảnh)")
        for filename in tqdm(image_files, desc="Resize", leave=False):
            src_path = os.path.join(src_class_path, filename)
            dst_path = os.path.join(dst_class_path, filename)

            img = cv2.imread(src_path)
            if img is None:
                print(f"Không đọc được ảnh: {filename}")
                continue

            processed_img = resize_and_center_crop(img, target_size=TARGET_SIZE)
            cv2.imwrite(dst_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\nĐã resize + crop xong. Ảnh được lưu vào: {dest_dir}")

if __name__ == "__main__":
    resize_and_save_all(SOURCE_DIR, DEST_DIR)
