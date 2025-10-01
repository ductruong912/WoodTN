import os
import cv2
import shutil
from tqdm import tqdm
from imgaug import augmenters as iaa

# Thư mục gốc chứa các class
DATA_DIR = r"C:\WoodAI\data_duplicate_blur"
OUTPUT_DIR = r"C:\WoodAI\data_train"
TARGET = 2000

# Augmentation pipeline
aug = iaa.Sequential([
    iaa.Fliplr(0.5),                 # lật ngang
    iaa.Flipud(0.2),                 # lật dọc
    iaa.Affine(rotate=(-20, 20)),    # xoay
    iaa.Multiply((0.8, 1.2)),        # thay đổi độ sáng
    iaa.LinearContrast((0.8, 1.2))   # thay đổi contrast
])

def resize_max(img, max_size=400):
    """Resize ảnh sao cho chiều dài lớn nhất = max_size, giữ tỉ lệ."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def sharpness_score(image):
    """Đo độ rõ nét bằng Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def balance_class(class_name, class_path, out_path):
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    os.makedirs(out_path, exist_ok=True)

    n = len(images)

    # Trường hợp dư ảnh → lọc theo sharpness
    if n > TARGET:
        print(f"{class_name}: giảm từ {n} → {TARGET}")
        sharpness = []
        for f in tqdm(images, desc=f"Tính sharpness {class_name}", unit="img"):
            p = os.path.join(class_path, f)
            img = cv2.imread(p)
            if img is None:
                continue
            img = resize_max(img, 400)   # resize trước khi tính sharpness
            sharpness.append((p, sharpness_score(img)))

        sharpness.sort(key=lambda x: x[1], reverse=True)  # giảm dần
        selected = [p for p, _ in sharpness[:TARGET]]

        for p in tqdm(selected, desc=f"Copy {class_name}", unit="img"):
            img = cv2.imread(p)
            if img is None: continue
            img = resize_max(img, 400)
            cv2.imwrite(os.path.join(out_path, os.path.basename(p)), img)

    # Trường hợp thiếu ảnh → copy + augment thêm
    elif n < TARGET:
        print(f"{class_name}: augment thêm {TARGET - n} ảnh (từ {n})")

        # Copy toàn bộ trước
        imgs = []
        for f in tqdm(images, desc=f"Copy {class_name}", unit="img"):
            p = os.path.join(class_path, f)
            img = cv2.imread(p)
            if img is None: continue
            img = resize_max(img, 400)
            imgs.append(img)
            cv2.imwrite(os.path.join(out_path, os.path.basename(p)), img)

        # Augment thêm cho đủ
        need = TARGET - len(imgs)
        for i in tqdm(range(need), desc=f"Augment {class_name}", unit="img"):
            base_img = imgs[i % len(imgs)]
            aug_img = aug(image=base_img)
            out_name = f"aug_{i}_{os.path.basename(images[i % len(images)])}"
            cv2.imwrite(os.path.join(out_path, out_name), aug_img)

    # Đúng bằng TARGET → copy nguyên
    else:
        print(f"{class_name}: giữ nguyên {n}")
        for f in tqdm(images, desc=f"Copy {class_name}", unit="img"):
            p = os.path.join(class_path, f)
            img = cv2.imread(p)
            if img is None: continue
            img = resize_max(img, 400)
            cv2.imwrite(os.path.join(out_path, os.path.basename(p)), img)

def main():
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for c in classes:
        class_clear = os.path.join(DATA_DIR, c, "clear")  # chỉ lấy folder clear
        out_path = os.path.join(OUTPUT_DIR, c)
        if os.path.exists(class_clear):
            balance_class(c, class_clear, out_path)

if __name__ == "__main__":
    main()
