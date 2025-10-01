import os
import cv2
import shutil
from tqdm import tqdm

def variance_of_laplacian(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return cv2.Laplacian(image, cv2.CV_64F).var()
    except:
        return -1

def select_sharpest_images_for_class(class_folder, output_folder, num_images):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        f for f in os.listdir(class_folder) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    scores = []
    print(f"\nĐang xử lý class: {os.path.basename(class_folder)}")

    for file in tqdm(image_files, desc="🔍 Đánh giá độ nét"):
        full_path = os.path.join(class_folder, file)
        score = variance_of_laplacian(full_path)
        if score != -1:
            scores.append((full_path, score))

    if len(scores) < num_images:
        print(f"Chỉ có {len(scores)} ảnh hợp lệ, ít hơn số lượng yêu cầu {num_images}")
        num_images = len(scores)

    # Sắp xếp theo độ nét giảm dần
    scores.sort(key=lambda x: x[1], reverse=True)

    # Chọn top N ảnh
    selected = scores[:num_images]

    # Copy ảnh ra thư mục đích
    for src_path, score in selected:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(output_folder, fname)
        shutil.copy(src_path, dst_path)

    print(f"Đã chọn {len(selected)} ảnh nét nhất lưu vào: {output_folder}")

# Đổi đường dẫn và số lượng tại đây
select_sharpest_images_for_class(
    class_folder=r"C:\WoodAI\data\data_blur\Keo lai\clear",
    output_folder=r"C:\WoodAI\data\data_train\Keo lai",
    num_images=1500
)
