import os
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm

threshold = 6

def get_image_paths(folder):
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

def hash_images(image_paths, threshold=threshold):
    print("Đang tính hash và tìm ảnh trùng nhau")
    hashes = {}
    for path in tqdm(image_paths, desc="So sánh"):
        try:
            with Image.open(path) as img:
                h = imagehash.dhash(img)

            is_duplicate = False
            for existing_hash in hashes:
                distance = h - existing_hash
                if distance <= threshold:
                    hashes[existing_hash].append(path)
                    is_duplicate = True
                    break

            if not is_duplicate:
                hashes[h] = [path]
            
        except Exception as e:
            print(f"[Lỗi đọc ảnh] {path}: {e}")
    
    return list(hashes.values())

def get_image_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            return img.width * img.height
    except:
        return 0
    
def copy_non_duplicates(groups, source_folder, target_root):
    total_copied = 0
    relative_folder = os.path.relpath(source_folder, os.path.commonpath([source_folder]))
    target_folder = os.path.join(target_root, relative_folder)
    os.makedirs(target_folder, exist_ok=True)

    for i, group in enumerate(groups, 1):
        if len(group) > 1:
            resolutions = [(get_image_resolution(p), p) for p in group]
            resolutions.sort(reverse=True)
            keep_path = resolutions[0][1]
        else:
            keep_path = group[0]

        filename = os.path.basename(keep_path)
        dest_path = os.path.join(target_folder, filename)

        try:
            shutil.copy2(keep_path, dest_path)
            total_copied += 1
        except Exception as e:
            print(f"[Lỗi sao chép] {keep_path}: {e}")

    print(f"Đã sao chép {total_copied} ảnh không trùng lặp đến {target_folder}")
    return total_copied

if __name__ == "__main__":
    source_folder = "Bach dan Lang Son"
    target_folder = "Bach dan Lang Son - no duplicates"

    image_paths = get_image_paths(source_folder)
    print(f"Tổng số ảnh: {len(image_paths)}")

    groups = hash_images(image_paths, threshold=threshold)
    print(f"Tổng số nhóm ảnh duy nhất (sau loại trùng): {len(groups)}")

    # Ghi danh sách nhóm trùng nhau ra file
    output_path = "duplicate_groups.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, group in enumerate(groups, 1):
            if len(group) > 1:
                f.write(f"Nhóm {idx} ({len(group)} ảnh):\n")
                for path in group:
                    filename = os.path.basename(path)
                    f.write(f" {filename}\n")
                f.write("\n")
    print(f"Danh sách nhóm trùng nhau đã được ghi ra file: {output_path}")

    copy_non_duplicates(groups, source_folder, target_folder)