import os
import imagehash
from PIL import Image
from glob import glob
from tqdm import tqdm

folder_paths = {
    "LS": r"C:\Users\xuant\Documents\GitHub\WoodTN\data\Bach dan Lang Son",
    "TN": r"C:\Users\xuant\Documents\GitHub\WoodTN\data\Bach dan Thai Nguyen"
}

hash_dict = {}

print("Hashing toàn bộ ảnh...")
for class_name, folder in folder_paths.items():
    for img_path in tqdm(glob(os.path.join(folder, "*"))):
        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((300, 300))
            
            # Tính hash
            h = imagehash.dhash(img, hash_size=16)
            hash_dict[img_path] = (class_name, h)
        except Exception as e:
            print(f"Lỗi đọc ảnh {img_path}: {e}")

print("\nSo sánh ảnh giữa 2 class...")
items = list(hash_dict.items())
found = False

for i in range(len(items)):
    path1, (cls1, hash1) = items[i]
    for j in range(i+1, len(items)):
        path2, (cls2, hash2) = items[j]
        if cls1 != cls2:
            dist = hash1 - hash2
            if dist <= 5:
                print(f"{os.path.basename(path1)} ({cls1}) <-> "
                      f"{os.path.basename(path2)} ({cls2}) | hash_dist = {dist}")
                found = True

if not found:
    print("Không có ảnh trùng giữa 2 class.")
