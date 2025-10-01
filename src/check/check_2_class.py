import os
import imagehash
from PIL import Image
from glob import glob
from tqdm import tqdm

folder_paths = {
    "LS": r"C:\WoodAI\data\data_khongtrung\Bach dan Lang Son",
    "TN": r"C:\WoodAI\data\data_khongtrung\Bach dan Thai Nguyen"
}

# Lưu thông tin ảnh: path, class, hash
hash_dict = {}

print("Hashing toàn bộ ảnh...")
for class_name, folder in folder_paths.items():
    for img_path in tqdm(glob(os.path.join(folder, "*"))):
        img = Image.open(img_path).convert("RGB").resize((300, 300))
        h = imagehash.phash(img, hash_size=16)
        hash_dict[img_path] = (class_name, h)

# So sánh từng cặp ảnh
print("\nSo sánh ảnh giữa 2 class (dù trùng tên hay không)...")
checked = set()
found = False

items = list(hash_dict.items())

for i in range(len(items)):
    path1, (cls1, hash1) = items[i]
    for j in range(i+1, len(items)):
        path2, (cls2, hash2) = items[j]

        if cls1 != cls2:
            dist = hash1 - hash2
            if dist <= 10:
                print(f"{os.path.basename(path1)} ({cls1}) <-> {os.path.basename(path2)} ({cls2}) | hash_dist = {dist}")
                found = True

if not found:
    print("Không có ảnh trùng giữa 2 class.")

# import os
# from PIL import Image
# import imagehash

# # Thư mục chứa ảnh 1 class
# img_folder = r"C:\WoodAI\data\data_renamed"

# # File kết quả
# output_file = r"C:\WoodAI\data\similar_images.txt"

# # Lấy danh sách ảnh
# files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# # Tạo dict lưu hash
# hash_dict = {}

# # Ngưỡng cho ảnh gần giống (0 = giống tuyệt đối, càng nhỏ càng giống)
# SIMILARITY_THRESHOLD = 5

# # Mở file để ghi kết quả
# with open(output_file, "w", encoding="utf-8") as f:
#     for filename in files:
#         path = os.path.join(img_folder, filename)
#         try:
#             img = Image.open(path)
#             img_hash = imagehash.phash(img)  # perceptual hash
#         except Exception as e:
#             print("❌ Lỗi khi mở ảnh:", filename, e)
#             continue

#         # So sánh với các ảnh trước đó
#         for prev_name, prev_hash in hash_dict.items():
#             diff = img_hash - prev_hash
#             if diff <= SIMILARITY_THRESHOLD:
#                 line = f"Ảnh {filename} gần giống với {prev_name} (diff={diff})\n"
#                 print(line.strip())  # vẫn in ra màn hình
#                 f.write(line)        # ghi vào file

#         # Lưu hash
#         hash_dict[filename] = img_hash

# print(f"✅ Đã lưu danh sách ảnh gần giống vào: {output_file}")

