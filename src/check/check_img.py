# Kiem tra duoi anh
import os
from collections import Counter

folder_path = r"C:\WoodAI\data\data_crop" 
extensions = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        extensions.append(ext)

counter = Counter(extensions)

print("Các định dạng ảnh có trong thư mục:")
for ext, count in counter.items():
    print(f"{ext} : {count} file")

