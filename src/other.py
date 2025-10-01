import os
import cv2
import hashlib
from icrawler.builtin import GoogleImageCrawler

SAVE_DIR = r"C:\WoodAI\data\other"
KEYWORDS = [
    "notebook", "charger", "earphones", "desk", "chair",
    "pillow", "blanket", "toothbrush", "comb", "scissors",
    "clock", "calculator", "flashlight", "water bottle", "sunglasses",
    "trash can", "broom", "mop", "kitchen knife", "tissue box"
]

NUM_IMAGES = 70
TARGET_SIZE = 300

os.makedirs(SAVE_DIR, exist_ok=True)
downloaded_hashes = set()

def file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def resize_image(path, size=TARGET_SIZE):
    img = cv2.imread(path)
    if img is None:
        return False
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img)
    return True

# Tải ảnh
for kw in KEYWORDS:
    print(f"🔍 Đang tải: {kw}")
    crawler = GoogleImageCrawler(storage={'root_dir': SAVE_DIR})
    crawler.crawl(
        keyword=kw,
        max_num=NUM_IMAGES,
        file_idx_offset=len(os.listdir(SAVE_DIR))  # tránh ghi đè
    )

# Xoá trùng + resize
print("🧹 Đang xử lý ảnh...")
all_files = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
for filename in all_files:
    path = os.path.join(SAVE_DIR, filename)
    h = file_hash(path)
    if h in downloaded_hashes:
        os.remove(path)
    else:
        downloaded_hashes.add(h)
        resize_image(path, TARGET_SIZE)

print(f"✅ Xong! Tổng ảnh sau xử lý: {len(downloaded_hashes)}")
