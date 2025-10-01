import os
import shutil
from PIL import Image
from pillow_heif import register_heif_opener

# ==== Đăng ký hỗ trợ đọc HEIC ====
register_heif_opener()

# ==== Cấu hình ==C:\WoodAI\data\train_sample_100_test==
src_root = r"C:\WoodAI\Go_that"         # ← thư mục chứa cả .jpg và .heic
dst_root = r"C:\WoodAI\Go_that_jpg"         # ← thư mục lưu tất cả ảnh .jpg đã chuẩn hoá

# ==== Tạo thư mục đích nếu chưa có ====
os.makedirs(dst_root, exist_ok=True)

count_converted = 0
count_copied = 0
count_failed = 0

print("Bắt đầu xử lý ảnh...")

for foldername, _, filenames in os.walk(src_root):
    for filename in filenames:
        ext = filename.lower().split('.')[-1]
        if ext not in ['jpg', 'jpeg', 'heic']:
            continue

        src_path = os.path.join(foldername, filename)

        # Tạo đường dẫn thư mục con trong thư mục đích
        relative_path = os.path.relpath(foldername, src_root)
        dst_folder = os.path.join(dst_root, relative_path)
        os.makedirs(dst_folder, exist_ok=True)

        if ext == "heic":
            # Đổi tên đuôi .heic → .jpg
            dst_filename = os.path.splitext(filename)[0] + ".jpg"
            dst_path = os.path.join(dst_folder, dst_filename)

            try:
                image = Image.open(src_path)
                # Giữ nguyên kích thước và chất lượng cao nhất
                image.save(dst_path, "JPEG", quality=100, subsampling=0)
                print(f"Chuyển HEIC → JPG: {src_path} → {dst_path}")
                count_converted += 1
            except Exception as e:
                print(f"Lỗi chuyển: {src_path}: {e}")
                count_failed += 1

        elif ext in ["jpg", "jpeg"]:
            dst_path = os.path.join(dst_folder, filename)

            try:
                # Copy nguyên ảnh .jpg gốc (không nén lại)
                shutil.copy2(src_path, dst_path)
                print(f"Copy JPG: {src_path} → {dst_path}")
                count_copied += 1
            except Exception as e:
                print(f"Lỗi copy: {src_path}: {e}")
                count_failed += 1

# ==== Tổng kết ====
print("\nTổng kết:")
print(f"HEIC → JPG: {count_converted}")
print(f"JPG copy:  {count_copied}")
print(f"Lỗi:       {count_failed}")
