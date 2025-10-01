import os
import shutil

# Thư mục chứa ảnh gốc
src_folder = r"C:\WoodAI\data\data_netro\Gao"

# Thư mục đích sau khi đổi tên
dst_folder = r"C:\WoodAI\data\BDLS_BDTN_renamed"

# Prefix mới
prefix = "Gao_"

# Tạo folder đích nếu chưa có
os.makedirs(dst_folder, exist_ok=True)

# Lấy danh sách file ảnh và sort để tên ổn định
files = sorted([f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))])

# Đổi tên theo số thứ tự
for idx, filename in enumerate(files, start=1):
    src_path = os.path.join(src_folder, filename)
    
    # Lấy phần mở rộng (.jpg, .png, ...)
    _, ext = os.path.splitext(filename)
    
    # Tạo tên mới theo số thứ tự
    new_name = f"{prefix}{idx}{ext}"
    dst_path = os.path.join(dst_folder, new_name)
    
    # Copy sang folder mới
    shutil.copy2(src_path, dst_path)

print(f"Đã đổi tên {len(files)} ảnh theo số thứ tự và lưu vào:", dst_folder)
