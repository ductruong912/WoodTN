import os
import shutil

# Đường dẫn thư mục ảnh gốc
folder1 = r"C:\WoodAI\data\data_khongtrung\Xoan ta Dinh Hoa" 
folder2 = r"C:\WoodAI\data\data_khongtrung\BSXTDH" 
output_folder = r"C:\WoodAI\data\data_gop\Xoan ta Dinh Hoa" 

# Tạo folder output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Hàm copy ảnh từ folder này sang folder khác
def copy_images(src, dst):
    for i, filename in enumerate(os.listdir(src)):
        src_path = os.path.join(src, filename)
        if os.path.isfile(src_path):
            # Đặt tên lại cho tránh trùng lặp
            new_filename = f"{i}_{filename}"
            dst_path = os.path.join(dst, new_filename)
            shutil.copy(src_path, dst_path)

# Copy ảnh từ folder1 và folder2
copy_images(folder1, output_folder)
copy_images(folder2, output_folder)

print("✅ Đã gộp xong, tổng số ảnh:", len(os.listdir(output_folder)))
