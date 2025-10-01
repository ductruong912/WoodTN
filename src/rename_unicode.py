import os
import unicodedata

def remove_unicode(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text

def rename_all(root_folder):
    for folder_path, subdirs, files in os.walk(root_folder, topdown=False):
        # Đổi tên file trước
        for name in files:
            old_path = os.path.join(folder_path, name)
            new_name = remove_unicode(name)
            new_path = os.path.join(folder_path, new_name)
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                    print(f"📝 Đổi file: {old_path} ➜ {new_path}")
                except Exception as e:
                    print(f"❌ Lỗi đổi file {old_path}: {e}")

        # Đổi tên folder
        for name in subdirs:
            old_folder = os.path.join(folder_path, name)
            new_name = remove_unicode(name)
            new_folder = os.path.join(folder_path, new_name)
            if old_folder != new_folder:
                try:
                    os.rename(old_folder, new_folder)
                    print(f"📁 Đổi thư mục: {old_folder} ➜ {new_folder}")
                except Exception as e:
                    print(f"❌ Lỗi đổi folder {old_folder}: {e}")

    # Cuối cùng đổi thư mục gốc
    parent, name = os.path.split(root_folder)
    new_name = remove_unicode(name)
    new_path = os.path.join(parent, new_name)
    if root_folder != new_path:
        try:
            os.rename(root_folder, new_path)
            print(f"🏁 Đổi thư mục gốc: {root_folder} ➜ {new_path}")
        except Exception as e:
            print(f"❌ Lỗi đổi thư mục gốc: {e}")

if __name__ == "__main__":
    ROOT = r"C:\WoodAI\data\data_goc"  # Thư mục cha chứa các class
    rename_all(ROOT)
