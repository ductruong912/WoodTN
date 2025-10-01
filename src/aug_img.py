import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tqdm import tqdm

# === CẤU HÌNH ===
TRAIN_DIR = r"C:\WoodAI\data\data_split1\train"
TARGET_COUNT = 400  # Mỗi class sẽ có đúng 400 ảnh sau augment

# === Lấy danh sách ảnh thật và ảnh augment ===
def get_image_lists(folder):
    real_images = []
    aug_images = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            if f.startswith("aug_"):
                aug_images.append(f)
            else:
                real_images.append(f)
    return real_images, aug_images

# === Tạo ảnh augment cho 1 class (phân bổ đều) ===
def augment_class(class_name, class_path, target_count):
    real_images, aug_images = get_image_lists(class_path)
    current_total = len(real_images) + len(aug_images)

    if current_total >= target_count:
        print(f"✅ {class_name}: đã có {current_total} ảnh (gốc + augment), bỏ qua.")
        return

    need_more = target_count - current_total
    print(f"🔄 {class_name}: có {current_total}, cần thêm {need_more} ảnh")

    generated = 0
    aug_index = 0
    if aug_images:
        existing_numbers = [
            int(f.split("_")[1].split(".")[0]) for f in aug_images if f.startswith("aug_")
        ]
        aug_index = max(existing_numbers) + 1

    aug_per_image = (need_more + len(real_images) - 1) // len(real_images)

    with tqdm(total=need_more, desc=f"🔄 {class_name}", ncols=80) as pbar:
        for img_name in real_images:
            img_path = os.path.join(class_path, img_name)
            img = load_img(img_path)  # Giữ nguyên màu, không giảm saturation

            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            gen = augmentor.flow(x, batch_size=1)

            for _ in range(aug_per_image):
                if generated >= need_more:
                    break
                aug_filename = f"aug_{aug_index:06d}.jpg"
                save_path = os.path.join(class_path, aug_filename)
                batch = next(gen)
                array_to_img(batch[0]).save(save_path)
                aug_index += 1
                generated += 1
                pbar.update(1)

    total_now = len(os.listdir(class_path))
    print(f"✅ {class_name} xong: hiện có {total_now} ảnh")

# === Cấu hình augment nhẹ, tránh viền, giữ đặc trưng gỗ ===
augmentor = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.02,
    zoom_range=0.05,
    horizontal_flip=True,
    brightness_range=[0.95, 1.05],
    fill_mode="constant",
    cval=0,
)

# === CHẠY TRÊN TOÀN BỘ CLASS ===
for class_name in sorted(os.listdir(TRAIN_DIR)):
    class_path = os.path.join(TRAIN_DIR, class_name)
    if os.path.isdir(class_path):
        augment_class(class_name, class_path, TARGET_COUNT)

print("\n🎉 Augment hoàn tất – Tất cả class đã được nâng đều lên đủ 400 ảnh.")
