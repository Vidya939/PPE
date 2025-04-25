import os
import shutil
import random

# Configurable split ratio
train_ratio = 0.8

# Paths
src_folder = "output/frames"
dst_images_train = "datasets/ppe/train/images"
dst_images_val = "datasets/ppe/valid/images"
dst_labels_train = "datasets/ppe/train/labels"
dst_labels_val = "datasets/ppe/valid/labels"

# Ensure all folders exist
for folder in [dst_images_train, dst_images_val, dst_labels_train, dst_labels_val]:
    os.makedirs(folder, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(src_folder) if f.endswith(".jpg")]

# Shuffle and split
random.shuffle(image_files)
split_idx = int(len(image_files) * train_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_files(image_list, dst_images, dst_labels):
    for img in image_list:
        img_src = os.path.join(src_folder, img)
        label_file = img.replace(".jpg", ".txt")
        label_src = os.path.join(src_folder, label_file)

        # Move image
        if os.path.exists(img_src):
            shutil.move(img_src, os.path.join(dst_images, img))
            print(f"✅ Moved image: {img} → {dst_images}")

        # Move label
        if os.path.exists(label_src):
            shutil.move(label_src, os.path.join(dst_labels, label_file))
            print(f"✅ Moved label: {label_file} → {dst_labels}")

# Move the files
move_files(train_files, dst_images_train, dst_labels_train)
move_files(val_files, dst_images_val, dst_labels_val)
