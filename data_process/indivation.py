import os
import shutil
import random

def split_dataset(root_dir, train_dir, val_dir, train_ratio=0.9, seed=42):

    random.seed(seed)

    img_dir = os.path.join(root_dir, 'img')
    label_dir = os.path.join(root_dir, 'label')
    
    img_files = sorted(os.listdir(img_dir))
    label_files = sorted(os.listdir(label_dir))

    if len(img_files) != len(label_files):
        raise ValueError(f"Image file count ({len(img_files)}) and label file count ({len(label_files)}) do not match")
    

    for img, label in zip(img_files, label_files):
        img_name, img_ext = os.path.splitext(img)
        label_name, label_ext = os.path.splitext(label)
        if img_name != label_name:
            raise ValueError(f"Filenames do not match: image {img} and label {label}")
    
    indices = list(range(len(img_files)))
    random.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    os.makedirs(os.path.join(train_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'label'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'label'), exist_ok=True)

    print("Copying training set files...")
    for idx in train_indices:

        src_img = os.path.join(img_dir, img_files[idx])
        dst_img = os.path.join(train_dir, 'img', img_files[idx])
        shutil.copy(src_img, dst_img)

        src_label = os.path.join(label_dir, label_files[idx])
        dst_label = os.path.join(train_dir, 'label', label_files[idx])
        shutil.copy(src_label, dst_label)

    print("Copying validation set files...")
    for idx in val_indices:

        src_img = os.path.join(img_dir, img_files[idx])
        dst_img = os.path.join(val_dir, 'img', img_files[idx])
        shutil.copy(src_img, dst_img)

        src_label = os.path.join(label_dir, label_files[idx])
        dst_label = os.path.join(val_dir, 'label', label_files[idx])
        shutil.copy(src_label, dst_label)
    
    print(f"Split completed! Training set: {len(train_indices)} samples, Validation set: {len(val_indices)} samples")


if __name__ == "__main__":
    root_dir = '' 
    train_dir = ''   
    val_dir = ''          

    try:
        split_dataset(root_dir, train_dir, val_dir, train_ratio=0.9, seed=42)
    except Exception as e:
        print(f"An error occurred: {e}")
