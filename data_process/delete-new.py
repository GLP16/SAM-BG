import os
import cv2
import numpy as np

folder_A = ''
folder_B = ''

def imread_unicode(path, mode=cv2.IMREAD_GRAYSCALE):
    """
    Read an image file, supporting paths that contain Chinese characters.
    
    Args:
        path (str): File path (supports Unicode strings)
        mode (int): Read mode, default is grayscale mode (cv2.IMREAD_GRAYSCALE)
    
    Returns:
        img (ndarray): The loaded image array, or None if loading fails
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, mode)
        return img
    except Exception as e:
        print(f"Error: Failed to read {path}, reason: {e}")
        return None

print("Starting step 1: Deleting images with fewer than 100 pixels...")
for filename in os.listdir(folder_A):
    file_path_A = os.path.join(folder_A, filename)
    if os.path.isfile(file_path_A):
        img_A = imread_unicode(file_path_A, cv2.IMREAD_GRAYSCALE)
        if img_A is None:
            print(f"Warning: Cannot read image {file_path_A}, skipping")
            continue
        pixel_count = np.sum(img_A > 0)
        if pixel_count < 100:
            os.remove(file_path_A)
            print(f"Deleted {file_path_A} (pixel count: {pixel_count})")
            file_path_B = os.path.join(folder_B, filename)
            if os.path.isfile(file_path_B):
                os.remove(file_path_B)
                print(f"Deleted {file_path_B}")
            else:
                print(f"Warning: {file_path_B} does not exist, skipped")

print("Step 1 completed.\n")

print("Starting step 2: Deleting images smaller than 256x256...")
for filename in os.listdir(folder_A):
    file_path_A = os.path.join(folder_A, filename)
    file_path_B = os.path.join(folder_B, filename)
    if os.path.isfile(file_path_A) and os.path.isfile(file_path_B):
        img_A = imread_unicode(file_path_A, cv2.IMREAD_GRAYSCALE)
        if img_A is None:
            print(f"Warning: Cannot read {file_path_A}, skipping")
            continue
        height, width = img_A.shape
        if height < 256 or width < 256:
            os.remove(file_path_A)
            os.remove(file_path_B)
            print(f"Deleted {filename} (size: {width}x{height})")

print("Step 2 completed.\n")

print("Starting step 3: Calculating IOU and filtering images...")
for filename in os.listdir(folder_A):
    file_path_A = os.path.join(folder_A, filename)
    file_path_B = os.path.join(folder_B, filename)
    if os.path.isfile(file_path_A) and os.path.isfile(file_path_B):
        img_A = imread_unicode(file_path_A, cv2.IMREAD_GRAYSCALE)
        img_B = imread_unicode(file_path_B, cv2.IMREAD_GRAYSCALE)
        if img_A is None or img_B is None:
            print(f"Warning: Cannot read {file_path_A} or {file_path_B}, skipping")
            continue
        if img_A.shape != img_B.shape:
            print(f"Warning: {filename} has inconsistent dimensions in A and B, skipping")
            continue
        intersection = np.logical_and(img_A > 0, img_B > 0)
        union = np.logical_or(img_A > 0, img_B > 0)
        intersection_sum = np.sum(intersection)
        union_sum = np.sum(union)
        if union_sum == 0:
            iou = 0
        else:
            iou = intersection_sum / union_sum
        if iou > 0.98 or iou < 0.5:
            os.remove(file_path_A)
            os.remove(file_path_B)
            print(f"Deleted {filename} (IOU: {iou:.4f})")

print("Step 3 completed. Filtering task finished.")
