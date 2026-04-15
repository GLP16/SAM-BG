import os
import sys
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath('/T2007061/pgl/pgl_experiment/'))
from model.sam.networks.sam_multi_lora import build_sam_vit_b_adapter_linknet_multi_lora

INFER_DIR = "data/dir"
FINAL_MODEL_PATH = 'weight/dir'
VIZ_DIR = "viz/dir"

os.makedirs(VIZ_DIR, exist_ok=True)

IMAGE_SIZE = 256
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THRESH = 0.5

IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN.tolist(), std=IMAGE_STD.tolist())
])

class BuildingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "img")
        self.label_dir = os.path.join(root_dir, "label")
        self.img_list = sorted(os.listdir(self.img_dir))
        if os.path.exists(self.label_dir) and len(os.listdir(self.label_dir)) > 0:
            self.label_list = sorted(os.listdir(self.label_dir))
        else:
            self.label_list = None
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        label_path = os.path.join(self.label_dir, self.label_list[idx]) if self.label_list else None
        img_rgb = Image.open(img_path).convert('RGB')
        img_for_model = self.transform(img_rgb) if self.transform else transforms.ToTensor()(img_rgb)
        label_for_model = None
        if self.label_list is not None:
            label_pil = Image.open(label_path).convert('L')
            label_for_model = transforms.ToTensor()(label_pil)
        src_binary_for_iou = None
        if self.label_list is not None:
            src_img_pil_L = Image.open(img_path).convert('L')
            src_binary_for_iou = transforms.ToTensor()(src_img_pil_L)
            src_binary_for_iou = (src_binary_for_iou > 0.5).float()
        return img_for_model, label_for_model, self.img_list[idx], src_binary_for_iou

def get_boundary_mask(mask, kernel_size=3):
    mask_uint8 = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    boundary = (dilated - eroded).astype(np.float32)
    return boundary

def compute_boundary_iou(pred_mask, gt_mask):
    pred_boundary = get_boundary_mask(pred_mask)
    gt_boundary = get_boundary_mask(gt_mask)
    TP = np.sum((pred_boundary == 1) & (gt_boundary == 1))
    FP = np.sum((pred_boundary == 1) & (gt_boundary == 0))
    FN = np.sum((pred_boundary == 0) & (gt_boundary == 1))
    boundary_iou = TP / (TP + FP + FN + 1e-8)
    return boundary_iou

def to_probability(x: torch.Tensor) -> torch.Tensor:
    if x.min() < 0 or x.max() > 1:
        return torch.sigmoid(x)
    return x

def calculate_metrics(pred, target, threshold=THRESH):
    pred_prob = to_probability(pred)
    pred_bin = (pred_prob > threshold).float()
    target_bin = (target > 0.5).float()
    TP = (pred_bin * target_bin).sum().item()
    FP = (pred_bin * (1 - target_bin)).sum().item()
    FN = ((1 - pred_bin) * target_bin).sum().item()
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    intersection = TP
    union = TP + FP + FN
    iou = intersection / (union + 1e-6)
    pred_bin_np = pred_bin.cpu().numpy().astype(np.uint8)
    target_bin_np = target_bin.cpu().numpy().astype(np.uint8)
    batch_boundary_iou = []
    for i in range(pred_bin_np.shape[0]):
        boundary_iou_val = compute_boundary_iou(pred_bin_np[i, 0, :, :], target_bin_np[i, 0, :, :])
        batch_boundary_iou.append(boundary_iou_val)
    boundary_iou_avg = np.mean(batch_boundary_iou) if batch_boundary_iou else 0
    return precision, recall, f1, iou, boundary_iou_avg

def calculate_metrics_for_mask(mask, threshold=0.5):
    mask_bin = (mask > threshold).astype(np.uint8) * 255
    area_threshold = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    clean = np.zeros_like(mask_bin)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= area_threshold:
            clean[labels == lab] = 255
    mask_bin = clean
    contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    metrics = []
    if not contours or hierarchy is None:
        return []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            A = cv2.contourArea(contour)
            P = cv2.arcLength(contour, True)
            child_idx = hierarchy[0][i][2]
            while child_idx != -1:
                hole = contours[child_idx]
                A -= cv2.contourArea(hole)
                P += cv2.arcLength(hole, True)
                child_idx = hierarchy[0][child_idx][0]
            if P == 0:
                continue
            if A > 0 and P > 1:
                FI = 1 - (np.log(A) / (2 * np.log(P)))
            else:
                FI = 0
            metrics.append({'FI': FI})
    return metrics

infer_dataset = BuildingDataset(INFER_DIR, transform=transform)
infer_loader = DataLoader(infer_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

SAM_PRETRAINED_PATH = "/T2007061/pgl/pgl_experiment/model/sam/sam_vit_b_01ec64.pth"
model, _ = build_sam_vit_b_adapter_linknet_multi_lora(SAM_PRETRAINED_PATH, image_size=IMAGE_SIZE)
model.to(DEVICE)
if torch.cuda.device_count() > 1 and DEVICE.type == 'cuda':
    model = nn.DataParallel(model)

print("Loading final model weights...")
state_dict = torch.load(FINAL_MODEL_PATH, map_location=DEVICE)
is_model_parallel = isinstance(model, nn.DataParallel)
has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
if is_model_parallel and not has_module_prefix:
    new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
elif not is_model_parallel and has_module_prefix:
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
else:
    new_state_dict = state_dict
model.load_state_dict(new_state_dict)
model.eval()
print("Model loaded. Starting inference.")

all_sample_results = []

def calculate_iou_binary_tensors(mask1_bin, mask2_bin):
    intersection = (mask1_bin * mask2_bin).sum().item()
    union = TP = (mask1_bin + mask2_bin - (mask1_bin * mask2_bin)).sum().item()
    iou = intersection / (union + 1e-6)
    return iou

def compute_and_print_metrics_for_group(results_list, group_name):
    if not results_list:
        print(f"\n--- {group_name} ---")
        print("No data available, skipping metrics.")
        return
    group_seg_preds = torch.cat([res['seg_pred'].unsqueeze(0) for res in results_list], dim=0)
    group_targets = torch.cat([res['target'].unsqueeze(0) for res in results_list], dim=0)
    precision, recall, f1, iou, boundary_iou = calculate_metrics(group_seg_preds, group_targets)
    all_geometric_metrics = [metric for res in results_list for metric in res['geom_metrics']]
    print(f"\n--- {group_name} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Boundary IoU: {boundary_iou:.4f}")
    if all_geometric_metrics:
        avg_FI = np.mean([m['FI'] for m in all_geometric_metrics])
        print(f"Average FI: {avg_FI:.4f}")
    else:
        print("No valid objects found in this group.")

csv_file = open(os.path.join(VIZ_DIR, 'metrics.csv'), 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Image', 'src_tar_iou', 'FI'])

with torch.no_grad():
    for batch_idx, (inputs, targets, file_names, src_binary_for_iou_batch) in enumerate(tqdm(infer_loader, desc="Inferring")):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        src_binary_for_iou_batch = src_binary_for_iou_batch.to(DEVICE)
        outputs = model(inputs)
        seg_pred, bnd_pred = outputs
        for i in range(inputs.size(0)):
            current_file_name = file_names[i]
            seg_img_prob_for_metrics = to_probability(seg_pred[i]).cpu().numpy().squeeze()
            geometric_metrics_for_current_image = calculate_metrics_for_mask(seg_img_prob_for_metrics)
            current_src_tar_iou = 0.0
            if targets is not None:
                src_bin = src_binary_for_iou_batch[i].cpu().squeeze()
                tar_bin = (targets[i].cpu().squeeze() > 0.5).float()
                current_src_tar_iou = calculate_iou_binary_tensors(src_bin, tar_bin)
            all_sample_results.append({
                'file_name': current_file_name,
                'src_tar_iou': current_src_tar_iou,
                'seg_pred': seg_pred[i].cpu(),
                'target': targets[i].cpu(),
                'geom_metrics': geometric_metrics_for_current_image,
                'input_tensor': inputs[i].cpu()
            })
            for metric in geometric_metrics_for_current_image:
                csv_writer.writerow([
                    current_file_name,
                    current_src_tar_iou,
                    metric['FI']
                ])

csv_file.close()

print("\nStarting prediction map generation...")

for sample_result in tqdm(all_sample_results, desc="Saving predictions"):
    current_file_name = sample_result['file_name']
    seg_pred_tensor = sample_result['seg_pred']
    seg_img_prob = to_probability(seg_pred_tensor).numpy().squeeze()
    seg_img_bin = (seg_img_prob > THRESH).astype(np.uint8)
    seg_img_bin_viz = (seg_img_bin * 255).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(seg_img_bin_viz, cmap='gray')
    ax.axis('off')
    save_path = os.path.join(VIZ_DIR, current_file_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print(f"\nAll prediction maps saved to: {VIZ_DIR}")

if not all_sample_results:
    print("\nNo samples processed, metrics unavailable.")
else:
    compute_and_print_metrics_for_group(all_sample_results, "Overall results")
