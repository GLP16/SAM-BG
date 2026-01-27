import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
import argparse
import sys
import csv
import copy
import random
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset

sys.path.append(os.path.abspath('/T2007061/pgl/pgl_experiment/'))
from model.sam.networks.sam_multi_lora_128 import build_sam_vit_b_adapter_linknet_multi_lora

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                if torch.is_floating_point(tensor):
                    self.shadow[name] = tensor.detach().clone()

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for name, tensor in msd.items():
                if torch.is_floating_point(tensor) and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        with torch.no_grad():
            msd = model.state_dict()
            for name, tensor in msd.items():
                if torch.is_floating_point(tensor) and name in self.shadow:
                    self.backup[name] = tensor.detach().clone()
                    tensor.copy_(self.shadow[name])

    def restore(self, model):
        if not self.backup:
            return
        with torch.no_grad():
            msd = model.state_dict()
            for name, tensor in msd.items():
                if name in self.backup:
                    tensor.copy_(self.backup[name])
        self.backup = {}

    def get_ema_state_dict(self, model):
        ema_sd = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            for name, tensor in ema_sd.items():
                if torch.is_floating_point(tensor) and name in self.shadow:
                    tensor.copy_(self.shadow[name])
        return ema_sd

class Args:
    def __init__(self):
        self.name = 'b_adapter_sam_multi_lora32_sp24'
        self.SAM_pretrained_path = 'sam_vit_b_01ec64.pth'
        self.log_dir = ''
        self.weight_dir = ''
        self.record_dir = ''
        self.viz_dir = ''
        self.image_size = 256
        self.seed = 2333
        self.base_lr = 4e-4
        self.only_eval = False
        self.weight_path = None
        self.use_rd_branch = False
        self.world_size = 1
        self.local_rank = -1
        self.dist_on_itp = False
        self.dist_url = 'env://'
        self.device = 'cuda'
        self.num_workers = 8
        self.pin_mem = True
        self.batch_size = 16
        self.learning_rate = 0.0004
        self.num_epochs = 120
        self.patience = 3
        self.alpha = 1.0
        self.b_beta = 1.0
        self.lambda_seg = 0.7
        self.lambda_bnd = 0.3
        self.train_dir = ""
        self.val_dir = ""
        self.test_dir = ""
        self.crop_size = (256, 256)
        self.scheduler_t_max = 60
        self.scheduler_eta_min = 1e-8
        self.grad_clip = 20.0
        self.ema_decay = 0.999

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

args = Args()
set_seed(args.seed)
os.makedirs(args.record_dir, exist_ok=True)
os.makedirs(args.weight_dir, exist_ok=True)
os.makedirs(args.viz_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
device_type = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
DEVICE = torch.device(device_type)
args.device = device_type

def get_soft_boundary_target(mask, kernel_size=3, gaussian_ksize=5, gaussian_sigma=1):
    mask_np = (mask * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    boundary_hard = (dilated - eroded).astype(np.float32)
    boundary_soft = cv2.GaussianBlur(boundary_hard, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    max_val = boundary_soft.max()
    if max_val > 0:
        boundary_soft /= max_val
    return boundary_soft

class BuildingDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False, crop_size=(256, 256)):
        super(BuildingDataset, self).__init__()
        self.img_dir = os.path.join(root_dir, "img")
        self.label_dir = os.path.join(root_dir, "label")
        self.img_list = sorted(os.listdir(self.img_dir))
        self.label_list = sorted(os.listdir(self.label_dir))
        self.transform = transform
        self.is_test = is_test
        self.crop_size = crop_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        label_path = os.path.join(self.label_dir, self.label_list[idx])
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        original_size = torch.tensor(img.size, dtype=torch.long)
        if not self.is_test:
            angle = np.random.uniform(-180, 180)
            img = img.rotate(angle)
            label = label.rotate(angle)
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_size)
            img = transforms.functional.crop(img, i, j, h, w)
            label = transforms.functional.crop(label, i, j, h, w)
        else:
            img = transforms.functional.resize(img, self.crop_size)
            label = transforms.functional.resize(label, self.crop_size)
        label_np = np.array(label) / 255.0
        soft_boundary_target_np = get_soft_boundary_target(label_np)
        if self.transform:
            img = self.transform(img)
            label = torch.from_numpy(label_np).unsqueeze(0).float()
            soft_boundary_target = torch.from_numpy(soft_boundary_target_np).unsqueeze(0).float()
        else:
            img = transforms.ToTensor()(img)
            label = transforms.ToTensor()(label)
            soft_boundary_target = torch.from_numpy(soft_boundary_target_np).unsqueeze(0).float()
        return img, label, soft_boundary_target, original_size

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        inputs = torch.nan_to_num(inputs, nan=0.5, posinf=1.0, neginf=0.0)
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        targets = torch.clamp(targets, 0.0, 1.0)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        if self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss

criterion_l2 = nn.MSELoss()
criterion_focal = FocalLoss(alpha=0.25, gamma=2.0)

def combined_loss(pred, target):
    loss_l2 = criterion_l2(pred, target)
    return args.alpha * loss_l2

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

def calculate_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
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
    boundary_iou_avg = np.mean(batch_boundary_iou)
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = BuildingDataset(args.train_dir, transform=transform, crop_size=args.crop_size)
val_dataset = BuildingDataset(args.val_dir, transform=transform, is_test=True, crop_size=args.crop_size)
test_dataset = BuildingDataset(args.test_dir, transform=transform, is_test=True, crop_size=args.crop_size)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_mem)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem)

def multi_task_loss(outputs, seg_target, bnd_target):
    seg_pred, bnd_pred = outputs
    seg_pred = seg_pred.float()
    bnd_pred = bnd_pred.float()
    seg_pred = torch.nan_to_num(seg_pred, nan=0.5, posinf=1.0, neginf=0.0)
    bnd_pred = torch.nan_to_num(bnd_pred, nan=0.5, posinf=1.0, neginf=0.0)
    seg_pred = torch.clamp(seg_pred, min=1e-5, max=1 - 1e-5)
    bnd_pred = torch.clamp(bnd_pred, min=1e-5, max=1 - 1e-5)
    seg_target = torch.nan_to_num(seg_target, nan=0.0, posinf=1.0, neginf=0.0)
    seg_target = torch.clamp(seg_target, 0.0, 1.0)
    bnd_target_resized = F.interpolate(bnd_target, size=bnd_pred.shape[2:], mode='bilinear', align_corners=False)
    bnd_target_resized = torch.nan_to_num(bnd_target_resized, nan=0.0, posinf=1.0, neginf=0.0)
    bnd_target_resized = torch.clamp(bnd_target_resized, min=0, max=1)
    seg_loss = combined_loss(seg_pred, seg_target)
    bnd_loss = args.b_beta * criterion_focal(bnd_pred, bnd_target_resized)
    total_loss = args.lambda_seg * seg_loss + args.lambda_bnd * bnd_loss
    return total_loss

def train_epoch(model, dataloader, optimizer, device, scaler, ema=None):
    model.train()
    running_loss = 0.0
    skipped_batches = 0
    seen_samples = 0
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for i, (inputs, targets, bnd_targets, _) in enumerate(pbar):
        try:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bnd_targets = bnd_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            use_amp = (scaler is not None) and (device.type == 'cuda')
            if use_amp:
                amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
                with autocast(dtype=amp_dtype):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            loss = multi_task_loss(outputs, targets, bnd_targets)
            if not torch.isfinite(loss):
                raise ValueError("Detected invalid loss value")
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            running_loss += loss.item() * inputs.size(0)
            seen_samples += inputs.size(0)
            pbar.set_postfix({'Loss': loss.item()})
        except (torch.cuda.OutOfMemoryError, ValueError, RuntimeError) as e:
            skipped_batches += 1
            print(f"\nWarning: error at batch {i+1}: '{e}'. Skipping this batch.")
            torch.cuda.empty_cache()
            pbar.set_postfix({'Loss': 'Skipped'})
            continue
    if skipped_batches > 0:
        print(f"Skipped {skipped_batches} batches in this epoch.")
    epoch_loss = running_loss / seen_samples if seen_samples > 0 else 0.0
    return epoch_loss

def validate_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_seg_preds = []
    all_targets = []
    pbar = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs, targets, bnd_targets, _ in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bnd_targets = bnd_targets.to(device, non_blocking=True)
            if device.type == 'cuda':
                amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
                with autocast(dtype=amp_dtype):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            loss = multi_task_loss(outputs, targets, bnd_targets)
            if torch.isfinite(loss):
                running_loss += loss.item() * inputs.size(0)
            seg_pred, bnd_pred = outputs
            seg_pred = torch.clamp(seg_pred.float().cpu(), 0, 1)
            all_seg_preds.append(seg_pred)
            all_targets.append(targets.cpu())
            pbar.set_postfix({'Loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    all_seg_preds = torch.cat(all_seg_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    precision, recall, f1, iou, boundary_iou_val = calculate_metrics(all_seg_preds, all_targets)
    return epoch_loss, precision, recall, f1, iou, boundary_iou_val

def resize_to_original(pred, original_size):
    w, h = int(original_size[0]), int(original_size[1])
    pred_uint8 = (pred * 255).astype(np.uint8)
    pred_pil = Image.fromarray(pred_uint8)
    pred_resized = pred_pil.resize((w, h), Image.NEAREST)
    return np.array(pred_resized) / 255.0

def test_epoch(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_seg_preds = []
    all_targets = []
    all_bnd_preds = []
    all_fi = []
    pbar = tqdm(dataloader, desc='Testing', leave=False)
    with torch.no_grad():
        for inputs, targets, bnd_targets, _ in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bnd_targets = bnd_targets.to(device, non_blocking=True)
            if device.type == 'cuda':
                amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
                with autocast(dtype=amp_dtype):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            loss = multi_task_loss(outputs, targets, bnd_targets)
            if torch.isfinite(loss):
                running_loss += loss.item() * inputs.size(0)
            seg_pred, bnd_pred = outputs
            seg_pred_cpu = seg_pred.detach().float().cpu().numpy()
            for i in range(seg_pred_cpu.shape[0]):
                mask = seg_pred_cpu[i, 0]
                metrics = calculate_metrics_for_mask(mask)
                if metrics:
                    fi = np.mean([m['FI'] for m in metrics])
                else:
                    fi = 0
                all_fi.append(fi)
            all_seg_preds.append(seg_pred.float().cpu())
            all_bnd_preds.append(bnd_pred.float().cpu())
            all_targets.append(targets.cpu())
            pbar.set_postfix({'Loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    all_seg_preds = torch.cat(all_seg_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_bnd_preds = torch.cat(all_bnd_preds, dim=0)
    seg_precision, seg_recall, seg_f1, seg_iou, seg_boundary_iou = calculate_metrics(all_seg_preds, all_targets)
    avg_fi = np.mean(all_fi) if all_fi else 0
    print(f'\nSegmentation metrics: Precision: {seg_precision:.4f} | Recall: {seg_recall:.4f} | F1: {seg_f1:.4f} | IoU: {seg_iou:.4f} | Boundary IoU: {seg_boundary_iou:.4f}')
    return (epoch_loss, seg_precision, seg_recall, seg_f1, seg_iou, seg_boundary_iou,
            avg_fi, all_seg_preds, all_bnd_preds)

# Number of consecutive training runs requested
num_training_rounds = 3
# Collect metrics for all runs to print at the end
test_run_metrics = []

for run_idx in range(1, num_training_rounds + 1):
    print(f"\n{'='*30} Starting training run {run_idx}/{num_training_rounds} {'='*30}\n")
    if args.name == 'b_adapter_sam_multi_lora32_sp24':
        model, encoder_global_attn_indexes = build_sam_vit_b_adapter_linknet_multi_lora(
            args.SAM_pretrained_path,
            image_size=args.image_size)
    else:
        raise ValueError(f"Unknown model name: {args.name}")
    model.to(DEVICE)
    if torch.cuda.device_count() > 1 and args.device == 'cuda':
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    log_path = os.path.join(args.log_dir, f"{args.name}")
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, f'{args.name}.log'), 'w') as mylog:
        print("Model structure: %s" % str(model_without_ddp), file=mylog)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max, eta_min=args.scheduler_eta_min)
    scaler = GradScaler()
    ema = ModelEMA(model, decay=args.ema_decay)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_boundary_iou = 0.0
    best_f1 = 0.0
    epochs_no_improve = 0
    record_file = os.path.join(args.record_dir, 'training_metrics.csv')
    with open(record_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate', 'Precision', 'Recall', 'F1 Score',
                         'IoU', 'Boundary IoU'])
        start_time = time.time()
        for epoch in range(args.num_epochs):
            print(f'\n--- Epoch {epoch+1}/{args.num_epochs} ---')
            print('-' * 60)
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE, scaler, ema)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            ema.apply_shadow(model)
            val_loss, precision, recall, f1, iou, boundary_iou_val = validate_epoch(model, val_loader, DEVICE)
            print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | '
                  f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | '
                  f'IoU: {iou:.4f} | Boundary IoU: {boundary_iou_val:.4f}')
            writer.writerow([epoch+1, train_loss, val_loss, current_lr, precision, recall, f1, iou, boundary_iou_val])
            if np.isnan(train_loss) or np.isnan(val_loss):
                print("Detected NaN loss. Stopping training early.")
                ema.restore(model)
                break
            if f1 > best_f1:
                best_f1 = f1
                epochs_no_improve = 0
                print(f'Validation F1 improved to {best_f1:.4f}. Reset patience counter.')
            else:
                epochs_no_improve += 1
                print(f'Validation F1 did not improve. Patience counter: {epochs_no_improve}.')
            if epochs_no_improve >= args.patience:
                print(f'No improvement for {args.patience} consecutive epochs. Triggering early stopping.')
                ema.restore(model)
                break
            if boundary_iou_val > best_boundary_iou:
                best_boundary_iou = boundary_iou_val
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'New best validation boundary IoU: {best_boundary_iou:.4f}')
            ema.restore(model)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'\nTotal training time for run {run_idx}: {total_time:.2f} seconds')

    print(f"\n--- Evaluating the best model on the test set (run {run_idx}) ---")
    model.load_state_dict(best_model_wts)
    test_loss, test_precision, test_recall, test_f1, test_iou, test_boundary_iou, \
    avg_fi, seg_preds, bnd_preds = test_epoch(model, test_loader, DEVICE)
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F1: {test_f1:.4f} | Test IoU: {test_iou:.4f} | Test Boundary IoU: {test_boundary_iou:.4f}')
    print(f'Average FI: {avg_fi:.4f}')
    final_model_path = os.path.join(args.weight_dir, 'final_best_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved best model for run {run_idx} to: {final_model_path}')
    print(f"\nGenerating prediction maps for run {run_idx}...")
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, bnd_targets, original_sizes) in enumerate(tqdm(test_loader, desc="Generating visualizations")):
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + inputs.size(0)
            if start_idx >= len(seg_preds):
                continue
            seg_batch = seg_preds[start_idx:end_idx]
            for i in range(inputs.size(0)):
                global_idx = batch_idx * args.batch_size + i
                if global_idx >= len(test_dataset.img_list):
                    continue
                original_size_i = original_sizes[i]
                seg_img = seg_batch[i].float().numpy().squeeze()
                seg_img = np.clip(seg_img, 0, 1)
                seg_img_resized = resize_to_original(seg_img, original_size_i)
                seg_uint8 = (seg_img_resized * 255).astype(np.uint8)
                seg_pil = Image.fromarray(seg_uint8)
                file_name = test_dataset.img_list[global_idx]
                save_path = os.path.join(args.viz_dir, file_name)
                seg_pil.save(save_path)
    print(f'\nSaved prediction maps for run {run_idx} to: {args.viz_dir}')
    test_run_metrics.append({
        'run': run_idx,
        'loss': test_loss,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'iou': test_iou,
        'boundary_iou': test_boundary_iou,
        'avg_fi': avg_fi
    })

print("\n=== Consolidated test metrics across runs ===")
for metrics in test_run_metrics:
    print(f"Run {metrics['run']}: Test Loss: {metrics['loss']:.4f} | "
          f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
          f"F1: {metrics['f1']:.4f} | IoU: {metrics['iou']:.4f} | "
          f"Boundary IoU: {metrics['boundary_iou']:.4f} | Average FI: {metrics['avg_fi']:.4f}")
