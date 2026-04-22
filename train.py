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
from model.sam.networks.sam_multi_lora import build_sam_vit_b_adapter_linknet_multi_lora


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
        self.SAM_pretrained_path = 'SAM_PRETRAINED_PATH'
        self.image_size = '<IMAGE_SIZE>'
        self.seed = 2333
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
        self.batch_size = '<BATCH_SIZE>'
        self.learning_rate = '<learning_rate>'
        self.num_epochs = '<NUM_EPOCHS>'
        self.patience = '<patience>'
        self.alpha = 1.0
        self.b_beta = 1.0
        self.lambda_seg = 0.7
        self.lambda_bnd = 0.3
        self.train_dir = "TRAIN_DIR"
        self.val_dir = "VAL_DIR"
        self.test_dir = "TEST_DIR"
        self.viz_dir = "VIZ_DIR"
        self.crop_size = '(IMAGE_SIZE, IMAGE_SIZE)'
        self.scheduler_t_max = 60
        self.scheduler_eta_min = 1e-8
        self.grad_clip = 20.0
        self.ema_decay = 0.9992


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
    def __init__(self, root_dir, transform=None, is_test=False, crop_size=(128, 128)):
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
    return precision, recall, f1, iou


def compute_ci95(data):
    n = len(data)
    if n == 0:
        return None
    mean_val = np.mean(data)
    if n == 1:
        return mean_val, mean_val, mean_val, 0.0, 0.0
    std_val = np.std(data, ddof=1)
    t_critical_95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042
    }
    t_val = t_critical_95.get(n - 1, 1.96)
    half_width = t_val * std_val / np.sqrt(n)
    return mean_val, mean_val - half_width, mean_val + half_width, half_width, std_val


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
            seg_pred, _ = outputs
            seg_pred = torch.clamp(seg_pred.float().cpu(), 0, 1)
            all_seg_preds.append(seg_pred)
            all_targets.append(targets.cpu())
            pbar.set_postfix({'Loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    all_seg_preds = torch.cat(all_seg_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    precision, recall, f1, iou = calculate_metrics(all_seg_preds, all_targets)
    return epoch_loss, precision, recall, f1, iou


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
            seg_pred, _ = outputs
            all_seg_preds.append(seg_pred.float().cpu())
            all_targets.append(targets.cpu())
            pbar.set_postfix({'Loss': loss.item()})
    epoch_loss = running_loss / len(dataloader.dataset)
    all_seg_preds = torch.cat(all_seg_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    seg_precision, seg_recall, seg_f1, seg_iou = calculate_metrics(all_seg_preds, all_targets)
    print(f'\nSegmentation metrics: Precision: {seg_precision:.4f} | Recall: {seg_recall:.4f} | F1: {seg_f1:.4f} | IoU: {seg_iou:.4f}')
    return epoch_loss, seg_precision, seg_recall, seg_f1, seg_iou, all_seg_preds


def main():
    print("\n" + "=" * 80)
    print("Starting training on a single dataset")
    print(f"TRAIN_DIR: {args.train_dir}")
    print(f"VAL_DIR  : {args.val_dir}")
    print(f"TEST_DIR : {args.test_dir}")
    print(f"VIZ_DIR  : {args.viz_dir}")
    print(f"BATCH_SIZE: {args.batch_size}")
    print("=" * 80 + "\n")

    os.makedirs(args.viz_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BuildingDataset(args.train_dir, transform=transform, crop_size=args.crop_size)
    val_dataset = BuildingDataset(args.val_dir, transform=transform, is_test=True, crop_size=args.crop_size)
    test_dataset = BuildingDataset(args.test_dir, transform=transform, is_test=True, crop_size=args.crop_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_mem)

    num_runs = 5
    test_run_metrics = []

    for run_idx in range(num_runs):
        print(f"\n{'='*25} Starting training run {run_idx + 1}/{num_runs} {'='*25}\n")

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

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max, eta_min=args.scheduler_eta_min)
        scaler = GradScaler()
        ema = ModelEMA(model, decay=args.ema_decay)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_f1 = 0.0
        epochs_no_improve = 0

        start_time = time.time()
        for epoch in range(args.num_epochs):
            print(f'\n--- Epoch {epoch+1}/{args.num_epochs} ---')
            print('-' * 60)

            train_loss = train_epoch(model, train_loader, optimizer, DEVICE, scaler, ema)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            ema.apply_shadow(model)
            val_loss, precision, recall, f1, iou = validate_epoch(model, val_loader, DEVICE)

            print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | '
                  f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}')

            if np.isnan(train_loss) or np.isnan(val_loss):
                print("Detected NaN loss. Stopping training early.")
                ema.restore(model)
                break

            if f1 > best_f1:
                best_f1 = f1
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                print(f'Validation F1 improved to {best_f1:.4f}. Reset patience counter.')
            else:
                epochs_no_improve += 1
                print(f'Validation F1 did not improve. Patience counter: {epochs_no_improve}.')

            if epochs_no_improve >= args.patience:
                print(f'No improvement for {args.patience} consecutive epochs. Triggering early stopping.')
                ema.restore(model)
                break

            ema.restore(model)

        end_time = time.time()
        total_time = end_time - start_time
        print(f'\nTraining run {run_idx + 1}/{num_runs} completed, total time: {total_time:.2f} seconds')

        print(f"\n--- Evaluating the best model on the test set (run {run_idx+1}) ---")
        model.load_state_dict(best_model_wts)

        test_loss, test_precision, test_recall, test_f1, test_iou, seg_preds = test_epoch(model, test_loader, DEVICE)

        print(f'\nTest results for run {run_idx + 1}: Test Loss: {test_loss:.4f} | '
              f'Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | '
              f'F1: {test_f1:.4f} | IoU: {test_iou:.4f}')

        print(f"\nGenerating prediction maps for run {run_idx+1}...")
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
                    save_path = os.path.join(args.viz_dir, f"run_{run_idx+1}_{file_name}")
                    seg_pil.save(save_path)
        print(f'\nSaved prediction maps for run {run_idx+1} to: {args.viz_dir}')

        test_run_metrics.append({
            'run': run_idx + 1,
            'loss': test_loss,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'iou': test_iou
        })

    print("\n=== Consolidated test metrics across runs ===")
    for metrics in test_run_metrics:
        print(f"Run {metrics['run']}: Test Loss: {metrics['loss']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
              f"F1: {metrics['f1']:.4f} | IoU: {metrics['iou']:.4f}")

    print(f"\n{'='*25} All {num_runs} training runs have been completed {'='*25}\n")

    metrics_to_ci = {
        'Test Loss': [r['loss'] for r in test_run_metrics],
        'Test Precision': [r['precision'] for r in test_run_metrics],
        'Test Recall': [r['recall'] for r in test_run_metrics],
        'Test F1': [r['f1'] for r in test_run_metrics],
        'Test IoU': [r['iou'] for r in test_run_metrics]
    }

    print("\nFinal results (mean ± half-width of the 95% confidence interval):")
    for metric_name, values in metrics_to_ci.items():
        ci_result = compute_ci95(values)
        if ci_result is None:
            continue
        mean_val, lower, upper, half_width, std_val = ci_result
        print(f"{metric_name}: {mean_val:.6f} ± {half_width:.6f} (CI95: [{lower:.6f}, {upper:.6f}], N={len(values)})")


if __name__ == '__main__':
    main()
