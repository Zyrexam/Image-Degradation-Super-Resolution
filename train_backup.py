import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import json
from datetime import datetime

from degrade import SRDataset
from model import DegradationAwareSR

# ==============================================================================
# CONFIG - FAST TRAINING (WINDOWS COMPATIBLE)
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 32 
NUM_WORKERS = 0 
PIN_MEMORY = False  
LR = 2e-4  
CONSIST_WEIGHT = 0.1  
CROP_SIZE = 64  
SAVE_PATH = "Newcheckpoints"
TRAIN_HR_DIR = "train_subset_1000_highres/HR"
MAX_TRAIN_IMAGES = 400  
USE_AMP = True  

os.makedirs(SAVE_PATH, exist_ok=True)

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ==============================================================================
# DATASET SETUP
# ==============================================================================

print(f"Device: {DEVICE}")
print("Loading dataset...")
train_dataset = SRDataset(
    TRAIN_HR_DIR,
    scale=4,
    degrade=True,
    crop_size=CROP_SIZE,
    return_lr_pair=True,
)

if MAX_TRAIN_IMAGES is not None and len(train_dataset) > MAX_TRAIN_IMAGES:
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=g)[:MAX_TRAIN_IMAGES].tolist()
    train_dataset = Subset(train_dataset, indices)
    print(f"Using fast subset: {len(train_dataset)} images")

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=0,
    pin_memory=False, 
)

# ==============================================================================
# MODEL & OPTIMIZATION
# ==============================================================================

model = DegradationAwareSR(scale=4, d_channels=8, feat_channels=32).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,} (smaller = faster)")

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda")) 

scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and DEVICE == 'cuda')) #type: ignore

# Aggressive scheduler (decay faster)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

# ==============================================================================
# METRICS
# ==============================================================================

def psnr(sr, hr):
    sr = torch.clamp(sr, 0, 1)
    hr = torch.clamp(hr, 0, 1)
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ==============================================================================
# TRAINING LOOP
# ==============================================================================

history = {"epoch": [], "loss": [], "loss_sr": [], "loss_consist": []}
best_epoch = 0

print(f"\n{'='*70}")
print(f"TRAINING CONFIG")
print(f"{'='*70}")
print(f"Batch Size:        {BATCH_SIZE}")
print(f"Crop Size:         {CROP_SIZE}")
print(f"Max Train Images:  {MAX_TRAIN_IMAGES}")
print(f"AMP Enabled:       {USE_AMP and DEVICE == 'cuda'}")
print(f"Learning Rate:     {LR}")
print(f"Consistency Weight: {CONSIST_WEIGHT}")
print(f"Total Params:      {total_params:,}")
print(f"Device:            {DEVICE}")
print(f"{'='*70}\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_loss_sr = 0
    train_loss_consist = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for lr1, lr2, hr_img in loop:
        lr1 = lr1.to(DEVICE)
        lr2 = lr2.to(DEVICE)
        hr_img = hr_img.to(DEVICE)
        
        # break

        # Forward
        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(USE_AMP and DEVICE == "cuda"),
        ):
            sr, d1, d2, _ = model(lr1, lr2)
            loss_sr = criterion(sr, hr_img)
            loss_consist = criterion(d1, d2)
            loss = loss_sr + CONSIST_WEIGHT * loss_consist

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate
        train_loss += loss.item()
        train_loss_sr += loss_sr.item()
        train_loss_consist += loss_consist.item()

        loop.set_postfix(
            loss=loss.item(),
            lr_sr=loss_sr.item(),
            lr_cs=loss_consist.item()
        )

    # Average losses
    n_batches = len(train_loader)
    train_loss /= n_batches
    train_loss_sr /= n_batches
    train_loss_consist /= n_batches

    # Logging
    history["epoch"].append(epoch + 1)
    history["loss"].append(train_loss)
    history["loss_sr"].append(train_loss_sr)
    history["loss_consist"].append(train_loss_consist)

    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Loss Total:      {train_loss:.6f}")
    print(f"  Loss SR:         {train_loss_sr:.6f}")
    print(f"  Loss Consistency:{train_loss_consist:.6f}")
    print(f"  LR:              {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'='*70}\n")

    # Save checkpoint every epoch
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"model_epoch_{epoch+1}.pth"))

    # Save best
    if epoch == 0 or train_loss < min(history["loss"][:-1]):
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model_best.pth"))
        print(f"✅ Best model saved at epoch {best_epoch}\n")

    scheduler.step()

print(f"\n{'='*70}")
print(f"✅ TRAINING COMPLETE")
print(f"Best epoch: {best_epoch}")
print(f"Total training time: ~{EPOCHS * 0.1} hours (approx)")
print(f"{'='*70}")
