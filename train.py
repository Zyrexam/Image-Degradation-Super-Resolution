import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import json
from datetime import datetime
import gc

from degrade import SRDataset
from model import DegradationAwareSR

# ==============================================================================
# CONFIG - CPU-OPTIMIZED (Ryzen 7500H + 16GB RAM)
# ==============================================================================


# ==============================================================================
# CONFIG - WINDOWS + CUDA (Ryzen 7500H + 16GB RAM)
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS_PHASE_A = 10
EPOCHS_PHASE_B = 35
BATCH_SIZE = 16
NUM_WORKERS = 0          # ← MUST BE 0 ON WINDOWS
PIN_MEMORY = False
LR_INITIAL = 5e-4
LR_FINAL = 1e-4
CONSIST_WEIGHT = 0.25
CROP_SIZE = 96
USE_AMP = True if DEVICE == "cuda" else False   # ← Enable for GPU
MAX_TRAIN_IMAGES_PHASE_A = 500
MAX_TRAIN_IMAGES_PHASE_B = 1500
SAVE_PATH = "checkpoints_cpu"
TRAIN_HR_DIR = "train_subset_1000_highres/HR"

os.makedirs(SAVE_PATH, exist_ok=True)

if DEVICE == "cpu":
    torch.set_num_threads(6)  # Use all Ryzen cores

print(f"Device: {DEVICE}")
print(f"CPU Threads: {torch.get_num_threads() if DEVICE == 'cpu' else 'N/A'}")

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
# SCHEDULER
# ==============================================================================

def get_scheduler(optimizer, total_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs, 
        eta_min=LR_FINAL
    )

# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_one_epoch(model, loader, optimizer, criterion, epoch, total_epochs, phase_name):
    model.train()
    train_loss = 0
    train_loss_sr = 0
    train_loss_consist = 0
    
    loop = tqdm(loader, desc=f"{phase_name} Epoch {epoch+1}/{total_epochs}")
    
    for lr1, lr2, hr_img in loop:
        lr1 = lr1.to(DEVICE)
        lr2 = lr2.to(DEVICE)
        hr_img = hr_img.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        sr, d1, d2, _ = model(lr1, lr2)
        loss_sr = criterion(sr, hr_img)
        loss_consist = criterion(d1, d2)
        loss = loss_sr + CONSIST_WEIGHT * loss_consist
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate
        train_loss += loss.item()
        train_loss_sr += loss_sr.item()
        train_loss_consist += loss_consist.item()
        
        loop.set_postfix(
            loss=loss.item(),
            sr=loss_sr.item(),
            cs=loss_consist.item()
        )
    
    n_batches = len(loader)
    return (train_loss / n_batches, 
            train_loss_sr / n_batches, 
            train_loss_consist / n_batches)

# ==============================================================================
# MAIN TRAINING
# ==============================================================================

print("\n" + "="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"Device:           {DEVICE}")
print(f"Phase A Epochs:   {EPOCHS_PHASE_A}")
print(f"Phase B Epochs:   {EPOCHS_PHASE_B}")
print(f"Total Epochs:     {EPOCHS_PHASE_A + EPOCHS_PHASE_B}")
print(f"Batch Size:       {BATCH_SIZE}")
print(f"Crop Size (HR):   {CROP_SIZE}")
print(f"Crop Size (LR):   {CROP_SIZE // 4}")
print(f"Consist Weight:   {CONSIST_WEIGHT}")
print(f"LR Initial:       {LR_INITIAL}")
print(f"LR Final:         {LR_FINAL}")
print(f"Workers:          {NUM_WORKERS}")
print("="*70)

# ==============================================================================
# PHASE A: Quick validation (10 epochs, 500 images)
# ==============================================================================

print("\n" + "="*70)
print("PHASE A: Quick Validation (10 epochs)")
print("="*70)

train_dataset_phaseA = SRDataset(
    TRAIN_HR_DIR,
    scale=4,
    degrade=True,
    crop_size=CROP_SIZE,
    return_lr_pair=True,
)

if len(train_dataset_phaseA) > MAX_TRAIN_IMAGES_PHASE_A:
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset_phaseA), generator=g)[:MAX_TRAIN_IMAGES_PHASE_A].tolist()
    train_dataset_phaseA = Subset(train_dataset_phaseA, indices)
    print(f"Phase A using {len(train_dataset_phaseA)} images")

train_loader_phaseA = DataLoader(
    train_dataset_phaseA,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# Initialize model
model = DegradationAwareSR(scale=4, d_channels=8, feat_channels=64).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR_INITIAL)
scheduler = get_scheduler(optimizer, EPOCHS_PHASE_A + EPOCHS_PHASE_B)
criterion = nn.L1Loss()

history_phaseA = {"epoch": [], "loss": [], "loss_sr": [], "loss_consist": []}

for epoch in range(EPOCHS_PHASE_A):
    loss, loss_sr, loss_cs = train_one_epoch(
        model, train_loader_phaseA, optimizer, criterion, 
        epoch, EPOCHS_PHASE_A, "Phase A"
    )
    
    history_phaseA["epoch"].append(epoch + 1)
    history_phaseA["loss"].append(loss)
    history_phaseA["loss_sr"].append(loss_sr)
    history_phaseA["loss_consist"].append(loss_cs)
    
    print(f"\nPhase A Epoch {epoch+1}/{EPOCHS_PHASE_A}:")
    print(f"  Loss:      {loss:.6f}")
    print(f"  SR Loss:   {loss_sr:.6f}")
    print(f"  Cons Loss: {loss_cs:.6f}")
    print(f"  LR:        {optimizer.param_groups[0]['lr']:.2e}\n")
    
    scheduler.step()
    
    # Save checkpoint every 2 epochs
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"phase_a_epoch_{epoch+1}.pth"))
    
    gc.collect()  # Force garbage collection

# Save Phase A checkpoint
torch.save(model.state_dict(), os.path.join(SAVE_PATH, "phase_a_complete.pth"))
print(f" Phase A complete! Saved to {SAVE_PATH}/phase_a_complete.pth")



 


# ==============================================================================
# PHASE B: Full training (35 more epochs, 1500 images)
# ==============================================================================

print("\n" + "="*70)
print("PHASE B: Full Training (35 more epochs)")
print("="*70)

train_dataset_phaseB = SRDataset(
    TRAIN_HR_DIR,
    scale=4,
    degrade=True,
    crop_size=CROP_SIZE,
    return_lr_pair=True,
)

if len(train_dataset_phaseB) > MAX_TRAIN_IMAGES_PHASE_B:
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset_phaseB), generator=g)[:MAX_TRAIN_IMAGES_PHASE_B].tolist()
    train_dataset_phaseB = Subset(train_dataset_phaseB, indices)
    print(f"Phase B using {len(train_dataset_phaseB)} images")

train_loader_phaseB = DataLoader(
    train_dataset_phaseB,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# Model already loaded from Phase A, continue training
history_phaseB = {"epoch": [], "loss": [], "loss_sr": [], "loss_consist": []}
best_loss = float('inf')
best_epoch = 0

# # Initialize model
# model = DegradationAwareSR(scale=4, d_channels=8, feat_channels=64).to(DEVICE)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Model parameters: {total_params:,}")

# optimizer = torch.optim.Adam(model.parameters(), lr=LR_INITIAL)
# scheduler = get_scheduler(optimizer, EPOCHS_PHASE_A + EPOCHS_PHASE_B)
# criterion = nn.L1Loss()

for epoch in range(EPOCHS_PHASE_B):
    current_epoch = EPOCHS_PHASE_A + epoch + 1
    
    loss, loss_sr, loss_cs = train_one_epoch(
        model, train_loader_phaseB, optimizer, criterion, 
        epoch, EPOCHS_PHASE_B, "Phase B"
    )
    
    history_phaseB["epoch"].append(current_epoch)
    history_phaseB["loss"].append(loss)
    history_phaseB["loss_sr"].append(loss_sr)
    history_phaseB["loss_consist"].append(loss_cs)
    
    print(f"\nPhase B Epoch {epoch+1}/{EPOCHS_PHASE_B} (Total: {current_epoch}):")
    print(f"  Loss:      {loss:.6f}")
    print(f"  SR Loss:   {loss_sr:.6f}")
    print(f"  Cons Loss: {loss_cs:.6f}")
    print(f"  LR:        {optimizer.param_groups[0]['lr']:.2e}\n")
    
    scheduler.step()
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"phase_b_epoch_{current_epoch}.pth"))
    
    # Save best model
    if loss < best_loss:
        best_loss = loss
        best_epoch = current_epoch
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model_best.pth"))
        print(f"  ✅ Best model saved (epoch {current_epoch})")
    
    gc.collect()

# ==============================================================================
# SAVE FINAL MODEL AND HISTORY
# ==============================================================================

torch.save(model.state_dict(), os.path.join(SAVE_PATH, "model_final.pth"))

# Save training history
history = {
    # "phase_a": history_phaseA,
    "phase_b": history_phaseB,
    "config": {
        "epochs_phase_a": EPOCHS_PHASE_A,
        "epochs_phase_b": EPOCHS_PHASE_B,
        "batch_size": BATCH_SIZE,
        "crop_size": CROP_SIZE,
        "lr_initial": LR_INITIAL,
        "lr_final": LR_FINAL,
        "consist_weight": CONSIST_WEIGHT,
        "device": DEVICE
    }
}

with open(os.path.join(SAVE_PATH, "training_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best epoch:       {best_epoch}")
print(f"Best loss:        {best_loss:.6f}")
print(f"Final loss:       {loss:.6f}")
print(f"Checkpoints saved: {SAVE_PATH}")
print("="*70)


