import torch
import cv2
import numpy as np
from PIL import Image
import os
from model import DegradationAwareSR

# ==============================================================================
# CONFIG
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints_cpu/model_best.pth"  # Update this path
TEST_IMAGE_PATH = "test2.jpg"  # Put your test image here
OUTPUT_DIR = "test_results"
SCALE = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# LOAD MODEL
# ==============================================================================

print(f"Device: {DEVICE}")
print(f"Loading model from: {CHECKPOINT_PATH}")

model = DegradationAwareSR(scale=SCALE, d_channels=8, feat_channels=64).to(DEVICE)

# Load checkpoint
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print("✅ Model loaded successfully")
else:
    print(f"⚠️ Checkpoint not found: {CHECKPOINT_PATH}")
    print("Using random weights (will produce bad output)")
    print("Train model first with: python train.py")

model.eval()

# ==============================================================================
# LOAD AND PREPARE IMAGE
# ==============================================================================

def load_image(image_path, scale=4):
    """Load image and prepare for model"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get original size
    h, w = img_rgb.shape[:2]
    
    # For testing, we'll use the full image (no cropping)
    # But we need LR size to be divisible by scale
    new_h = (h // scale) * scale
    new_w = (w // scale) * scale
    
    if new_h != h or new_w != w:
        img_rgb = cv2.resize(img_rgb, (new_w, new_h))
        print(f"Resized from {h}x{w} to {new_h}x{new_w} (divisible by {scale})")
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor, img_rgb, new_h, new_w

def save_image(tensor, save_path):
    """Save tensor as image"""
    # Convert to numpy
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Save
    Image.fromarray(img).save(save_path)
    print(f"Saved: {save_path}")

# ==============================================================================
# PATCH INFERENCE (For large images on CPU/GPU)
# ==============================================================================

def patch_inference(model, img_tensor, patch_size=128, overlap=32):
    """
    Run inference on image patches and stitch results
    Good for large images or memory constraints
    """
    _, _, h, w = img_tensor.shape
    device = next(model.parameters()).device
    
    # Calculate output size
    out_h = h * SCALE
    out_w = w * SCALE
    
    # Initialize output
    output = torch.zeros((1, 3, out_h, out_w), device=device)
    weight = torch.zeros((1, 1, out_h, out_w), device=device)
    
    # Step size (non-overlapping part)
    step = patch_size - overlap
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Get patch bounds
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            
            # Extract patch
            patch = img_tensor[:, :, y:y_end, x:x_end]
            
            # Run model
            with torch.no_grad():
                sr_patch, _ = model(patch.to(device), lr2=None)  # Single image mode
                sr_patch = sr_patch.cpu()
            
            # Calculate output bounds
            out_y_start = y * SCALE
            out_y_end = y_end * SCALE
            out_x_start = x * SCALE
            out_x_end = x_end * SCALE
            
            # Add to output (with overlap averaging)
            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += sr_patch
            weight[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += 1
    
    # Average overlapping regions
    output = output / weight
    
    return output

def full_inference(model, img_tensor):
    """Run inference on full image (faster for small images)"""
    with torch.no_grad():
        sr, _ = model(img_tensor.to(DEVICE), lr2=None)
    return sr.cpu()

# ==============================================================================
# CREATE LR IMAGE (Bicubic downsampling for comparison)
# ==============================================================================

def create_lr_image(img_rgb, scale=4):
    """Create low-resolution version using bicubic"""
    h, w = img_rgb.shape[:2]
    lr = cv2.resize(img_rgb, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return lr

# ==============================================================================
# MAIN TEST
# ==============================================================================

print("\n" + "="*70)
print("TESTING MODEL")
print("="*70)

# Load image
img_tensor, img_rgb, h, w = load_image(TEST_IMAGE_PATH, SCALE)
print(f"Input shape: {img_tensor.shape}")
print(f"Output will be: {h * SCALE} x {w * SCALE}")

# Create LR version (for comparison)
lr_img = create_lr_image(img_rgb, SCALE)
save_image(torch.from_numpy(lr_img).permute(2, 0, 1).unsqueeze(0), 
           os.path.join(OUTPUT_DIR, "01_input_lr.png"))

# Save original HR (ground truth if available)
save_image(img_tensor, os.path.join(OUTPUT_DIR, "02_original_hr.png"))

# Run inference
print("\nRunning inference...")

# Choose method based on image size
if h * w > 1024 * 1024:  # If image > 1MP
    print("Large image detected - using patch inference")
    sr_output = patch_inference(model, img_tensor, patch_size=128, overlap=32)
else:
    print("Using full image inference")
    sr_output = full_inference(model, img_tensor)

# Save SR output
save_image(sr_output, os.path.join(OUTPUT_DIR, "03_super_resolved.png"))

# Also save bicubic upscaled LR for comparison
bicubic_upscaled = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)
save_image(torch.from_numpy(bicubic_upscaled).permute(2, 0, 1).unsqueeze(0),
           os.path.join(OUTPUT_DIR, "04_bicubic_upscaled.png"))

print("\n" + "="*70)
print("✅ TEST COMPLETE")
print(f"Results saved in: {OUTPUT_DIR}")
print("="*70)
print("\nCompare these files:")
print("  - 01_input_lr.png (Low resolution input)")
print("  - 02_original_hr.png (Original high resolution)")
print("  - 03_super_resolved.png (Your model's output)")
print("  - 04_bicubic_upscaled.png (Simple bicubic upscale)")
print("="*70)