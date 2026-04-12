import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from io import BytesIO
from PIL import Image

# ==============================================================================
# REALISTIC DEGRADATION (CPU-optimized for Ryzen)
# ==============================================================================

def _apply_motion_blur(img, rng):
    """Simple motion blur (5-15% chance, CPU-friendly)"""
    if rng.random() > 0.15:  # Only 15% chance (realistic)
        return img
    
    size = rng.choice([5, 7, 9])
    angle = rng.uniform(0, 180)
    
    # Create motion kernel
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size//2, :] = 1.0
    
    # FIX: Convert center to floats
    center = (float(size // 2), float(size // 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))
    kernel = kernel / np.sum(kernel)
    
    return cv2.filter2D(img, -1, kernel)

def _apply_gaussian_blur(img, rng):
    """Gaussian blur (always applied, variable strength)"""
    k = rng.choice([3, 5, 7])
    # sigma = rng.uniform(0.5, 2.5)
    sigma = rng.uniform(0.5, 2.0)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)

def _apply_jpeg_compression(img, rng):
    """Simulate JPEG compression artifacts (CPU-friendly)"""
    if rng.random() > 0.7:  # 70% chance (common in real world)
        return img
    
    quality = rng.randint(75, 95)
    
    # CRITICAL: Convert float32 to uint8 for PIL
    if img.dtype == np.float32:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    # Convert to PIL, compress, back to numpy
    img_pil = Image.fromarray(cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img_jpeg = np.array(Image.open(buffer))
    
    # Convert back to float32
    result = cv2.cvtColor(img_jpeg, cv2.COLOR_RGB2BGR).astype(np.float32)
    
    return result

def _apply_noise(img, rng):
    """Gaussian noise (light to medium)"""
    # sigma = rng.uniform(1.0, 5.0)  # Reduced range for CPU
    
    sigma = rng.uniform(0.5, 2.0)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255)

def _apply_downsample(img, scale, rng):
    """Downsample with INTER_AREA (best for CPU)"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

def apply_lr1_degradation(img, scale, rng):
    """LR1: Heavy degradation (blur + motion + noise + JPEG)"""
    img = img.astype(np.float32)
    
    # Step 1: Apply blur BEFORE downsampling (CRITICAL)
    img = _apply_gaussian_blur(img, rng)
    img = _apply_motion_blur(img, rng)
    
    # Step 2: Downsample
    img = _apply_downsample(img, scale, rng)
    
    # Step 3: Apply noise and compression AFTER downsampling
    img = _apply_noise(img, rng)
    img = _apply_jpeg_compression(img, rng)
    
    # Return as uint8
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_lr2_degradation(img, scale, rng):
    """LR2: Lighter degradation (blur + noise, less JPEG)"""
    img = img.astype(np.float32)
    
    # Step 1: Light blur only
    img = _apply_gaussian_blur(img, rng)
    
    # Step 2: Downsample
    img = _apply_downsample(img, scale, rng)
    
    # Step 3: Light noise, occasional JPEG
    img = _apply_noise(img, rng)
    if rng.random() > 0.5:  # 50% chance
        img = _apply_jpeg_compression(img, rng)
    
    # Return as uint8
    return np.clip(img, 0, 255).astype(np.uint8)

# ==============================================================================
# DATASET (No caching - memory safe)
# ==============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir=None, scale=4, degrade=True, crop_size=96, 
                 return_lr_pair=False, num_workers=0):
        self.hr_dir = hr_dir
        self.scale = scale
        self.crop_size = crop_size  # HR crop size (96 for x4 = LR 24)
        self.return_lr_pair = return_lr_pair
        
        # Get image list ONCE
        valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')
        self.images = []
        
        for f in os.listdir(hr_dir):
            if f.lower().endswith(valid_extensions):
                self.images.append(f)
        
        self.images = sorted(self.images)
        
        # Initialize RNG per worker
        self.rng = np.random.RandomState()
    
    def __len__(self):
        return len(self.images)
    
    def _load_hr(self, idx):
        """Load HR image (no caching)"""
        name = self.images[idx]
        hr_path = os.path.join(self.hr_dir, name)
        hr = cv2.imread(hr_path)
        
        if hr is None:
            raise ValueError(f"Failed to load {hr_path}")
        
        return hr, name
    
    def _random_crop_consistent(self, hr, lr1, lr2):
        """Crop HR and LR consistently"""
        if self.crop_size is None:
            return hr, lr1, lr2
        
        h_hr, w_hr = hr.shape[:2]
        lr_size = self.crop_size // self.scale  # 96/4 = 24
        
        # Ensure minimum size
        if h_hr < self.crop_size or w_hr < self.crop_size:
            # Resize if too small
            hr = cv2.resize(hr, (self.crop_size, self.crop_size))
            lr1 = cv2.resize(lr1, (lr_size, lr_size))
            lr2 = cv2.resize(lr2, (lr_size, lr_size))
            return hr, lr1, lr2
        
        # Random crop
        top = self.rng.randint(0, h_hr - self.crop_size + 1)
        left = self.rng.randint(0, w_hr - self.crop_size + 1)
        
        hr_crop = hr[top:top+self.crop_size, left:left+self.crop_size]
        
        # LR crop (aligned)
        lr_top = top // self.scale
        lr_left = left // self.scale
        
        lr1_crop = lr1[lr_top:lr_top+lr_size, lr_left:lr_left+lr_size]
        lr2_crop = lr2[lr_top:lr_top+lr_size, lr_left:lr_left+lr_size]
        
        return hr_crop, lr1_crop, lr2_crop
    
    def __getitem__(self, idx):
        # Load HR (no cache - memory safe)
        hr, name = self._load_hr(idx)
        
        # Create degradations on-the-fly (CPU-friendly)
        # Use different RNG seeds for variety
        rng1 = np.random.RandomState(self.rng.randint(0, 2**31))
        rng2 = np.random.RandomState(self.rng.randint(0, 2**31))
        
        # Apply degradations
        lr1 = apply_lr1_degradation(hr.copy(), self.scale, rng1)
        lr2 = apply_lr2_degradation(hr.copy(), self.scale, rng2)
        
        # Consistent cropping
        hr_crop, lr1_crop, lr2_crop = self._random_crop_consistent(hr, lr1, lr2)
        
        # BGR → RGB
        hr_crop = cv2.cvtColor(hr_crop, cv2.COLOR_BGR2RGB)
        lr1_crop = cv2.cvtColor(lr1_crop, cv2.COLOR_BGR2RGB)
        lr2_crop = cv2.cvtColor(lr2_crop, cv2.COLOR_BGR2RGB)
        
        # To tensor (0-1 range)
        hr_tensor = torch.from_numpy(hr_crop).float() / 255.0
        lr1_tensor = torch.from_numpy(lr1_crop).float() / 255.0
        lr2_tensor = torch.from_numpy(lr2_crop).float() / 255.0
        
        # HWC → CHW
        hr_tensor = hr_tensor.permute(2, 0, 1)
        lr1_tensor = lr1_tensor.permute(2, 0, 1)
        lr2_tensor = lr2_tensor.permute(2, 0, 1)
        
        return lr1_tensor, lr2_tensor, hr_tensor