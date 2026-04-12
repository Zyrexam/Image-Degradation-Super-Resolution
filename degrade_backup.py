import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

# ==============================================================================
# FAST DEGRADATION (CPU-optimized)
# ==============================================================================

def _apply_gaussian_blur(img, rng):
    k = rng.choice([3, 5, 7])  # Reduce kernel sizes (faster)
    sigma = rng.uniform(0.5, 2.0)  # Reduce sigma range
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)

def _apply_noise(img, rng):
    """Faster noise: only gaussian"""
    sigma = rng.uniform(2.0, 8.0)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255)

def _apply_downsample(img, scale, rng):
    """Single interpolation method (fastest)"""
    h, w = img.shape[:2]
    # Use INTER_AREA for downsampling (built-in optimization)
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

def apply_lr1_degradation(img, scale, rng):
    """LR1: Blur + Noise (no JPEG) - FAST"""
    img = img.astype(np.float32)
    img = _apply_gaussian_blur(img, rng)
    img = _apply_downsample(img, scale, rng)
    img = _apply_noise(img, rng)
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_lr2_degradation(img, scale, rng):
    """LR2: Gaussian blur + downsample (no JPEG) - FASTER"""
    img = img.astype(np.float32)
    img = _apply_gaussian_blur(img, rng)  # Different blur params
    img = _apply_downsample(img, scale, rng)
    return np.clip(img, 0, 255).astype(np.uint8)

# ==============================================================================
# FAST DATASET WITH SMART CACHING (WINDOWS COMPATIBLE)
# ==============================================================================

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir=None, scale=4, degrade=True, crop_size=128, 
                 return_lr_pair=False, num_workers=0, cache_size=500):
        self.hr_dir = hr_dir
        self.scale = scale
        self.crop_size = crop_size
        self.return_lr_pair = return_lr_pair
        
        # Get image list ONCE - Windows compatible
        valid_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')
        self.images = []
        
        for f in os.listdir(hr_dir):
            if f.lower().endswith(valid_extensions):
                self.images.append(f)
        
        self.images = sorted(self.images)  # Deterministic order
        
        # Initialize RNG per worker
        self.rng = np.random.RandomState()
        
        # Pre-compute LR pairs on initialization (one-time cost)
        self.cache = {}
        self._precompute_degradations()
    
    def _precompute_degradations(self):
        """Pre-degrade all images once (happens at init, not per epoch)"""
        print(f"🔄 Pre-computing degradations for {len(self.images)} images...")
        
        failed_count = 0
        for idx, name in enumerate(self.images):
            hr_path = os.path.join(self.hr_dir, name)
            
            try:
                hr = cv2.imread(hr_path)
                
                if hr is None:
                    print(f"  ⚠️ Failed to load: {name}")
                    failed_count += 1
                    continue
                
                # Store HR in cache
                self.cache[name] = {
                    'hr': hr,
                    'lr1': apply_lr1_degradation(hr.copy(), self.scale, self.rng),
                    'lr2': apply_lr2_degradation(hr.copy(), self.scale, self.rng),
                }
                
                if (idx + 1) % 500 == 0:
                    print(f"  {idx+1}/{len(self.images)} images cached")
            
            except Exception as e:
                print(f"  ⚠️ Error processing {name}: {e}")
                failed_count += 1
                continue
        
        print(f"✅ Cache complete: {len(self.cache)} images (failed: {failed_count})")
        
        # Update image list to only valid cached images
        self.images = list(self.cache.keys())

    def __len__(self):
        return len(self.cache)

    def _random_crop(self, img):
        if self.crop_size is None:
            return img
        h, w = img.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            return img
        top = self.rng.randint(0, max(1, h - self.crop_size + 1))
        left = self.rng.randint(0, max(1, w - self.crop_size + 1))
        return img[top:top + self.crop_size, left:left + self.crop_size]

    def __getitem__(self, idx):
        name = self.images[idx]
        
        # All data is already cached (instant retrieval)
        data = self.cache[name]
        hr = data['hr'].copy()
        lr1 = data['lr1'].copy()
        lr2 = data['lr2'].copy()
        
        # # Random crop (per epoch variation)
        # hr = self._random_crop(hr)
        # lr1 = self._random_crop(lr1)
        # lr2 = self._random_crop(lr2)
        
        
        
        # Define sizes
        lr_size = self.crop_size
        hr_size = lr_size * self.scale

        h, w = hr.shape[:2]

        if h < hr_size or w < hr_size:
            hr = cv2.resize(hr, (hr_size, hr_size))
            lr1 = cv2.resize(lr1, (lr_size, lr_size))
            lr2 = cv2.resize(lr2, (lr_size, lr_size))
        else:
            top = self.rng.randint(0, h - hr_size + 1)
            left = self.rng.randint(0, w - hr_size + 1)

            # HR crop
            hr = hr[top:top+hr_size, left:left+hr_size]

            # LR crop (aligned)
            lr_top = top // self.scale
            lr_left = left // self.scale

            lr1 = lr1[lr_top:lr_top+lr_size, lr_left:lr_left+lr_size]
            lr2 = lr2[lr_top:lr_top+lr_size, lr_left:lr_left+lr_size]
        
        # BGR → RGB (fast)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr1 = cv2.cvtColor(lr1, cv2.COLOR_BGR2RGB)
        lr2 = cv2.cvtColor(lr2, cv2.COLOR_BGR2RGB)
        
        # To tensor (fast)
        hr = torch.from_numpy(hr).float() / 255.0
        lr1 = torch.from_numpy(lr1).float() / 255.0
        lr2 = torch.from_numpy(lr2).float() / 255.0
        
        # HWC → CHW
        hr = hr.permute(2, 0, 1)
        lr1 = lr1.permute(2, 0, 1)
        lr2 = lr2.permute(2, 0, 1)
        
        return lr1, lr2, hr