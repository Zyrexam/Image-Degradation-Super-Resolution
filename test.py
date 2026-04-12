import sys
import os
import torch
import cv2
import numpy as np
from model import DegradationAwareSR

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Force CPU regardless of CUDA availability
DEVICE = "cpu"

# Load model

model = DegradationAwareSR(scale=4).to(DEVICE)
model.load_state_dict(torch.load("checkpoints_cpu/phase_a_complete.pth"))
model.eval()

# Load test image

img_path = "test3.jpg"
output_path = "output3.png"

img = cv2.imread(img_path)

if img is None:
    raise ValueError(f"Image not found at {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to tensor (assuming this is already LR)

lr = torch.from_numpy(img).float() / 255.0


# Simulate LR properly (VERY IMPORTANT)
# scale = 4
# h, w = img.shape[:2]

# lr_img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

# # Convert to tensor
# lr = torch.from_numpy(lr_img).float() / 255.0
lr = lr.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# Inference (SINGLE INPUT — correct usage)

with torch.no_grad():
    sr, d_map = model(lr)

# Convert SR output to image

sr = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)

# Save result

cv2.imwrite(output_path, cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))

print(f" SR image saved at: {output_path}")

# Optional: visualize degradation map (first channel)

d_vis = d_map[0, 0].cpu().numpy()
d_vis = (d_vis - d_vis.min()) / (d_vis.max() - d_vis.min() + 1e-8)
d_vis = (d_vis * 255).astype(np.uint8)

cv2.imwrite("degradation_map.png", d_vis)
print("Degradation map saved as: degradation_map.png")
