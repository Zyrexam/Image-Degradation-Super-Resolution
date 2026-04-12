import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from psdc_sr_v3 import PSDCSRModel, Trainer


# -------------------------------
# Dataset
# -------------------------------
class HRDataset(Dataset):
    def __init__(self, root_dir, crop_size=128):
        self.paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Loaded {len(self.paths)} images")
        self.crop_size = crop_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            raise ValueError(f"Failed to load {self.paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        ch = self.crop_size

        # random crop
        x = torch.randint(0, w - ch, (1,)).item()
        y = torch.randint(0, h - ch, (1,)).item()
        img = img[y:y+ch, x:x+ch]

        img = torch.tensor(img).float() / 255.0
        img = img.permute(2, 0, 1)

        return {'hr': img}


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    
    data_path = "/content/flickr2k/Flickr2K"
    dataset = HRDataset(data_path)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    # smaller model for your GPU
    model = PSDCSRModel(
        scale=4,
        base_ch=32,
        latent_ch=16
    )

    trainer = Trainer(model, device=device)


    # LOAD CHECKPOINT (ADD THIS)

    ckpt_path = "/content/drive/MyDrive/CV_Project/checkpoints/ckpt_step_3200.pth"

    if os.path.exists(ckpt_path):
      print("Loading checkpoint...")

      ckpt = torch.load(ckpt_path, map_location=device)

      model.load_state_dict(ckpt['model_state'])
      trainer.optimizer.load_state_dict(ckpt['optimizer_state'])

      start_epoch = ckpt['epoch']
      start_step  = ckpt['step']

      print(f"Resuming from epoch {start_epoch}, step {start_step}")

    else:
      print("No checkpoint found, starting fresh")
      start_epoch = 0
      start_step  = 0

# TRAIN (RESUME)

    trainer.fit(
      loader,
      num_epochs=16,   # continue till ~25 total
      start_epoch=start_epoch,
      start_step=start_step
    )