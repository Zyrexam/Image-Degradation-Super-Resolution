import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from psdc_sr_v2 import PSDCSRModel, Trainer


# -------------------------------
# Dataset
# -------------------------------
class HRDataset(Dataset):
    def __init__(self, root_dir, crop_size=128):
        self.paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.crop_size = crop_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
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

    # ⚠️ CHANGE THIS PATH
    data_path = "flickr2k/Flickr2K"  # folder with images

    dataset = HRDataset(data_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # smaller model for your GPU
    model = PSDCSRModel(
        scale=4,
        base_ch=32,
        latent_ch=16
    )

    trainer = Trainer(model, device=device)

    trainer.fit(loader, num_epochs=10)