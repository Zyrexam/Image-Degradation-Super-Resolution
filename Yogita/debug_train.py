import torch
from torch.utils.data import DataLoader
from psdc_sr_v2 import PSDCSRModel, Trainer

class DummyDataset:
    def __len__(self): return 500
    def __getitem__(self, idx):
        return {'hr': torch.rand(3,128,128)}

loader = DataLoader(DummyDataset(), batch_size=2)

model = PSDCSRModel(scale=4)
trainer = Trainer(model, device='cpu')

trainer.fit(loader, num_epochs=2)