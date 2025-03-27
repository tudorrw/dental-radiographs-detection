import lightning as L
import os
from torch.utils.data import DataLoader, Dataset

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.txt'))
        image = ...  # Load image
        labels = ...  # Load labels

        if self.transform:
            image = self.transform(image)

        return image, labels

class YOLODataModule(L.LightningDataModule):
    def __init__(self, img_dir, label_dir, batch_size=16):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = YOLODataset(self.img_dir, self.label_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)