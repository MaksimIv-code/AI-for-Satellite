import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

def read_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, bands=None):
        if classes is None:
            classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.classes = classes
        self.class2idx = {c:i for i,c in enumerate(self.classes)}
        self.samples = []
        for c in self.classes:
            p = os.path.join(root_dir, c)
            for f in os.listdir(p):
                if f.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(p,f), self.class2idx[c]))
        self.transform = transform
        self.bands = bands

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = read_image(path)  # HWC uint8
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        # To CHW float32 0..1
        img = img.astype('float32') / 255.0
        img = np.transpose(img, (2,0,1))
        return img, label