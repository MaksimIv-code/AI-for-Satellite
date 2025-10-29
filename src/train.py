import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from dataset import SatelliteDataset
from model import ClassifierModel
import albumentations as A

def get_transforms(train=True, size=64):
    if train:
        return A.Compose([
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        ])
    else:
        return A.Compose([
            A.CenterCrop(size, size),
        ])

def train_loop(data_dir, epochs=10, bs=16, lr=1e-4, size=64, device='cuda'):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    classes = sorted(os.listdir(train_dir))
    num_classes = len(classes)
    print("Classes:", classes)

    train_ds = SatelliteDataset(train_dir, classes, transform=get_transforms(True, size))
    print("Train dataset size:", len(train_ds))

    # One element test
    img, label = train_ds[0]
    print("Image shape:", img.shape, "Label:", label)
    val_ds = SatelliteDataset(val_dir, classes, transform=get_transforms(False, size))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)

    model = ClassifierModel(num_classes=num_classes, in_channels=3, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * imgs.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                p = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(p.tolist())
                truths.extend(labels.cpu().numpy().tolist())
        acc = accuracy_score(truths, preds)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state': model.state_dict(),
                'classes': classes
            }, 'best_model.pth')
            print("Saved best_model.pth")
    print("Training finished. Best val acc:", best_acc)

if __name__ == "__main__":
    train_loop(data_dir='data')