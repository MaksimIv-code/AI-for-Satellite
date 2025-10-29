import os
import folium
import rasterio
import torch
import numpy as np
from dataset import read_image
from model import ClassifierModel
from albumentations import Compose, Resize

def predict_image(model, classes, img_path, size=64, device='cpu'):
    img = read_image(img_path)
    aug = Compose([Resize(size, size)])
    img = aug(image=img)['image'].astype('float32') / 255.0
    img = np.transpose(img, (2,0,1))
    x = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
    return classes[idx], float(probs[idx])

def visualize_folder(folder, model_path='best_model.pth', zoom=8, out_html='map.html'):
    ckpt = torch.load(model_path, map_location='cpu')
    classes = ckpt['classes']
    model = ClassifierModel(num_classes=len(classes), in_channels=3, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    fmap = folium.Map(location=[52.0, 10.0], zoom_start=zoom, tiles='CartoDB positron')

    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith('.jpg'):
                continue
            path = os.path.join(root, f)
            with rasterio.open(path) as src:
                bounds = src.bounds
                lon = (bounds.left + bounds.right)/2
                lat = (bounds.top + bounds.bottom)/2

            pred, prob = predict_image(model, classes, path)
            color = {
                "Forest": "green",
                "Residential": "red",
                "River": "blue",
                "AnnualCrop": "yellow",
                "SeaLake": "cyan",
            }.get(pred, "gray")

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                popup=f"{os.path.basename(f)} → {pred} ({prob:.2f})"
            ).add_to(fmap)

    fmap.save(out_html)
    print(f"The map is saved: {out_html}")

if __name__ == "__main__":
    visualize_folder(folder='data/val')