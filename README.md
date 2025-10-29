# AI-for-Satellite
MVP of deep learning software for satellite image classification and geospatial anomaly visualization,
using a ResNet-18 backbone with custom head for multi-class land cover segmentation.

### Features
- Training pipeline with Albumentations augmentations and PyTorch DataLoader.
- Pretrained ResNet-18 adaptation for variable input channels.
- Interactive Folium map.

### Technologies
- Python
- PyTorch, Torchvision - model backbone and training
- Albumentations - image augmentations
- Scikit-learn - accuracy metrics
- Folium, Rasterio - geospatial visualization
- NumPy, PIL - image I/O and preprocessing
