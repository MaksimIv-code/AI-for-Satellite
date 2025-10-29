import torch.nn as nn
import torchvision.models as models

class ClassifierModel(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True):
        super().__init__()
        # Replace the conv1 if in_channels != 3 
        self.backbone = models.resnet18(pretrained=pretrained)
        if in_channels != 3:
            old = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(in_channels, old.out_channels,
                                            kernel_size=old.kernel_size,
                                            stride=old.stride,
                                            padding=old.padding,
                                            bias=old.bias)
        # Replace the head
        feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feats, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.classifier(f)
        return out