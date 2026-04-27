"""
This performs VGG-16
"""

import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGG16Model(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)