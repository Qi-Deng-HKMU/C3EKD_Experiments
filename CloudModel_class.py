import torch
import torch.nn as nn
from torchvision import transforms, models


# Complex Cloud Model：ResNet101
class CloudModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CloudModel, self).__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)