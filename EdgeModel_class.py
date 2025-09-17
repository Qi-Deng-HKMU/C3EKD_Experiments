import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Lightweight Edge Model：MobileNetV2
class EdgeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EdgeModel, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)

        # Freeze the weights in the front layers and only fine-tune the last few layers.
        for param in self.backbone.features[:-3].parameters():
            param.requires_grad = False
        
        # simpler classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.7),  
            nn.Linear(self.backbone.last_channel, 64),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)
    

# 分布式边缘节点类
class EdgeNode:
    def __init__(self, node_id, model_state_dict):
        self.node_id = node_id
        self.model = EdgeModel().to(device)
        self.model.load_state_dict(model_state_dict)
        
    
    def predict(self, images, temperature=1.0):
        self.model.eval() 
        with torch.no_grad():
            logits = self.model(images)
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=1)
        return probs