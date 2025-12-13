import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class PneumoniaNet(nn.Module):
    def __init__(self):
        super(PneumoniaNet, self).__init__()
        # Load pre-trained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer
        # Output is 1 feature (logits) for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)
