import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Block 1: Feature Extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Block 2: Deepening the network
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 3: Fully Connected
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
class EmotionTransformer(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionTransformer, self).__init__()
        
        # Loads a pre-trained Vision Transformer (ViT-B/16)
        # Weights are pre-trained on ImageNet for general feature recognition
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # ViT expects 3-channel (RGB) images, but FERPlus is 1-channel (Grayscale)
        # We modify the first layer to accept grayscale
        self.vit.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
        
        # Replace the final head with a custom one for 8 classes
        self.vit.heads = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        return self.vit(x)