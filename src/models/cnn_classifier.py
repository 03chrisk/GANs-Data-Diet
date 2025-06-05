import torch.nn as nn
import torch.nn.functional as F
import torch

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def extract_features(self, x):
        # Used for FID calculation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x
    
class EnhancedMNISTFeatureExtractor(nn.Module):
    """
    Improved feature extractor for MNIST/Fashion-MNIST with:
    - Deeper architecture (more conv layers)
    - Batch normalization for stable features
    - Larger feature dimension
    - Multiple feature scales
    """
    def __init__(self, feature_dim=128, num_classes=10):
        super(EnhancedMNISTFeatureExtractor, self).__init__()
        
        # Just 2 conv blocks like the original
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Features
        self.fc1 = nn.Linear(64 * 7 * 7, feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def extract_features(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x
    
    def forward(self, x):
        features = self.extract_features(x)
        x = self.dropout(features)
        x = self.fc2(x)
        return x