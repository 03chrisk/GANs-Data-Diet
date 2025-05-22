import torch
import torch.nn as nn
from torchvision import models

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # Get all modules except the final classifier
        modules = list(inception.children())[:-1]
        self.blocks = nn.ModuleList()
        
        # Split the model into smaller blocks to reduce memory usage
        self.blocks.append(nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ))
        
        self.blocks.append(nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ))
        
        self.blocks.append(nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d
        ))
        
        self.blocks.append(nn.Sequential(
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e
        ))
        
        self.blocks.append(nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        ))

        for block in self.blocks:
            block.eval()
    
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        with torch.no_grad():
            for block in self.blocks:
                x = block(x)

            features = torch.flatten(x, 1)
            
        return features

def preprocess_for_inception(images):
    """Inception V3 expects images in range [-1, 1]
    If images are already in range [-1, 1], we keep them as is
    If images are in range [0, 1], we need to rescale them"""
    
    if images.min() >= -1 and images.max() <= 1:
        pass
    elif images.min() >= 0 and images.max() <= 1:
        images = images * 2 - 1
    
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    return images