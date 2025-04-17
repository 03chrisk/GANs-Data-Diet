import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator network that evaluates whether an image is real or fake.
    
    The architecture consists of:
    - Input: Flattened image (28×28 = 784 dimensions)
    - Multiple fully connected layers with LeakyReLU activations and dropout
    - Output: Single value between 0-1 (probability of image being real)
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the Discriminator network.
        
        Args:
            input_dim (int): Size of the input image (flattened)
            hidden_dim (int): Size of the hidden layers
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0-1 (probability of being real)
        )

    def forward(self, img):
        """
        Forward pass of the discriminator.
        
        Args:
            img (torch.Tensor): Input image of shape (batch_size, 1, 28, 28)
            
        Returns:
            torch.Tensor: Probability of the image being real, shape (batch_size, 1)
        """
        img_flat = img.view(img.size(0), -1)  # Flatten the image
        validity = self.model(img_flat)
        return validity