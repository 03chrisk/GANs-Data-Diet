import torch.nn as nn

class Generator(nn.Module):
    """
    Generator network that transforms random noise vectors into synthetic images.
    
    The architecture consists of:
    - Input: Random noise vector (latent_dim)
    - Multiple fully connected layers with LeakyReLU activations
    - Output: Image with values in range [-1, 1] through Tanh activation
    """
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        Initialize the Generator network.
        
        Args:
            latent_dim (int): Size of the input noise vector
            hidden_dim (int): Size of the hidden layers
            output_dim (int): Size of the output image (flattened)
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, z):
        """
        Forward pass of the generator.
        
        Args:
            z (torch.Tensor): Random noise vector of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Generated images of shape (batch_size, 1, 28, 28)
        """
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)  # Reshape to image dimensions
        return img