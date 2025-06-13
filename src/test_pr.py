import numpy as np
from utils.eval_utils import compute_precision_recall
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import os
import torch
from torchvision.utils import make_grid
from models.conv_generator import ConvGenerator
from models.generator import Generator

# def calculate_fid(real_features, fake_features):
#     """
#     Calculate the FID score between two sets of features.
#     This is a placeholder function; replace with actual FID calculation.
#     """
#     mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
#     mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
#     ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
#     covmean = sqrtm(sigma1.dot(sigma2))
    
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
    
#     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
#     return fid

# test_features = np.random.randn(10000, 128)

# # # Test identical features
# # precision, recall = compute_precision_recall(test_features, test_features.copy(), k=1)
# # print(f"Identical features test: P={precision:.4f}, R={recall:.4f}")

# # Test different features  
# #different_features = np.random.randn(10000, 128)
# #take half of the features from the test set and half from a far away distribution
# #different_features = np.concatenate((test_features[:5000], np.random.randn(5000, 128) + 50), axis=0)
# # generare different features by adding noise
# different_features = test_features + np.random.randn(*test_features.shape) * 0.5
# precision, recall = compute_precision_recall(test_features, different_features, k=3)
# print(f"Different features test: P={precision:.4f}, R={recall:.4f}")

# fid_score = calculate_fid(test_features, different_features)
# print(f"FID score: {fid_score:.4f}")

def save_generated_images(epoch, generator, latent_dim, device, subset_percentage,
                         dataset_type='digits', fixed_noise=None, base_path="./generated_images"):
    """
    Generate and save a grid of images from the generator.
    
    Args:
        epoch (int): Current training epoch
        generator (Generator): The generator model
        latent_dim (int): Size of the latent dimension
        device (torch.device): Device to run the generator on
        subset_percentage (int): Percentage of data used for training
        fixed_noise (torch.Tensor, optional): Fixed noise for consistent comparisons
        base_path (str): Base directory to save images
        
    Returns:
        torch.Tensor: Grid of generated images
    """
    # Create directory structure
    subfolder = f"{dataset_type}_subset_{subset_percentage}_percent"
    save_path = os.path.join(base_path, subfolder)
    os.makedirs(save_path, exist_ok=True)
    
    # Create a batch of latent vectors or use fixed noise for comparison
    if fixed_noise is None:
        z = torch.randn(64, latent_dim).to(device)
    else:
        z = fixed_noise
    
    # Generate images
    with torch.no_grad():
        gen_imgs = generator(z).detach().cpu()
    
    # Rescale images from [-1, 1] to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Create image grid
    grid = make_grid(gen_imgs, nrow=8, normalize=True)
    
    # Save image
    filename = f"epoch_{epoch:03d}.png"
    filepath = os.path.join(save_path, filename)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.title(f"Generated {dataset_type.capitalize()} Images - {subset_percentage}% Data - Epoch {epoch}")
    plt.savefig(filepath)
    plt.close()  # Close to free memory
    
    return grid

#generator = ConvGenerator(latent_dim=100, ngf=64, image_channels=3 ).to('cuda')
generator = Generator(latent_dim=128, hidden_dim=256, output_dim=28 * 28).to('cuda')
#load weights from a path 
#generator_path = 'informed_experiments/100_epoch_cifar_easy\models\cifar10_subset_50_percent\generator.pth'
generator_path = 'informed_experiments/n_easiest_digits\models\digits_subset_70_percent\generator.pth'
generator.load_state_dict(torch.load(generator_path, map_location='cuda'))

save_generated_images(
    epoch=100, 
    generator=generator, 
    latent_dim=128, 
    device='cuda', 
    subset_percentage=50, 
    dataset_type='cifar10', 
    fixed_noise=None, 
    base_path="./cifar10_generated_images"
)
