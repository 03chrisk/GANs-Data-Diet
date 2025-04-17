"""
Visualization utilities for the GAN project.
Supports both MNIST digits and Fashion MNIST datasets.
"""
import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

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
        z = torch.randn(25, latent_dim).to(device)
    else:
        z = fixed_noise
    
    # Generate images
    with torch.no_grad():
        gen_imgs = generator(z).detach().cpu()
    
    # Rescale images from [-1, 1] to [0, 1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Create image grid
    grid = make_grid(gen_imgs, nrow=5, normalize=True)
    
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

def plot_losses(g_losses, d_losses, subset_percentage, dataset_type='digits', save_path="./loss_plots"):
    """
    Plot and save generator and discriminator losses.
    
    Args:
        g_losses (list): Generator losses
        d_losses (list): Discriminator losses
        subset_percentage (int): Percentage of data used for training
        save_path (str): Directory to save the plot
    """
    # Create folder structure
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title(f'{dataset_type.capitalize()} GAN Training Losses - {subset_percentage}% Data')
    
    # Save plot
    plt.savefig(f"{save_path}/{dataset_type}_losses_subset_{subset_percentage}_percent.png")
    plt.close()  # Close to free memory

def show_random_samples(generator, latent_dim, device, dataset_type='digits', n_samples=16):
    """
    Generate and display random samples from the generator.
    
    Args:
        generator (Generator): The generator model
        latent_dim (int): Size of the latent dimension
        device (torch.device): Device to run the generator on
        n_samples (int): Number of samples to generate
    """
    # Generate random samples
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(n_samples, latent_dim).to(device)
        # Generate images
        samples = generator(z).detach().cpu()
        # Rescale images
        samples = 0.5 * samples + 0.5
        
    # Display images
    grid = make_grid(samples, nrow=int(np.sqrt(n_samples)), normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.title(f"Random {dataset_type.capitalize()} Samples from Trained Generator")
    plt.show()