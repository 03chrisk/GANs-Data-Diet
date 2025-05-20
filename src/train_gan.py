import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from models.generator import Generator
from models.discriminator import Discriminator
from models.conv_generator import ConvGenerator
from models.conv_discirminator import ConvDiscriminator

from models.conv_discirminator import weights_init_normal

from utils.data_utils import (
    set_random_seed, 
    load_data, 
)

from utils.vizualization import (
    save_generated_images, 
    plot_losses
)

from configs import mnist_config as config


def train_gan(subset_percentage=100, dataset_type='digits'):
    """
    Train the GAN on the chosen dataset.
    
    Args:
        subset_percentage (int): Percentage of data to use for training
        dataset_type (str): Type of dataset ('digits' for MNIST or 'fashion' for Fashion MNIST)
        
    Returns:
        tuple: (g_losses, d_losses) - Lists of generator and discriminator losses
    """
    # Print device information just once at the start
    print(f"Using device: {config.DEVICE}")
    
    set_random_seed(config.RANDOM_SEED)
    
    # Load data
    train_loader, _ = load_data(
        batch_size=config.BATCH_SIZE, 
        subset_percentage=subset_percentage,
        dataset_type=dataset_type
    )
    
    if dataset_type in ['digits', 'fashion']:
        generator = Generator(
            config.LATENT_DIM, 
            config.HIDDEN_DIM, 
            config.IMAGE_SIZE
        ).to(config.DEVICE)
        
        discriminator = Discriminator(
            config.IMAGE_SIZE, 
            config.HIDDEN_DIM
        ).to(config.DEVICE)
    elif dataset_type == 'cifar10':
        generator = ConvGenerator(
            config.LATENT_DIM, 
            config.HIDDEN_DIM, 
            config.IMAGE_CHANNELS_CIFAR10
        ).to(config.DEVICE)
        
        discriminator = ConvDiscriminator(
            config.HIDDEN_DIM,
            config.IMAGE_CHANNELS_CIFAR10
        ).to(config.DEVICE)
        
        generator.main.apply(weights_init_normal)
        discriminator.main.apply(weights_init_normal)
    else:
        raise ValueError(
            f"Dataset type '{dataset_type}' is not supported. "
            "Supported types: 'digits', 'fashion', 'cifar10'"
        )
        
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(
        generator.parameters(), 
        lr=config.LEARNING_RATE, 
        betas=(config.BETA1, config.BETA2)
    )
    
    optimizer_D = optim.Adam(
        discriminator.parameters(), 
        lr=config.LEARNING_RATE, 
        betas=(config.BETA1, config.BETA2)
    )
    
    # Initialize tracking variables
    start_epoch = 0
    g_losses = []
    d_losses = []
    
    # Generate fixed noise for consistent image generation
    fixed_noise = torch.randn(25, config.LATENT_DIM).to(config.DEVICE)
    
    start_time = time.time()
    print(f"Starting Training on {dataset_type.capitalize()} Dataset...")
    
    # Create output directories
    checkpoint_dir = os.path.join(
        config.MODELS_PATH, 
        f"{dataset_type}_subset_{subset_percentage}_percent"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create labels with a small amount of label smoothing for stability
    valid = torch.ones(config.BATCH_SIZE, 1).to(config.DEVICE) * 0.9  # Real: 0.9 instead of 1
    fake = torch.zeros(config.BATCH_SIZE, 1).to(config.DEVICE)
    
    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        g_loss_epoch = 0
        d_loss_epoch = 0
        batch_count = 0
        
        for i, (real_imgs, _) in enumerate(train_loader):
            batch_count += 1
            
            # Configure input
            real_imgs = real_imgs.to(config.DEVICE)
            batch_size = real_imgs.size(0)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Loss on real images
            real_pred = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)
            
            # Sample noise and generate fake images
            z = torch.randn(batch_size, config.LATENT_DIM).to(config.DEVICE)
            fake_imgs = generator(z)
            
            # Loss on fake images
            fake_pred = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, config.LATENT_DIM).to(config.DEVICE)
            fake_imgs = generator(z)
            
            # Try to fool the discriminator
            validity = discriminator(fake_imgs)
            g_loss = -torch.mean(torch.log(validity + 1e-8))
            
            g_loss.backward()
            optimizer_G.step()
            
            # Save losses for plotting
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            
            # Print progress for a few batches
            # if i % 100 == 0:
            #     elapsed = time.time() - start_time
            #     print(f"[Epoch {epoch}/{config.NUM_EPOCHS}] "
            #           f"[Batch {i}/{len(train_loader)}] "
            #           f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
            #           f"[Time: {elapsed:.2f}s]")
        
        # Calculate and store average losses for this epoch
        g_losses.append(g_loss_epoch / batch_count)
        d_losses.append(d_loss_epoch / batch_count)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"[Epoch {epoch}/{config.NUM_EPOCHS}] "
              f"[Avg D loss: {d_losses[-1]:.4f}] [Avg G loss: {g_losses[-1]:.4f}] "
              f"[Epoch time: {epoch_time:.2f}s]")
        
        # Only save checkpoints, generate images, and plot losses every 10 epochs or at the final epoch
        # if epoch % 10 == 0 or epoch == config.NUM_EPOCHS - 1:
        #     # Save generated images
        #     save_start = time.time()
        #     _ = save_generated_images(
        #         epoch, 
        #         generator, 
        #         config.LATENT_DIM, 
        #         config.DEVICE, 
        #         subset_percentage,
        #         dataset_type,
        #         fixed_noise,
        #         config.GENERATED_IMAGES_PATH
        #     )
        #     print(f"Image saving took: {time.time() - save_start:.2f}s")
            
            # Plot losses
            #plot_losses(g_losses, d_losses, subset_percentage, dataset_type, config.LOSS_PLOTS_PATH)
    
    # Training summary
    training_time = (time.time() - start_time) / 60
    print("Training finished!")
    print(f"Total training time: {training_time:.2f} minutes")
    
    # Save final models
    final_model_path = os.path.join(
        config.MODELS_PATH, 
        f"{dataset_type}_subset_{subset_percentage}_percent"
    )
    
    #torch.save(generator.state_dict(), f'{final_model_path}/generator_final.pth')
    #torch.save(discriminator.state_dict(), f'{final_model_path}/discriminator_final.pth')
    
    #print("Final models saved!")
    
    return g_losses, d_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MNIST GAN with different data subsets')
    parser.add_argument('--subset', type=int, default=100,
                        help='Percentage of data to use (default: 100)')
    parser.add_argument('--dataset', type=str, default='digits', choices=['digits', 'fashion', 'cifar10'],
                        help='Dataset to use: "digits" for MNIST or "fashion" for Fashion MNIST (default: digits)')
    
    args = parser.parse_args()
    
    g_losses, d_losses = train_gan(args.subset, args.dataset)