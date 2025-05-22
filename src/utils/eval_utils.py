import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.inception_feature_extractor import InceptionV3FeatureExtractor, preprocess_for_inception
from models.cnn_classifier import MNISTClassifier
import numpy as np
from scipy.linalg import sqrtm
from torchvision import datasets, transforms

from configs import mnist_config, cifar_config

def calculate_fid(generator, dataset_type, num_samples=50000):
    if dataset_type in ['digits', 'fashion']:
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        if dataset_type == 'digits':
            dataset = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        else:
            dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transform)
            
        fid = calculate_cnn_fid(generator, dataset, num_samples=num_samples, latent_dim=mnist_config.LATENT_DIM, device=mnist_config.DEVICE)
    
    elif dataset_type == 'cifar10':
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar10_transform)
        
        fid = calculate_inception_fid(generator, dataset, num_samples=num_samples, latent_dim=cifar_config.LATENT_DIM, device=cifar_config.DEVICE)
    
    return fid

def calculate_cnn_fid(generator, dataset, num_samples=5000, latent_dim=mnist_config.LATENT_DIM, device='cuda'):
    feature_extractor = MNISTClassifier().to(device)
    feature_extractor.load_state_dict(torch.load('src\models\mnist_classifier.pth', map_location=device))

    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    
    real_images, labels = next(iter(dataloader))
    real_images = real_images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        fake_images = generator(z)
    
    with torch.no_grad():
        real_features = feature_extractor.extract_features(real_images).detach().cpu().numpy()
        fake_features = feature_extractor.extract_features(fake_images).detach().cpu().numpy()
    
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid

def calculate_inception_fid(generator, dataset, num_samples=5000, batch_size=cifar_config.BATCH_SIZE, latent_dim=cifar_config.LATENT_DIM, device='cuda'):
    """
    Memory-efficient implementation of FID calculation.
    Processes images in small batches and accumulates statistics rather than storing all images.
    """
    feature_extractor = InceptionV3FeatureExtractor().to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize variables to accumulate statistics
    real_features_sum = None
    real_features_sq_sum = None
    real_count = 0
    
    print("Processing real images...")
    # Process real images batch by batch
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            images = preprocess_for_inception(images)
            features = feature_extractor(images).cpu().numpy()
            
            # Accumulate statistics for mean and covariance
            if real_features_sum is None:
                real_features_sum = features.sum(axis=0)
                real_features_sq_sum = np.dot(features.T, features)
            else:
                real_features_sum += features.sum(axis=0)
                real_features_sq_sum += np.dot(features.T, features)
            
            real_count += features.shape[0]
            
            if real_count >= num_samples:
                break
    
    # Calculate mean and covariance for real images
    mu1 = real_features_sum / real_count
    sigma1 = real_features_sq_sum / real_count - np.outer(mu1, mu1)
    
    # Initialize variables for fake images
    fake_features_sum = None
    fake_features_sq_sum = None
    fake_count = 0
    
    print("Processing generated images...")
    # Process fake images batch by batch
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_size_i = min(batch_size, num_samples - i)
            z = torch.randn(batch_size_i, latent_dim).to(device)
            fake_batch = generator(z)
            
            fake_batch = preprocess_for_inception(fake_batch)
            
            features = feature_extractor(fake_batch).cpu().numpy()
            
            if fake_features_sum is None:
                fake_features_sum = features.sum(axis=0)
                fake_features_sq_sum = np.dot(features.T, features)
            else:
                fake_features_sum += features.sum(axis=0)
                fake_features_sq_sum += np.dot(features.T, features)
            
            fake_count += features.shape[0]
            
            # Free memory
            del fake_batch, features
            torch.cuda.empty_cache()
    
    mu2 = fake_features_sum / fake_count
    sigma2 = fake_features_sq_sum / fake_count - np.outer(mu2, mu2)
    
    # Calculate FID
    print("Calculating final FID score...")
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid