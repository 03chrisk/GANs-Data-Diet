import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.inception_feature_extractor import InceptionV3FeatureExtractor, preprocess_for_inception, preprocess_for_inception_imagenet
from models.cnn_classifier import MNISTClassifier, EnhancedMNISTFeatureExtractor
import numpy as np
from scipy.linalg import sqrtm
from torchvision import datasets, transforms
import tqdm
from sklearn.neighbors import NearestNeighbors


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
            
        fid = calculate_cnn_fid(generator, dataset, num_samples=num_samples, latent_dim=mnist_config.LATENT_DIM, device=mnist_config.DEVICE, dataset_type=dataset_type)
    
    elif dataset_type == 'cifar10':
        cifar10_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar10_transform)
        
        fid = calculate_inception_fid(generator, dataset, num_samples=num_samples, latent_dim=cifar_config.LATENT_DIM, device=cifar_config.DEVICE)
    
    return fid

def calculate_cnn_fid(generator, dataset, num_samples=5000, latent_dim=mnist_config.LATENT_DIM, device='cuda', dataset_type='digits'):
    #feature_extractor = EnhancedMNISTFeatureExtractor(feature_dim=128).to(device)
    feature_extractor = MNISTClassifier().to(device)
    if dataset_type == 'digits':
        feature_extractor.load_state_dict(torch.load('src/models/digit_mnist_classifier.pth', map_location=device))
        #feature_extractor.load_state_dict(torch.load('checkpoints\digits_feature_extractor_final.pth', map_location=device))
        print("Using digit MNIST classifier for FID calculation.")
    elif dataset_type == 'fashion':
        #feature_extractor.load_state_dict(torch.load('src/models/fashion_mnist_classifier.pth', map_location=device))
        feature_extractor.load_state_dict(torch.load('checkpoints/fashion_feature_extractor_final.pth', map_location=device))
        print("Using Fashion MNIST classifier for FID calculation.")
    else:
        raise ValueError("dataset_type must be either 'digits' or 'fashion'")

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
    
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
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
    
    fake_features_sum = None
    fake_features_sq_sum = None
    fake_count = 0
    
    print("Processing generated images...")
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
            
            del fake_batch, features
            torch.cuda.empty_cache()
    
    mu2 = fake_features_sum / fake_count
    sigma2 = fake_features_sq_sum / fake_count - np.outer(mu2, mu2)
    
    print("Calculating final FID score...")
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    
    return fid



# === PRECISION AND RECALL FUNCTIONS ===

def calculate_pr(generator, dataset_type, num_samples=10000, k=5):
    """
    Calculate precision and recall for GAN evaluation.
    
    Args:
        generator: The trained generator model
        dataset_type: Type of dataset ('digits', 'fashion', 'cifar10')
        num_samples: Number of samples to use for evaluation
        k: Number of nearest neighbors for manifold estimation
        
    Returns:
        tuple: (precision, recall) scores
    """
    if dataset_type in ['digits', 'fashion']:
        return calculate_pr_cnn(generator, dataset_type, num_samples, k)
    elif dataset_type == 'cifar10':
        return calculate_pr_inception(generator, dataset_type, num_samples, k)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def calculate_pr_cnn(generator, dataset_type, num_samples=10000, k=5):
    """
    Calculate precision and recall using CNN features for MNIST/Fashion-MNIST.
    """
    device = mnist_config.DEVICE
    
    print(f"Loading feature extractor for {dataset_type}...")
    #feature_extractor = EnhancedMNISTFeatureExtractor(feature_dim=128).to(device)
    feature_extractor = MNISTClassifier().to(device)
    if dataset_type == 'digits':
        feature_extractor.load_state_dict(torch.load('src/models/digit_mnist_classifier.pth', map_location=device))
        #feature_extractor.load_state_dict(torch.load('checkpoints\digits_feature_extractor_final.pth', map_location=device))
        print("Loaded digit MNIST classifier for P&R calculation.")
    elif dataset_type == 'fashion':
        #feature_extractor.load_state_dict(torch.load('src/models/fashion_mnist_classifier.pth', map_location=device))
        feature_extractor.load_state_dict(torch.load('checkpoints/fashion_feature_extractor_final.pth', map_location=device))
        print("Loaded fashion MNIST classifier for P&R calculation.")
    feature_extractor.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    if dataset_type == 'digits':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    real_features, fake_features = extract_features_cnn(
        generator, feature_extractor, dataset, num_samples, 
        mnist_config.LATENT_DIM, device, mnist_config.BATCH_SIZE
    )
    
    precision, recall = compute_precision_recall(real_features, fake_features, k)
    
    return precision, recall


def calculate_pr_inception(generator, dataset_type, num_samples=50000, k=5):
    """
    Calculate precision and recall using Inception features for CIFAR-10.
    """
    device = cifar_config.DEVICE
    
    feature_extractor = InceptionV3FeatureExtractor().to(device)
    feature_extractor.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    real_features, fake_features = extract_features_inception(
        generator, feature_extractor, dataset, num_samples,
        cifar_config.LATENT_DIM, device, cifar_config.BATCH_SIZE
    )
    
    precision, recall = compute_precision_recall(real_features, fake_features, k)
    
    return precision, recall


def extract_features_cnn(generator, feature_extractor, dataset, num_samples, latent_dim, device, batch_size):
    """
    Extract features from real and generated images using CNN feature extractor.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    real_features = []
    
    print("Extracting features from real images...")
    with torch.no_grad():
        for images, _ in dataloader:
            if len(real_features) * batch_size >= num_samples:
                break
            images = images.to(device)
            features = feature_extractor.extract_features(images).cpu().numpy()
            real_features.append(features)
    
    real_features = np.concatenate(real_features, axis=0)[:num_samples]
    
    fake_features = []
    
    print("Extracting features from generated images...")
    with torch.no_grad():
        num_batches = (num_samples + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_size_i = min(batch_size, num_samples - batch_start)
            z = torch.randn(batch_size_i, latent_dim).to(device)
            fake_images = generator(z)
            features = feature_extractor.extract_features(fake_images).cpu().numpy()
            fake_features.append(features)
            
            # if (i + 1) % 10 == 0:
            #     print(f"  Processed {min((i + 1) * batch_size, num_samples)}/{num_samples} generated images")
    
    fake_features = np.concatenate(fake_features, axis=0)[:num_samples]
    
    return real_features, fake_features


def extract_features_inception(generator, feature_extractor, dataset, num_samples, latent_dim, device, batch_size):
    """
    Extract features from real and generated images using Inception feature extractor.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    real_features = []
    
    print("Extracting features from real images...")
    with torch.no_grad():
        for images, _ in dataloader:
            if len(real_features) * batch_size >= num_samples:
                break
            images = images.to(device)
            images = preprocess_for_inception(images)
            features = feature_extractor(images).cpu().numpy()
            real_features.append(features)
    
    real_features = np.concatenate(real_features, axis=0)[:num_samples]

    fake_features = []
    
    print("Extracting features from generated images...")
    with torch.no_grad():
        num_batches = (num_samples + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_size_i = min(batch_size, num_samples - batch_start)
            z = torch.randn(batch_size_i, latent_dim).to(device)
            fake_images = generator(z)
            fake_images = preprocess_for_inception(fake_images)
            features = feature_extractor(fake_images).cpu().numpy()
            fake_features.append(features)
            
            del fake_images, features
            torch.cuda.empty_cache()
            
            # if (i + 1) % 10 == 0:
            #     print(f"  Processed {min((i + 1) * batch_size, num_samples)}/{num_samples} generated images")
    
    fake_features = np.concatenate(fake_features, axis=0)[:num_samples]
    
    return real_features, fake_features


def compute_precision_recall(real_features, fake_features, k=3):
    """
    Compute precision and recall using k-nearest neighbors manifold estimation.
    
    This implements the improved precision and recall metric from the paper:
    "Improved Precision and Recall Metric for Assessing Generative Models"
    
    Args:
        real_features: Feature vectors of real images
        fake_features: Feature vectors of generated images
        k: Number of nearest neighbors
        
    Returns:
        tuple: (precision, recall)
    """
    print(f"Computing precision and recall with k={k}...")
    
    if torch.is_tensor(real_features):
        real_features = real_features.cpu().numpy()
    if torch.is_tensor(fake_features):
        fake_features = fake_features.cpu().numpy()
    
    # Fit k-NN on real features
    print("Fitting k-NN on real features...")
    nbrs_real = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=4, algorithm='ball_tree').fit(real_features)
    distances_real, _ = nbrs_real.kneighbors(real_features)
    # Get distance to k-th nearest neighbor (excluding self)
    radii_real = distances_real[:, k]
    
    # Fit k-NN on fake features
    print("Fitting k-NN on fake features...")
    nbrs_fake = NearestNeighbors(n_neighbors=k+1, metric='euclidean', n_jobs=4, algorithm='ball_tree').fit(fake_features)
    distances_fake, _ = nbrs_fake.kneighbors(fake_features)
    # Get distance to k-th nearest neighbor (excluding self)
    radii_fake = distances_fake[:, k]
    
    print("Computing precision...")
    precision = compute_manifold_coverage(fake_features, real_features, radii_real)
    
    print("Computing recall...")
    recall = compute_manifold_coverage(real_features, fake_features, radii_fake)
    
    return precision, recall


def compute_manifold_coverage(query_features, reference_features, reference_radii):
    """
    Compute the fraction of query samples that fall within the reference manifold.

    Args:
        query_features: Features to check
        reference_features: Features defining the manifold
        reference_radii: Radii of hyperspheres around reference features

    Returns:
        float: Fraction of query features within reference manifold
    """
    n_query = len(query_features)
    n_covered = 0

    # Build a K-D tree or Ball Tree on the reference features
    # This allows for efficient radius queries
    nbrs_reference_tree = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='ball_tree', n_jobs=4).fit(reference_features)

    # For each query feature, find its nearest neighbor in the reference set
    # and check if the distance is less than or equal to the radius of that reference neighbor.
    # We can iterate through query_features in batches to manage memory.

    batch_size = 1000 # You can tune this
    for i in range(0, n_query, batch_size):
        batch = query_features[i:i + batch_size]
        
        # Query the tree for the nearest neighbor in reference_features
        # distances will be shape (batch_size, 1)
        # indices will be shape (batch_size, 1)
        distances, indices = nbrs_reference_tree.kneighbors(batch)

        # Check if the distance to the nearest reference point is less than or equal to its radius
        covered_batch = (distances.flatten() <= reference_radii[indices.flatten()])
        n_covered += np.sum(covered_batch)
            
    coverage = n_covered / n_query
    return coverage