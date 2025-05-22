import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os

def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def load_data(batch_size, subset_percentage=100, dataset_type='digits', train=True, subset_strategy='random'):
    if subset_strategy in ['easiest', 'hardest', 'easiest_balanced']:
        return load_data_informed(batch_size, subset_percentage, dataset_type, subset_strategy)
    elif subset_strategy == 'random':
        return load_data_random(batch_size, subset_percentage, dataset_type, train)
    else:
        raise ValueError("subset_strategy must be either or 'easiest', 'hardest', 'easiest_balanced' 'random'")

def load_data_informed(batch_size, subset_percentage=100, dataset_type='digits', subset_strategy='easy'):
    
    if subset_strategy == 'easiest':
        data_path = f'src/data/easiest_subsets_{dataset_type}'
        file_name = f"{dataset_type}_train_easiest_{subset_percentage}p.pt"
        file_path = os.path.join(data_path, file_name)
        print(f"Loading easiest subset from {file_path}")
    elif subset_strategy == 'hardest':
        data_path = f'src/data/hardest_subsets_{dataset_type}'
        file_name = f"{dataset_type}_train_hardest_{subset_percentage}p.pt"
        file_path = os.path.join(data_path, file_name)
    elif subset_strategy == 'easiest_balanced':
        data_path = f'src/data/easiest_balanced_subsets_{dataset_type}'
        file_name = f"{dataset_type}_train_easiest_balanced_{subset_percentage}p.pt"
        file_path = os.path.join(data_path, file_name)
    
    dataset = torch.load(file_path)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Discard incomplete batches
        num_workers=4,   # Use multiple workers for faster loading
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2 
    )
    
    print(f"\nLoaded dataset from {file_path}")
    print(f"Selected subset size: {len(data_loader.dataset)} images")
    print(f"Number of batches: {len(data_loader)}")
    
    return data_loader, dataset

def load_data_random(batch_size, subset_percentage=100, dataset_type='digits', train=True):
    """
    Load dataset with optional stratified subset selection.
    
    Args:
        batch_size (int): Batch size for the data loader
        subset_percentage (int): Percentage of data to use (1-100)
        dataset_type (str): Type of dataset ('digits' for MNIST or 'fashion' for Fashion MNIST)
        
    Returns:
        tuple: (DataLoader, Dataset) for the selected subset
    """
    
    dataset_type = dataset_type.lower()
    
    if dataset_type in ['digits', 'fashion']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    elif dataset_type == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load the appropriate dataset based on dataset_type
    if dataset_type == 'digits':
        full_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        class_names = [str(i) for i in range(10)]  # 0-9 digits
    elif dataset_type == 'fashion':
        full_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_type == 'cifar10':
        full_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog',
                       'frog', 'horse', 'ship', 'truck']
    else:
        raise ValueError("dataset_type must be either 'digits' or 'fashion'")

    # Get all targets as numpy array for easier processing
    all_targets = np.array(full_dataset.targets)

    # Count distribution of digits in full dataset
    digit_counts = [0] * 10
    for label in all_targets:
        digit_counts[label] += 1
    
    print(f"Full {dataset_type} dataset distribution:")
    for class_idx, count in enumerate(digit_counts):
        class_label = class_names[class_idx] if dataset_type.lower() == 'fashion' else f"Digit {class_idx}"
        print(f"{class_label}: {count} samples")
    
    if subset_percentage == 100:
        # Use the full dataset
        selected_dataset = full_dataset
    else:
        # Create a stratified subset
        digit_indices = [[] for _ in range(10)]
        
        # Group indices by digit
        for idx, label in enumerate(all_targets):
            digit_indices[label].append(idx)
        
        # Calculate total subset size and samples per digit
        total_subset_size = int(len(full_dataset) * subset_percentage / 100)
        samples_per_digit = total_subset_size // 10
        
        # Create stratified subset
        stratified_indices = []
        for digit in range(10):
            digit_idx = digit_indices[digit]
            random_idx = torch.randperm(len(digit_idx))
            selected_idx = [digit_idx[i] for i in random_idx[:samples_per_digit]]
            stratified_indices.extend(selected_idx)
        
        # Shuffle the indices
        random.shuffle(stratified_indices)
        
        # Create the subset
        selected_dataset = Subset(full_dataset, stratified_indices)
        
        # Count distribution of digits in subset
        subset_digit_counts = [0] * 10
        for idx in stratified_indices:
            label = all_targets[idx]
            subset_digit_counts[label] += 1
        
        print(f"\nStratified subset distribution ({dataset_type}):")
        for class_idx, count in enumerate(subset_digit_counts):
            class_label = class_names[class_idx] if dataset_type.lower() == 'fashion' else f"Digit {class_idx}"
            print(f"{class_label}: {count} samples")
    
    # Create data loader
    data_loader = DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # Discard incomplete batches
        num_workers=4,   # Use multiple workers for faster loading
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2 
    )
    
    print(f"\nFull dataset size: {len(full_dataset)} images")
    print(f"Selected subset size: {len(selected_dataset)} images")
    print(f"Number of batches: {len(data_loader)}")
    
    return data_loader, selected_dataset