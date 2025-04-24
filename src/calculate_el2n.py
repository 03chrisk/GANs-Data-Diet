import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm

def compute_el2n_scores(model_class, dataset, num_networks=10, num_iterations=1000, 
                        batch_size=64, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Computes EL2N scores for a dataset.
    
    Args:
        model_class: Class of the neural network model to train
        dataset: PyTorch dataset containing the examples to evaluate
        num_networks: Number of networks to train (K)
        num_iterations: Number of training iterations (t)
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to run the computation on ('cuda' or 'cpu')
        
    Returns:
        Dictionary mapping example indices to their EL2N scores
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Dictionary to store L2 norms for each example from each network
    example_l2_norms = {i: [] for i in range(len(dataset))}
    for k in range(num_networks):
        print(f"Training network {k+1}/{num_networks}")
        
        # Initialize a new network with random weights
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train the network for specified iterations
        model.train()
        for iteration in tqdm(range(num_iterations)):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Convert targets to one-hot encoding
                target_one_hot = torch.zeros(target.size(0), output.size(1), device=device)
                target_one_hot.scatter_(1, target.unsqueeze(1), 1)
                
                # Compute loss
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                if iteration + 1 == num_iterations:
                    # For the final iteration, compute L2 norms
                    softmax_outputs = torch.nn.functional.softmax(output, dim=1)
                    l2_norms = torch.norm(softmax_outputs - target_one_hot, p=2, dim=1)
                    
                    # Store L2 norms for each example
                    for i, idx in enumerate(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(dataset)))):
                        if idx < len(dataset):
                            example_l2_norms[idx].append(l2_norms[i].item())
        
    # Compute EL2N scores as the average of L2 norms across networks
    el2n_scores = {idx: np.mean(norms) for idx, norms in example_l2_norms.items()}
    
    return el2n_scores

# Example usage with the MNISTClassifier from the uploaded file
def compute_mnist_el2n_scores(mnist_dataset, num_networks=10, num_iterations=1000):
    from models.cnn_classifier import MNISTClassifier
    
    return compute_el2n_scores(
        model_class=MNISTClassifier,
        dataset=mnist_dataset,
        num_networks=num_networks,
        num_iterations=num_iterations
    )

# Function to identify examples with highest and lowest EL2N scores
def identify_examples_by_el2n(el2n_scores, dataset, num_examples=10):
    """
    Identifies examples with highest and lowest EL2N scores.
    
    Args:
        el2n_scores: Dictionary mapping example indices to their EL2N scores
        dataset: The dataset containing the examples
        num_examples: Number of examples to return in each category
        
    Returns:
        Tuple of (easiest_examples, hardest_examples) where each is a list of (idx, score, data, label)
    """
    # Sort examples by EL2N scores
    sorted_indices = sorted(el2n_scores.keys(), key=lambda idx: el2n_scores[idx])
    
    # Get examples with lowest scores (easiest to learn)
    easiest_indices = sorted_indices[:num_examples]
    easiest_examples = [(idx, el2n_scores[idx], dataset[idx][0], dataset[idx][1]) 
                        for idx in easiest_indices]
    
    # Get examples with highest scores (hardest to learn)
    hardest_indices = sorted_indices[-num_examples:]
    hardest_examples = [(idx, el2n_scores[idx], dataset[idx][0], dataset[idx][1]) 
                        for idx in hardest_indices]
    
    return easiest_examples, hardest_examples

# Complete example with MNIST dataset
def run_el2n_analysis():
    import torchvision
    import torchvision.transforms as transforms
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Compute EL2N scores
    el2n_scores = compute_mnist_el2n_scores(
        mnist_dataset=dataset,
        num_networks=5,  # Reduced for demonstration
        num_iterations= 5  # Reduced for demonstration
    )
    
    # Identify easiest and hardest examples
    easiest, hardest = identify_examples_by_el2n(el2n_scores, dataset)
    
    # Print results
    print("\nEasiest examples (lowest EL2N scores):")
    for idx, score, _, label in easiest:
        print(f"Example {idx}: Score={score:.4f}, Label={label}")
    
    print("\nHardest examples (highest EL2N scores):")
    for idx, score, _, label in hardest:
        print(f"Example {idx}: Score={score:.4f}, Label={label}")
        
    return el2n_scores, easiest, hardest

def select_samples_by_el2n(dataset, el2n_scores, selection_type='hardest', percentage=50):
    """
    Selects a subset of the dataset based on EL2N scores.
    
    Args:
        dataset: The original dataset
        el2n_scores: Dictionary mapping example indices to their EL2N scores
        selection_type: 'hardest', 'easiest', or 'mixed'
        percentage: Percentage of original dataset to select
        
    Returns:
        Subset of the dataset containing the selected examples
    """
    # Sort indices by EL2N scores
    sorted_indices = sorted(el2n_scores.keys(), key=lambda idx: el2n_scores[idx])
    num_samples = int(len(sorted_indices) * percentage / 100)
    
    if selection_type == 'hardest':
        # Select the hardest examples (highest EL2N scores)
        selected_indices = sorted_indices[-num_samples:]
    elif selection_type == 'easiest':
        # Select the easiest examples (lowest EL2N scores)
        selected_indices = sorted_indices[:num_samples]
    elif selection_type == 'mixed':
        # Select a mix of hardest and easiest examples
        half_samples = num_samples // 2
        selected_indices = sorted_indices[:half_samples] + sorted_indices[-half_samples:]
    else:
        # Random selection
        selected_indices = sorted_indices[:num_samples]
        np.random.shuffle(selected_indices)
    
    return Subset(dataset, selected_indices), selected_indices


if __name__ == "__main__":
    from utils.data_utils import set_random_seed
    import torchvision
    import torchvision.transforms as transforms
    set_random_seed(42)
    el2n_score, easiest, hardest = run_el2n_analysis()
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Select a subset of the dataset based on EL2N scores
    selected_dataset = select_samples_by_el2n(dataset, el2n_score, selection_type='hardest', percentage=50)
    
    print(f"Selected dataset size: {len(selected_dataset)}")
    
    # save the selected dataset
    torch.save(selected_dataset, 'selected_dataset.pt')
