import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_classifier import MNISTClassifier

from utils.data_utils import (
    set_random_seed, 
    load_data, 
)

from configs.mnist_config import CLASSIFIER_PATH, BATCH_SIZE

from tqdm import tqdm

def train_classifier(subset_percentage=100, dataset_type='digits', epochs=15):
    
    set_random_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, _ = load_data(
        batch_size=BATCH_SIZE, 
        subset_percentage=subset_percentage,
        dataset_type=dataset_type
    )

    # Initialize classifier
    classifier = MNISTClassifier().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Training loop
    # Train the classifier for 5 epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/5")):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Save the trained classifier
    os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
    torch.save(classifier.state_dict(), CLASSIFIER_PATH)
    print("Classifier training complete and model saved")
    
    return classifier
    
def evaluate_classifier(classifier, dataset_type='digits'):
    """
    Evaluate the trained classifier on the test dataset.
    
    Args:
        dataset_type (str): Type of dataset ('digits' for MNIST or 'fashion' for Fashion MNIST)
        
    Returns:
        float: Accuracy of the classifier on the test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    test_loader, _ = load_data(
        batch_size=BATCH_SIZE,
        subset_percentage=100,  # Use full test set
        dataset_type=dataset_type,
        train=False  # Use test set instead of training set
    )
    
    # Set model to evaluation mode
    classifier.eval()
    
    # Tracking metrics
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    # Define class names based on dataset type
    if dataset_type.lower() == 'digits':
        class_names = [f"Digit {i}" for i in range(10)]
    else:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # No gradient calculation needed for evaluation
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = classifier(images)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    # Overall accuracy
    accuracy = 100 * correct / total
    print(f"\nOverall accuracy on the {dataset_type} test set: {accuracy:.2f}%")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(10):
        class_accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"{class_names[i]}: {class_accuracy:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MNIST Classifier')
    parser.add_argument('--subset', type=int, default=100,
                        help='Percentage of data to use (default: 100)')
    parser.add_argument('--dataset', type=str, default='digits', choices=['digits', 'fashion'],
                        help='Dataset to use: "digits" for MNIST or "fashion" for Fashion MNIST (default: digits)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 15)')
    
    args = parser.parse_args()
    
    classifier = train_classifier(subset_percentage=args.subset, dataset_type=args.dataset, epochs=args.epochs)
        
    evaluate_classifier(classifier, dataset_type=args.dataset)