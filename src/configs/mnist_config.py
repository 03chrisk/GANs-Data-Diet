"""
Configuration and hyperparameters for the project.
"""
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GAN Hyperparameters
BATCH_SIZE = 128
LATENT_DIM = 128 #100     # Size of generator input noise vector
HIDDEN_DIM = 256 #64     # Size of hidden layers
IMAGE_SIZE = 28 * 28  # MNIST image dimensions flattened
LEARNING_RATE = 0.0002  # Learning rate
BETA1 = 0.5          # Adam optimizer beta1
BETA2 = 0.999        # Adam optimizer beta2
NUM_EPOCHS = 10     # Number of training epochs
SAMPLE_INTERVAL = 10  # Save images every 10 epochs
IMAGE_CHANNELS_MNIST = 1  # Number of channels in MNIST images


DATASET_TYPE = 'digits'  # 'digits' for MNIST or 'fashion' for Fashion MNIST

DATASET_CLASSES = {
    'digits': [str(i) for i in range(10)],  # 0-9 digits
    'fashion': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    'cifar10': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
}

# Paths
GENERATED_IMAGES_PATH = "./generated_images"
LOSS_PLOTS_PATH = "./loss_plots" 
MODELS_PATH = "./saved_models"
CLASSIFIER_PATH = MODELS_PATH + "/mnist_classifier"

# Random seed
RANDOM_SEED = 64