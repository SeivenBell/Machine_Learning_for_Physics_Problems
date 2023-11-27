import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_even_mnist_data(data_path, batch_size, test_size):
    """
    Loads and processes the even MNIST dataset from a CSV file.

    This function reads the MNIST dataset (containing only even digits),
    preprocesses the images, and splits the data into training and test sets.
    The data is then converted into PyTorch DataLoader objects for easy batch processing
    during model training.

    Args:
        data_path (str): The path to the CSV file containing the dataset.
        batch_size (int, optional): The number of samples per batch to load. Defaults to 64.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and test DataLoaders.

    """
    
    # Load the data from the CSV file
    data = np.genfromtxt(data_path, delimiter=' ')
    print("Data loading...")
    print(f"Initial Data Shape: {data.shape}")
    

    # Extract and preprocess images
    images = data[:, :-1]  # All columns except the last one
    num_samples, _ = images.shape
    images = images.reshape((num_samples, 1, 14, 14))
    print(f"Images Shape: {images.shape}")
    images = images / 255.0  # Normalize between 0-1

    # Convert to PyTorch tensors
    images = torch.from_numpy(images).float()
    print(f"Images Shape: {images.shape}")

    # Split into training and test sets
    num_train = int((1 - test_size) * num_samples)
    train_images, test_images = images[:num_train], images[num_train:]
    print("Train Images Shape: ", train_images.shape)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images)
    test_dataset = TensorDataset(test_images)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Data loading done.")

    return train_loader, test_loader
