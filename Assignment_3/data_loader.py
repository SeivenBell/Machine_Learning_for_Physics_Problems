import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_even_mnist_data(data_path, batch_size=64, test_size=0.2):
    # Load the data from the CSV file
    data = np.genfromtxt(data_path, delimiter=' ')
    print("Data loading...")
    print(data.shape)
    

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
