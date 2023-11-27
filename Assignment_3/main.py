import argparse
import json
import torch
import torch.optim as optim
from vae_model import VariationalAutoencoder, vae_loss
from data_loader import load_even_mnist_data
from utils import save_digit_samples, plot_loss

def train_model(model, train_loader, test_loader, optimizer, epochs, device, verbose):
    """Trains the VAE model and evaluates it on a test dataset.

    Args:
        model (VariationalAutoencoder): The VAE model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        optimizer (Optimizer): Optimizer used for training.
        epochs (int): Number of epochs to train the model.
        device (torch.device): The device (CPU or GPU) to train the model on.
        verbose (bool): If True, prints detailed progress for each epoch.

    Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists:
            - Training losses for each epoch.
            - Test losses for each epoch.
    """
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data[0].to(device)  # Data is the first element of the batch
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(data)
            loss = vae_loss(reconstructed, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Verbose mode: print training progress
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Test the model after each epoch
        test_loss = test_model(model, test_loader, device)
        test_losses.append(test_loss)
        
        if verbose:
            print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            
    return train_losses, test_losses
    
def test_model(model, test_loader, device):
    """Evaluates the VAE model on the test dataset.

    Args:
        model (VariationalAutoencoder): The VAE model to be evaluated.
        test_loader (DataLoader): DataLoader for the test data.
        device (torch.device): The device (CPU or GPU) for model evaluation.

    Returns:
        float: The average loss on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # No gradients needed for evaluation
        for data in test_loader:  # Unpack only data
            data = data[0].to(device)  # Data is the first element of the batch
            reconstructed, mu, logvar = model(data)
            loss = vae_loss(reconstructed, data, mu, logvar)
            test_loss += loss.item()
    average_loss = test_loss / len(test_loader.dataset)
    return average_loss

# Rest of your main.py remains the same


def main(params, output_dir, num_samples, verbose):
    """
    Main function to run the VAE training and digit sample generation.

    Args:
        params (dict): Configuration parameters for the model and training.
            Includes data_path, epochs, lr, latent_dim, batch_size, test_size.
        output_dir (str): Directory where the output files will be saved.
        num_samples (int): Number of digit samples to generate.
        verbose (bool): Flag to enable verbose output during training.

    This function loads the data, initializes the VAE model, trains it, plots losses,
    and saves generated digit samples.
    """
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
        # Load data
    train_loader, test_loader = load_even_mnist_data(
        data_path=params["data_path"], 
        batch_size=params["batch_size"], 
        test_size=params["test_size"]
    )


    # Initialize model and optimizer
    model = VariationalAutoencoder(params["latent_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # # Train and test the model
    # train_model(model, train_loader, test_loader, optimizer, args.epochs, device)
    # Train and test the model, capturing the loss history
    train_losses, test_losses = train_model(model, train_loader, test_loader, optimizer, params["epochs"], device, verbose)

    plot_loss(train_losses, test_losses, output_dir)
    
    save_digit_samples(model, params["latent_dim"], num_samples, output_dir, device)


if __name__ == "__main__":
    
    """
    Executes the main functionality of the script when run as a standalone program.

    This block parses command-line arguments, loads configuration parameters from a JSON file,
    and then calls the main function with these parameters to train the VAE model and generate digit samples.

    The script accepts command-line arguments to specify the output directory, the number of digit samples to generate,
    and whether to enable verbose output. Configuration parameters such as the path to the dataset, number of training epochs,
    learning rate, latent dimension size, batch size, and test data size are read from 'params.json'.
    """
    parser = argparse.ArgumentParser(description="Train a VAE on Even MNIST Numbers")
    
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('-n', '--num_samples', type=int, required=True, help='Number of digit samples to generate')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    

    args = parser.parse_args()

    # Load configuration from JSON file
    with open('params.json', 'r') as f:
        params = json.load(f)

    main(params, args.output_dir, args.num_samples, args.verbose)



