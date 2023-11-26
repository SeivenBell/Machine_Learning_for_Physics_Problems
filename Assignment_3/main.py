import argparse
import torch
import torch.optim as optim
from vae_model import VariationalAutoencoder, vae_loss
from data_loader import load_even_mnist_data
from utils import save_digit_samples 

def train_model(model, train_loader, test_loader, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:  # Update here
            data = data[0].to(device)  # Data is the first element of the batch
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(data)
            loss = vae_loss(reconstructed, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Verbose mode: print training progress
        train_loss /= len(train_loader.dataset)
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')

        # Test the model after each epoch
        test_loss = test_model(model, test_loader, device)
        print(f'Test Loss: {test_loss:.4f}')

def test_model(model, test_loader, device):
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


def main(args):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Load data
    train_loader, test_loader = load_even_mnist_data(data_path=args.data_path, batch_size=args.batch_size, test_size=0.2)

    # Initialize model and optimizer
    model = VariationalAutoencoder(args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train and test the model
    train_model(model, train_loader, test_loader, optimizer, args.epochs, device)

    # Optionally save some digit samples after training
    save_digit_samples(model, args.latent_dim, args.num_samples, args.output_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE on Even MNIST Numbers")
    # Define command line arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=50, help='Dimension of the latent space')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of digit samples to generate')

    args = parser.parse_args()
    main(args)


# Command to run:
# python main.py --data_path "data/even_mnist.csv" --output_dir "results" --epochs 10 --lr 0.001 --latent_dim 50 --num_samples 100

