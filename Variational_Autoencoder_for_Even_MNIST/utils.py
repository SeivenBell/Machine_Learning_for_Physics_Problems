import os
import matplotlib.pyplot as plt
import torch

def plot_loss(train_losses, test_losses, output_dir):
    """
    Plots the training and test losses over epochs and saves the plot as a PDF.

    This function takes the history of training and test losses and plots them on
    the same graph to provide a visual representation of the model's learning progress
    over time. The plot is saved as a PDF file in the specified output directory.

    Args:
        train_losses (List[float]): A list of float values representing the training loss at each epoch.
        test_losses (List[float]): A list of float values representing the test loss at each epoch.
        output_dir (str): The directory where the plot will be saved.

    The function checks if the output directory exists and creates it if not. It then
    creates a plot with the specified losses and saves it as 'loss.pdf' in the output directory.
    """
        # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/loss.pdf')
    plt.close()

def save_digit_samples(model, latent_dim, num_samples, output_dir, device):
    """
    Generates and saves images of digits sampled from the trained VAE model.

    This function samples points from the latent space of the Variational Autoencoder (VAE),
    decodes them to generate digit images, and then saves these images as PDF files in the specified output directory.

    Args:
        model (VariationalAutoencoder): The trained VAE model used for generating digit images.
        latent_dim (int): The dimension of the latent space in the VAE.
        num_samples (int): The number of digit images to generate and save.
        output_dir (str): The directory where the generated images will be saved.
        device (torch.device): The device on which the model and tensors are allocated.

    This function iterates `num_samples` times, each time sampling a point from the latent space,
    generating a digit image using the VAE's decoder, and saving the image to `output_dir`.
    The function ensures that the directory for saving images exists, creating it if necessary.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No need to track gradients
        for i in range(num_samples):
            # Sample from the latent space and generate an image
            z = torch.randn(1, latent_dim).to(device)  # Ensure z is on the same device as the model
            reconstructed_img = model.decoder(z).cpu()  # Move the generated image to CPU for plotting

            # Reshape the generated image and save it
            img = reconstructed_img.view(14, 14).numpy()  # Assuming the output is 14x14
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(f'{output_dir}/{i+1}.pdf', bbox_inches='tight')
            plt.close()
        print(f"Samples saved to {output_dir}")



# The plot_loss function remains the same
