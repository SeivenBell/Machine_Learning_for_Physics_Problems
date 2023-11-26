import os
import matplotlib.pyplot as plt
import torch

def plot_loss(losses, output_dir):
    """
    Plot the training loss and save the plot to a file.
    
    Args:
    - losses (list): List of loss values.
    - output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/loss.pdf')
    plt.close()


import os
import matplotlib.pyplot as plt
import torch

def save_digit_samples(model, latent_dim, num_samples, output_dir, device):
    """
    Generate and save digit samples using the trained VAE model.

    Args:
    - model (VariationalAutoencoder): Trained VAE model.
    - latent_dim (int): Dimension of the latent space of the VAE.
    - num_samples (int): Number of digit samples to generate.
    - output_dir (str): Directory to save the sample images.
    - device (torch.device): Device on which the model is running.
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



# The plot_loss function remains the same
