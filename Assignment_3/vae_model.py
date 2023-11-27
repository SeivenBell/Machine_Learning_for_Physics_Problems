import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encoder module of a Variational Autoencoder.

    This encoder uses convolutional layers to compress input data into a latent space representation.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        fc_mu (nn.Linear): Fully connected layer to produce mean (mu) for the latent space.
        fc_logvar (nn.Linear): Fully connected layer to produce log variance (logvar) for the latent space.

    Args:
        latent_dim (int): Size of the latent space.
    """
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)  # Output: 8 x 7 x 7
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # Output: 16 x 4 x 4
        self.fc_mu = nn.Linear(16 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(16 * 4 * 4, latent_dim)

    def forward(self, x):
        #print(f"Encoder Input Shape: {x.shape}")
        x = F.relu(self.conv1(x))
        #print(f"Shape after Conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        #print(f"Shape after Conv2: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten
        #print(f"Shape after Flatten: {x.shape}")
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        #print(f"Mu Shape: {mu.shape}, Logvar Shape: {logvar.shape}")
        return mu, logvar

class Decoder(nn.Module):
    """
    Decoder module of a Variational Autoencoder.

    This decoder uses transposed convolutional layers to reconstruct data from the latent space representation.

    Attributes:
        fc (nn.Linear): Fully connected layer for initial transformation from the latent space.
        conv_trans1 (nn.ConvTranspose2d): First transposed convolutional layer.
        conv_trans2 (nn.ConvTranspose2d): Second transposed convolutional layer.

    Args:
        latent_dim (int): Size of the latent space.
    """
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 16 * 4 * 4)
        self.conv_trans1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.conv_trans2 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)#, output_padding=0)

    def forward(self, z):
        #print(f"Decoder Input Shape: {z.shape}")
        z = self.fc(z)
        z = z.view(z.size(0), 16, 4, 4)  # Unflatten
        #print(f"Shape after Unflatten: {z.shape}")
        z = F.relu(self.conv_trans1(z))
        #print(f"Shape after ConvTrans1: {z.shape}")
        z = torch.sigmoid(self.conv_trans2(z))  # Output between 0 and 1
        #print(f"Shape after ConvTrans2: {z.shape}")
        return z

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) combining the Encoder and Decoder modules.

    This class defines the overall VAE architecture including the encoder, decoder, and the reparameterization step.

    Attributes:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.

    Args:
        latent_dim (int): Size of the latent space.
    """
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        #print(f"Latent Vector Shape: {z.shape}")
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

def vae_loss(reconstructed_x, x, mu, logvar):
    """
    Computes the loss for the VAE.

    The loss function combines reconstruction loss and Kullback-Leibler divergence.

    Args:
        reconstructed_x (torch.Tensor): The reconstructed data.
        x (torch.Tensor): The original input data.
        mu (torch.Tensor): The mean from the latent space representation.
        logvar (torch.Tensor): The log variance from the latent space representation.

    Returns:
        torch.Tensor: The computed loss for the VAE.
    """
    recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    #print(torch.sum(x))
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
