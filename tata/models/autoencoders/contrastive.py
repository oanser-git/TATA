"""
Contrastive Autoencoder model for TATA preliminary phase.
Encoder: MLP mapping input -> latent space.
Decoder: MLP mapping latent -> reconstruction.
"""

from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron."""
    
    def __init__(self, dims: List[int], activation: str = "relu", final_activation: bool = True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or final_activation:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastiveAutoencoder(nn.Module):
    """
    Contrastive Autoencoder for structured latent space learning.
    
    Architecture:
        Input -> Encoder -> Latent (z) -> Decoder -> Reconstruction
    
    Args:
        input_dim: Dimensionality of input features.
        encoder_dims: Hidden layer dimensions for encoder (excluding latent).
        latent_dim: Dimensionality of latent space.
        decoder_dims: Hidden layer dimensions for decoder (excluding latent).
    """
    
    def __init__(
        self,
        input_dim: int,
        encoder_dims: List[int] = [64, 32],
        latent_dim: int = 3,
        decoder_dims: List[int] = [32, 64],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> encoder_dims -> latent_dim
        encoder_layer_dims = [input_dim] + encoder_dims + [latent_dim]
        self.encoder = MLP(encoder_layer_dims, activation="relu", final_activation=False)
        
        # Decoder: latent_dim -> decoder_dims -> input_dim
        decoder_layer_dims = [latent_dim] + decoder_dims + [input_dim]
        self.decoder = MLP(decoder_layer_dims, activation="relu", final_activation=False)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to input space."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            z: Latent representation.
            x_recon: Reconstructed input.
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (Hadsell et al. style).
    
    Pulls same-label samples together, pushes different-label samples apart.
    
    L_contrastive(y, d) = (1-y) * 0.5 * d^2 + y * 0.5 * max(0, margin - d)^2
    where y=0 for positive pair, y=1 for negative pair, d=||z_i - z_j||.
    
    Args:
        margin: Minimum distance for negative pairs.
    """
    
    def __init__(self, margin: float = 10.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_i: Latent vectors (batch_size, latent_dim).
            z_j: Latent vectors (batch_size, latent_dim).
            labels: Binary labels where 0 = positive pair (same class),
                    1 = negative pair (different class).
        
        Returns:
            Scalar loss.
        """
        distances = torch.nn.functional.pairwise_distance(z_i, z_j, p=2)
        
        # Positive pairs (same label): minimize distance
        loss_positive = (1 - labels) * 0.5 * distances.pow(2)
        
        # Negative pairs (different label): push apart, but capped at margin
        loss_negative = labels * 0.5 * torch.clamp(self.margin - distances, min=0.0).pow(2)
        
        return (loss_positive + loss_negative).mean()


class TataLoss(nn.Module):
    """
    Combined loss for TATA contrastive autoencoder:
    L_total = MSE_reconstruction + lambda * L_contrastive
    """
    
    def __init__(self, margin: float = 10.0, lambda_contrastive: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.contrastive = ContrastiveLoss(margin=margin)
        self.lambda_contrastive = lambda_contrastive
    
    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        pair_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            x: Original input.
            x_recon: Reconstructed input.
            z_i, z_j: Latent vectors for pairs.
            pair_labels: Pair labels (0 = same class, 1 = different class).
        
        Returns:
            total_loss, dict with component losses.
        """
        recon_loss = self.mse(x_recon, x)
        contrastive_loss = self.contrastive(z_i, z_j, pair_labels)
        total_loss = recon_loss + self.lambda_contrastive * contrastive_loss
        
        return total_loss, {
            "total": total_loss.item(),
            "reconstruction": recon_loss.item(),
            "contrastive": contrastive_loss.item(),
        }
