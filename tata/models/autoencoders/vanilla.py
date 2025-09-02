"""
Vanilla Autoencoder model for TATA ablation (baseline comparison).
No contrastive loss -- only MSE reconstruction.
"""

from typing import List

import torch
import torch.nn as nn

from tata.models.autoencoders.contrastive import MLP


class VanillaAutoencoder(nn.Module):
    """
    Vanilla Autoencoder for baseline comparison in ablation study.
    
    Architecture:
        Input -> Encoder -> Latent (z) -> Decoder -> Reconstruction
    
    Uses same MLP encoder/decoder structure as ContrastiveAutoencoder,
    but trains with MSE only (no contrastive pairs).
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
        
        encoder_layer_dims = [input_dim] + encoder_dims + [latent_dim]
        self.encoder = MLP(encoder_layer_dims, activation="relu", final_activation=False)
        
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


class VanillaLoss(nn.Module):
    """Simple MSE loss for vanilla autoencoder."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            x: Original input.
            x_recon: Reconstructed input.
        
        Returns:
            loss, dict with component losses.
        """
        loss = self.mse(x_recon, x)
        return loss, {
            "total": loss.item(),
            "reconstruction": loss.item(),
            "contrastive": 0.0,
        }
