"""
Encoder wrapper for loading and inference.
Wraps the trained contrastive autoencoder encoder part.
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

import torch.nn as nn

from tata.models.autoencoder import ContrastiveAutoencoder


class Encoder:
    """
    Wrapper for the trained encoder to encode data into latent space.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        input_dim: Optional[int] = None,
        encoder_dims: Optional[list[int]] = None,
        latent_dim: Optional[int] = None,
        decoder_dims: Optional[list[int]] = None,
        device: str = "cpu",
    ) -> "Encoder":
        """
        Load encoder from a PyTorch checkpoint.
        
        If the checkpoint was saved with embedded config (recommended),
        no architecture parameters are needed.
        If the checkpoint is a raw state dict (legacy), architecture
        parameters must be provided.
        
        Args:
            checkpoint_path: Path to .pt or .pth file.
            input_dim: Required only for legacy checkpoints.
            encoder_dims: Required only for legacy checkpoints.
            latent_dim: Required only for legacy checkpoints.
            decoder_dims: Required only for legacy checkpoints.
            device: Device to load on.
        
        Returns:
            Encoder instance.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "config" in checkpoint:
            # New format: embedded config
            config = checkpoint["config"]
            model = ContrastiveAutoencoder(
                input_dim=config["input_dim"],
                encoder_dims=config["encoder_dims"],
                latent_dim=config["latent_dim"],
                decoder_dims=config["decoder_dims"],
            )
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Legacy format: raw state dict
            if any(p is None for p in [input_dim, encoder_dims, latent_dim, decoder_dims]):
                raise ValueError(
                    "This checkpoint was saved in legacy format (raw state dict). "
                    "Please provide input_dim, encoder_dims, latent_dim, and decoder_dims, "
                    "or re-save the checkpoint using Encoder.save() with embedded config."
                )
            assert input_dim is not None
            assert encoder_dims is not None
            assert latent_dim is not None
            assert decoder_dims is not None
            model = ContrastiveAutoencoder(
                input_dim=input_dim,
                encoder_dims=encoder_dims,
                latent_dim=latent_dim,
                decoder_dims=decoder_dims,
            )
            model.load_state_dict(checkpoint)
        
        model.eval()
        return cls(model, device=device)
    
    def encode(
        self,
        X: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Encode numpy array into latent space.
        
        Args:
            X: Array of shape (n_samples, input_dim).
            batch_size: Batch size for inference.
        
        Returns:
            Latent embeddings of shape (n_samples, latent_dim).
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=self.device)
                z = self.model.encode(batch)  # type: ignore[attr-defined]
                embeddings.append(z.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def save(self, path: Union[str, Path], config: Optional[dict[str, Any]] = None):
        """
        Save model checkpoint.
        
        Args:
            path: Save path.
            config: Optional model architecture config dict to embed.
                    If provided, saves as {"state_dict": ..., "config": ...}
                    for self-contained loading.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if config is not None:
            torch.save({"state_dict": self.model.state_dict(), "config": config}, path)
        else:
            torch.save(self.model.state_dict(), path)
