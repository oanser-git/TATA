"""Autoencoder models for TATA."""

from tata.models.autoencoders.contrastive import (
    ContrastiveAutoencoder,
    ContrastiveLoss,
    MLP,
    TataLoss,
)
from tata.models.autoencoders.vanilla import VanillaAutoencoder

__all__ = [
    "MLP",
    "ContrastiveAutoencoder",
    "ContrastiveLoss",
    "TataLoss",
    "VanillaAutoencoder",
]
