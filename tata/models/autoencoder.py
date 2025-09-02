"""
Backward-compatible re-export of contrastive autoencoder.
New code should import from tata.models.autoencoders directly.
"""

from tata.models.autoencoders.contrastive import (
    MLP,
    ContrastiveAutoencoder,
    ContrastiveLoss,
    TataLoss,
)

__all__ = [
    "MLP",
    "ContrastiveAutoencoder",
    "ContrastiveLoss",
    "TataLoss",
]
