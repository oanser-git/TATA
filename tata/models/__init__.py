"""Models package for TATA."""

from tata.models.autoencoders import (
    ContrastiveAutoencoder,
    VanillaAutoencoder,
)
from tata.models.nids import (
    AbstractNIDS,
    RandomForestNIDS,
    SVMNIDS,
    DNNNIDS,
)

__all__ = [
    "ContrastiveAutoencoder",
    "VanillaAutoencoder",
    "AbstractNIDS",
    "RandomForestNIDS",
    "SVMNIDS",
    "DNNNIDS",
]
