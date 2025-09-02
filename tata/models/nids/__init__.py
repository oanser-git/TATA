"""NIDS models for TATA."""

from tata.models.nids.base import AbstractNIDS
from tata.models.nids.random_forest import RandomForestNIDS
from tata.models.nids.svm import SVMNIDS
from tata.models.nids.dnn import DNNNIDS

__all__ = [
    "AbstractNIDS",
    "RandomForestNIDS",
    "SVMNIDS",
    "DNNNIDS",
]
