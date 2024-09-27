"""
torchortho - Learnable Polynomial, Trigonometric, and Tropical Activations

Author: Ismail Khalfaoui-Hassani
Date: February 2025
License: GPL-3.0
Description: This module initializes the torchortho package and imports activation functions.

For reproducing experiments:
- Vision: https://github.com/K-H-Ismail/ConvNeXt-ortho
- Language: https://github.com/K-H-Ismail/pytorch-language-models
"""

from .hermite_activation import HermiteActivation
from .fourier_activation import FourierActivation
from .tropical_activation import TropicalActivation, TropicalRationalActivation
from .cuda import hermite_polynomials_numba

__all__ = [
    "HermiteActivation",
    "FourierActivation",
    "hermite_polynomials_numba",
    "TropicalActivation",
    "TropicalRationalActivation",
]
