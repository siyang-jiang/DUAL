"""
DUAL: Dual Alignment Framework for Few-shot Learning with Inter-Set and Intra-Set Shifts

This package implements the DUAL framework for few-shot learning, addressing both
inter-set and intra-set distribution shifts in few-shot classification tasks.
"""

__version__ = "0.1.0"
__author__ = "Siyang Jiang"

# Import submodules (actual implementations will be added later)
from . import models
from . import data  
from . import utils

__all__ = ["models", "data", "utils"]