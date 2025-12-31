"""Multi-survey fusion for microplex.

Combines CPS + PUF (and optionally other surveys) into a single
stacked dataset with missing value indicators for masked MAF training.
"""

from .harmonize import harmonize_surveys, stack_surveys, COMMON_SCHEMA
from .masked_maf import MaskedMAF

__all__ = [
    "harmonize_surveys",
    "stack_surveys",
    "COMMON_SCHEMA",
    "MaskedMAF",
]
