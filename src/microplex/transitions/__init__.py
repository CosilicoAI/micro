"""Transition models for panel synthesis.

This module provides hazard/transition models for simulating
state changes over time in panel data synthesis.

Available models:
- Mortality: Age/gender-specific death probabilities from SSA life tables
- Disability: Onset and recovery models based on SSA DI rates
- Demographic: Marriage and divorce transitions based on CPS/ACS data

Example:
    >>> from microplex.transitions import MarriageTransition, DivorceTransition
    >>> marriage = MarriageTransition()
    >>> divorce = DivorceTransition()
    >>> marriage_rates = marriage.apply(panel_data)
    >>> divorce_rates = divorce.apply(panel_data)
"""

from .mortality import Mortality
from .disability import (
    DisabilityOnset,
    DisabilityRecovery,
    DisabilityTransitionModel,
)
from .demographic import (
    MarriageTransition,
    DivorceTransition,
)

__all__ = [
    # Mortality
    "Mortality",
    # Disability transitions
    "DisabilityOnset",
    "DisabilityRecovery",
    "DisabilityTransitionModel",
    # Demographic transitions
    "MarriageTransition",
    "DivorceTransition",
]
