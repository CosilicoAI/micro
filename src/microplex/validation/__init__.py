"""
Validation utilities for microplex.

This module provides tools for validating synthetic microdata against
administrative targets (IRS SOI, SNAP, etc.).
"""

from microplex.validation.soi import (
    AGI_BRACKETS,
    FILING_STATUSES,
    SOITargets,
    get_available_years as get_soi_years,
    load_soi_targets,
    compute_validation_metrics,
    ValidationResult,
    validate_against_soi,
)
from microplex.validation.baseline import (
    MetricComparison,
    BaselineComparison,
    compute_baseline_comparison,
    export_comparison_json,
)

__all__ = [
    # SOI validation
    "AGI_BRACKETS",
    "FILING_STATUSES",
    "SOITargets",
    "get_soi_years",
    "load_soi_targets",
    "compute_validation_metrics",
    "ValidationResult",
    "validate_against_soi",
    # Baseline comparison
    "MetricComparison",
    "BaselineComparison",
    "compute_baseline_comparison",
    "export_comparison_json",
]
