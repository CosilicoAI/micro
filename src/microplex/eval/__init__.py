"""
Evaluation metrics for synthetic data quality.

Includes PRDC (Precision, Recall, Density, Coverage) metrics
and imputation quality metrics.
"""

from .coverage import (
    PRDCResult,
    compute_prdc,
    compute_coverage_with_embeddings,
    compute_trajectory_coverage,
    compute_coverage_by_segment,
    evaluate_imputation_quality,
)

__all__ = [
    "PRDCResult",
    "compute_prdc",
    "compute_coverage_with_embeddings",
    "compute_trajectory_coverage",
    "compute_coverage_by_segment",
    "evaluate_imputation_quality",
]
