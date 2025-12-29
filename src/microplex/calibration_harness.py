"""Calibration harness for PE parity testing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from .target_registry import (
    TargetRegistry,
    TargetSpec,
    TargetCategory,
    TargetLevel,
    get_registry,
)


@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    weights: np.ndarray
    targets_used: List[str]
    errors: Dict[str, float]  # target_name -> error_pct
    iterations: int
    converged: bool
    weight_stats: Dict[str, float]

    @property
    def mean_error(self) -> float:
        return np.mean(list(self.errors.values()))

    @property
    def max_error(self) -> float:
        return max(self.errors.values()) if self.errors else 0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Calibration Result:",
            f"  Targets: {len(self.targets_used)}",
            f"  Converged: {self.converged} ({self.iterations} iterations)",
            f"  Mean error: {self.mean_error:.2f}%",
            f"  Max error: {self.max_error:.2f}%",
            f"  Weight CV: {self.weight_stats.get('cv', 0):.2f}",
        ]
        return "\n".join(lines)


class CalibrationHarness:
    """Harness for running calibration experiments with various target sets.

    This harness provides a unified interface for:
    - Selecting subsets of targets to calibrate to
    - Running IPF calibration
    - Comparing results across different target combinations
    - Tracking data availability vs. calibration accuracy
    """

    def __init__(self, registry: Optional[TargetRegistry] = None):
        """Initialize harness.

        Args:
            registry: Target registry to use. If None, uses default.
        """
        self.registry = registry or get_registry()
        self._results: Dict[str, CalibrationResult] = {}

    def get_target_vector(
        self,
        df: pd.DataFrame,
        targets: List[TargetSpec],
        weight_col: str = "weight"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build design matrix and target vector.

        Args:
            df: Microdata DataFrame
            targets: List of target specifications
            weight_col: Weight column name

        Returns:
            (design_matrix, target_vector, target_names)
        """
        n = len(df)
        design_rows = []
        target_values = []
        target_names = []

        for spec in targets:
            # Skip if column doesn't exist
            if spec.column and spec.column not in df.columns:
                continue
            if spec.filter_column and spec.filter_column not in df.columns:
                continue

            # Build indicator/value vector for this target
            # For count aggregation: row[i] = 1 if matches filter, 0 otherwise
            #   weighted sum = sum(weights * row) = weighted count of matching records
            # For sum aggregation: row[i] = column_value if matches filter
            #   weighted sum = sum(weights * row) = weighted sum of column values

            if spec.filter_column and spec.filter_value:
                # Filtered aggregation
                mask = (df[spec.filter_column] == spec.filter_value).astype(float).values
                if spec.aggregation == "count" or spec.column is None:
                    # Count: just use mask (1 if matches, 0 otherwise)
                    row = mask
                elif spec.aggregation == "sum":
                    # Sum: multiply mask by column value
                    row = mask * df[spec.column].fillna(0).values
                else:
                    row = mask
            elif spec.column:
                # Unfiltered column aggregation
                if spec.aggregation == "count":
                    # Count records where column > 0
                    row = (df[spec.column] > 0).astype(float).values
                elif spec.aggregation == "sum":
                    # Sum column values
                    row = df[spec.column].fillna(0).values
                else:
                    row = df[spec.column].fillna(0).values
            else:
                # Population count (no filter, no column = count all)
                row = np.ones(n)

            design_rows.append(row)
            target_values.append(spec.value)
            target_names.append(spec.name)

        design_matrix = np.column_stack(design_rows) if design_rows else np.zeros((n, 0))
        target_vector = np.array(target_values)

        return design_matrix, target_vector, target_names

    def calibrate(
        self,
        df: pd.DataFrame,
        targets: List[TargetSpec],
        weight_col: str = "weight",
        max_iter: int = 100,
        tol: float = 1e-6,
        bounds: Tuple[float, float] = (0.01, 100.0),
        verbose: bool = True,
    ) -> CalibrationResult:
        """Run IPF calibration.

        Args:
            df: Microdata DataFrame
            targets: Target specifications to calibrate to
            weight_col: Initial weight column
            max_iter: Maximum iterations
            tol: Convergence tolerance
            bounds: Weight adjustment bounds
            verbose: Print progress

        Returns:
            CalibrationResult
        """
        # Build matrices
        X, target_vec, names = self.get_target_vector(df, targets, weight_col)

        n_samples, n_targets = X.shape
        if verbose:
            print(f"Calibrating {n_samples:,} samples to {n_targets} targets")

        # Initialize weights
        if weight_col in df.columns:
            weights = df[weight_col].values.copy().astype(float)
        else:
            weights = np.ones(n_samples)

        # IPF iteration
        converged = False
        for iteration in range(max_iter):
            old_weights = weights.copy()

            for j in range(n_targets):
                if target_vec[j] == 0:
                    continue

                current = np.sum(weights * X[:, j])
                if current > 0:
                    factor = target_vec[j] / current
                    factor = np.clip(factor, bounds[0], bounds[1])

                    mask = X[:, j] > 0
                    weights[mask] *= factor

            # Check convergence
            max_change = np.max(np.abs(weights - old_weights) / (old_weights + 1e-10))
            if max_change < tol:
                converged = True
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

        # Compute errors
        errors = {}
        if verbose:
            print(f"\n{'Target':<40} {'Computed':>15} {'Target':>15} {'Error':>10}")
            print("-" * 85)

        for j, name in enumerate(names):
            computed = np.sum(weights * X[:, j])
            target = target_vec[j]
            if target != 0:
                error = abs(computed - target) / abs(target) * 100
            else:
                error = 0 if computed == 0 else 100

            # Cap error at 100% for reporting
            errors[name] = min(error, 100.0)

            if verbose:
                if abs(target) > 1e9:
                    comp_str = f"${computed/1e9:.1f}B"
                    tgt_str = f"${target/1e9:.1f}B"
                elif abs(target) > 1e6:
                    comp_str = f"{computed/1e6:.1f}M"
                    tgt_str = f"{target/1e6:.1f}M"
                else:
                    comp_str = f"{computed:,.0f}"
                    tgt_str = f"{target:,.0f}"
                print(f"{name:<40} {comp_str:>15} {tgt_str:>15} {error:>9.1f}%")

        # Weight stats
        weight_stats = {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "cv": float(np.std(weights) / np.mean(weights)) if np.mean(weights) > 0 else 0,
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "zero_count": int(np.sum(weights == 0)),
        }

        return CalibrationResult(
            weights=weights,
            targets_used=names,
            errors=errors,
            iterations=iteration + 1,
            converged=converged,
            weight_stats=weight_stats,
        )

    def run_experiment(
        self,
        df: pd.DataFrame,
        name: str,
        categories: Optional[List[TargetCategory]] = None,
        levels: Optional[List[TargetLevel]] = None,
        groups: Optional[List[str]] = None,
        only_available: bool = False,
        **calibrate_kwargs
    ) -> CalibrationResult:
        """Run a calibration experiment with specified target subset.

        Args:
            df: Microdata DataFrame
            name: Experiment name for tracking
            categories: Target categories to include
            levels: Target levels to include
            groups: Target group names to include
            only_available: Only use targets available in CPS
            **calibrate_kwargs: Passed to calibrate()

        Returns:
            CalibrationResult
        """
        # Select targets
        all_targets = self.registry.get_all_targets()
        selected = []

        for target in all_targets:
            # Filter by category
            if categories and target.category not in categories:
                continue

            # Filter by level
            if levels and target.level not in levels:
                continue

            # Filter by group (check if target belongs to any specified group)
            if groups:
                in_group = False
                for group_name in groups:
                    group = self.registry.get_group(group_name)
                    if group and target in group.targets:
                        in_group = True
                        break
                if not in_group:
                    continue

            # Filter by availability
            if only_available and not target.available_in_cps:
                continue

            # Check if target has a valid value
            if target.value == 0 and target.aggregation != "count":
                continue

            selected.append(target)

        print(f"\n=== Experiment: {name} ===")
        print(f"Selected {len(selected)} targets")

        result = self.calibrate(df, selected, **calibrate_kwargs)
        self._results[name] = result

        return result

    def compare_experiments(self) -> pd.DataFrame:
        """Compare results across experiments.

        Returns:
            DataFrame with comparison metrics
        """
        records = []
        for name, result in self._results.items():
            records.append({
                "experiment": name,
                "n_targets": len(result.targets_used),
                "converged": result.converged,
                "iterations": result.iterations,
                "mean_error": result.mean_error,
                "max_error": result.max_error,
                "weight_cv": result.weight_stats["cv"],
                "weight_max": result.weight_stats["max"],
                "zero_weights": result.weight_stats["zero_count"],
            })

        return pd.DataFrame(records)

    def print_target_coverage(self, df: pd.DataFrame):
        """Print which targets can be computed from the data.

        Args:
            df: Microdata DataFrame
        """
        print("=" * 70)
        print("TARGET COVERAGE ANALYSIS")
        print("=" * 70)

        all_targets = self.registry.get_all_targets()
        columns = set(df.columns)

        available = []
        missing_column = []
        needs_imputation = []

        for target in all_targets:
            if target.column and target.column not in columns:
                missing_column.append(target)
            elif target.requires_imputation:
                needs_imputation.append(target)
            else:
                available.append(target)

        print(f"\n✅ Available ({len(available)} targets):")
        for cat in TargetCategory:
            cat_targets = [t for t in available if t.category == cat]
            if cat_targets:
                print(f"  {cat.value}: {len(cat_targets)}")

        print(f"\n⚠️ Missing column ({len(missing_column)} targets):")
        missing_cols = set(t.column for t in missing_column if t.column)
        for col in sorted(missing_cols):
            count = sum(1 for t in missing_column if t.column == col)
            print(f"  {col}: {count} targets")

        print(f"\n🔧 Requires imputation ({len(needs_imputation)} targets):")
        for cat in TargetCategory:
            cat_targets = [t for t in needs_imputation if t.category == cat]
            if cat_targets:
                print(f"  {cat.value}: {len(cat_targets)}")


def run_pe_parity_suite(df: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
    """Run the full PE parity test suite.

    Args:
        df: Microdata DataFrame with income/benefit columns
        weight_col: Weight column name

    Returns:
        DataFrame with experiment comparison
    """
    harness = CalibrationHarness()

    # Print coverage
    harness.print_target_coverage(df)

    # Run experiments
    print("\n" + "=" * 70)
    print("RUNNING CALIBRATION EXPERIMENTS")
    print("=" * 70)

    # 1. State populations only
    harness.run_experiment(
        df, "states_only",
        groups=["state_population"],
        verbose=True
    )

    # 2. Income targets only (available)
    harness.run_experiment(
        df, "income_available",
        categories=[TargetCategory.INCOME],
        only_available=True,
        verbose=True
    )

    # 3. Benefits only
    harness.run_experiment(
        df, "benefits_only",
        groups=["benefit_programs"],
        verbose=True
    )

    # 4. States + Income + Benefits (available)
    harness.run_experiment(
        df, "full_available",
        groups=["state_population", "irs_soi_income", "benefit_programs"],
        only_available=True,
        verbose=True
    )

    # 5. All targets (including unavailable - will show gaps)
    harness.run_experiment(
        df, "all_targets",
        only_available=False,
        verbose=True
    )

    # Compare
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)

    comparison = harness.compare_experiments()
    print(comparison.to_string(index=False))

    return comparison
