"""
Proper distributional quality metrics for evaluating synthetic data generation.

These metrics test whether synthetic data:
1. Captures the full conditional distribution (not just the mode)
2. Has properly calibrated uncertainty
3. Resembles real out-of-sample records

Current benchmarks only use KS tests and correlation error, which don't
capture distributional quality. These new metrics address that gap.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Compute pinball loss (quantile loss) for a specific quantile.

    Pinball loss is asymmetric: it penalizes over-prediction and under-prediction
    differently based on the target quantile. This measures how well predicted
    quantiles match true values.

    Args:
        y_true: True values (shape: [n_samples])
        y_pred: Predicted values at given quantile (shape: [n_samples])
        quantile: Target quantile in [0, 1]

    Returns:
        Mean pinball loss (lower is better)

    Examples:
        For quantile=0.5 (median):
            - If y_true=10, y_pred=8: loss = 0.5 * (10-8) = 1.0
            - If y_true=10, y_pred=12: loss = 0.5 * (12-10) = 1.0
        For quantile=0.9:
            - If y_true=10, y_pred=8: loss = 0.9 * (10-8) = 1.8  (penalizes under-prediction more)
            - If y_true=10, y_pred=12: loss = 0.1 * (12-10) = 0.2
    """
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def compute_quantile_losses(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_vars: List[str],
    quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
) -> Dict[str, Dict[str, float]]:
    """
    Compute pinball losses for multiple quantiles.

    For each target variable, compare the synthetic distribution quantiles
    against the real data quantiles. This tests whether the synthetic data
    captures the full conditional distribution.

    Args:
        real_data: Real/holdout data
        synthetic_data: Synthetic data (same size as real_data)
        target_vars: Variables to evaluate
        quantiles: Quantiles to test

    Returns:
        Dict mapping variable -> {quantile -> loss}

    Note:
        Lower loss = better match. Perfect match would have loss near 0.
    """
    results = {}

    for var in target_vars:
        var_results = {}
        real_values = real_data[var].values
        synth_values = synthetic_data[var].values

        for q in quantiles:
            # Compute empirical quantile from synthetic data
            synth_quantile = np.quantile(synth_values, q)

            # Measure pinball loss: how well does this quantile match real data?
            loss = pinball_loss(real_values, np.full_like(real_values, synth_quantile), q)
            var_results[q] = loss

        results[var] = var_results

    return results


def compute_crps(
    real_values: np.ndarray,
    synthetic_samples: np.ndarray,
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS is a proper scoring rule for probabilistic forecasts. It measures
    the integral of squared differences between the predicted CDF and the
    empirical CDF (step function at the true value).

    For a single true value y and predicted distribution F:
        CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent samples from F.

    Args:
        real_values: True values (shape: [n_samples])
        synthetic_samples: Multiple samples from predicted distribution
                          (shape: [n_samples, n_synthetic_per_sample])

    Returns:
        Mean CRPS across all samples (lower is better)

    Note:
        CRPS = 0 for perfect predictions, > 0 otherwise.
        CRPS is equivalent to the mean absolute error for deterministic forecasts.
    """
    n_samples = len(real_values)
    n_synthetic = synthetic_samples.shape[1]

    crps_scores = []

    for i in range(n_samples):
        y = real_values[i]
        samples = synthetic_samples[i, :]

        # E[|X - y|]
        term1 = np.mean(np.abs(samples - y))

        # E[|X - X'|] - compare all pairs
        term2 = 0.0
        for j in range(n_synthetic):
            for k in range(j + 1, n_synthetic):
                term2 += np.abs(samples[j] - samples[k])
        term2 = term2 / (n_synthetic * (n_synthetic - 1) / 2)

        crps = term1 - 0.5 * term2
        crps_scores.append(crps)

    return np.mean(crps_scores)


def compute_prediction_interval_coverage(
    real_values: np.ndarray,
    synthetic_samples: np.ndarray,
    coverage_levels: List[float] = [0.5, 0.8, 0.9],
) -> Dict[float, Dict[str, float]]:
    """
    Compute prediction interval coverage and calibration.

    For each coverage level (e.g., 90%), generate prediction intervals from
    the synthetic samples and check if the true values fall within them.

    A well-calibrated model should have:
        - 90% interval contains 90% of true values
        - 50% interval contains 50% of true values

    Args:
        real_values: True values (shape: [n_samples])
        synthetic_samples: Multiple samples from predicted distribution
                          (shape: [n_samples, n_synthetic_per_sample])
        coverage_levels: Target coverage probabilities

    Returns:
        Dict mapping coverage_level -> {
            'target': target coverage,
            'actual': actual coverage,
            'calibration_error': |target - actual|,
            'interval_width': mean interval width
        }
    """
    results = {}

    n_samples = len(real_values)

    for level in coverage_levels:
        # Compute prediction intervals
        alpha = 1 - level
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        lower_bounds = np.quantile(synthetic_samples, lower_q, axis=1)
        upper_bounds = np.quantile(synthetic_samples, upper_q, axis=1)

        # Check coverage
        in_interval = (real_values >= lower_bounds) & (real_values <= upper_bounds)
        actual_coverage = np.mean(in_interval)

        # Calibration error
        calibration_error = np.abs(level - actual_coverage)

        # Interval width
        interval_widths = upper_bounds - lower_bounds
        mean_width = np.mean(interval_widths)

        results[level] = {
            'target': level,
            'actual': actual_coverage,
            'calibration_error': calibration_error,
            'interval_width': mean_width,
        }

    return results


def compute_variance_ratio(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_vars: List[str],
) -> Dict[str, float]:
    """
    Compute ratio of synthetic variance to real variance.

    Tests if synthetic data has appropriate spread. Common failure modes:
    - Ratio < 1: Synthetic data is under-dispersed (mode collapse)
    - Ratio > 1: Synthetic data is over-dispersed
    - Ratio ≈ 1: Good variance matching

    Args:
        real_data: Real/holdout data
        synthetic_data: Synthetic data
        target_vars: Variables to evaluate

    Returns:
        Dict mapping variable -> variance_ratio
    """
    results = {}

    for var in target_vars:
        real_var = np.var(real_data[var])
        synth_var = np.var(synthetic_data[var])

        if real_var > 0:
            ratio = synth_var / real_var
        else:
            ratio = 1.0 if synth_var == 0 else np.inf

        results[var] = ratio

    return results


def compute_conditional_variance_check(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_vars: List[str],
    condition_vars: List[str],
    n_bins: int = 3,
) -> Dict[str, float]:
    """
    Check if within-group variance is preserved in demographic subgroups.

    This tests if the model captures heteroscedasticity (variance that changes
    with demographics). For example:
    - Young people may have more variable income than old people
    - High earners may have more variable assets

    Args:
        real_data: Real/holdout data
        synthetic_data: Synthetic data
        target_vars: Target variables to check
        condition_vars: Demographic variables defining groups
        n_bins: Number of bins per condition variable

    Returns:
        Dict mapping variable -> mean_variance_ratio_error across groups
    """
    results = {}

    # Create demographic bins
    binned_real = real_data[condition_vars].copy()
    binned_synth = synthetic_data[condition_vars].copy()

    for var in condition_vars:
        if real_data[var].nunique() <= n_bins:
            binned_real[var] = real_data[var]
            binned_synth[var] = synthetic_data[var]
        else:
            # Bin continuous variables
            bins = pd.qcut(real_data[var], q=n_bins, labels=False, duplicates='drop')
            binned_real[var] = bins
            # Use same bin edges for synthetic
            binned_synth[var] = pd.cut(
                synthetic_data[var],
                bins=pd.qcut(real_data[var], q=n_bins, duplicates='drop', retbins=True)[1],
                labels=False,
                include_lowest=True,
            )

    for target_var in target_vars:
        variance_errors = []

        # Group by demographics
        for group_vals, real_group in binned_real.groupby(list(condition_vars)):
            if len(real_group) < 5:
                continue

            # Get corresponding synthetic group
            mask = np.ones(len(binned_synth), dtype=bool)
            for i, cond_var in enumerate(condition_vars):
                mask &= binned_synth[cond_var] == group_vals[i]

            if mask.sum() < 5:
                continue

            synth_group = synthetic_data[mask]

            # Compute variances
            real_var = np.var(real_data.loc[real_group.index, target_var])
            synth_var = np.var(synth_group[target_var])

            if real_var > 0:
                variance_ratio = synth_var / real_var
                variance_errors.append(np.abs(1.0 - variance_ratio))

        if variance_errors:
            results[target_var] = np.mean(variance_errors)
        else:
            results[target_var] = 0.0

    return results


def generate_conditional_samples(
    model,
    conditions: pd.DataFrame,
    n_samples_per_condition: int = 100,
) -> np.ndarray:
    """
    Generate multiple synthetic samples for each conditioning context.

    This is needed for CRPS and prediction interval metrics. For each
    row in conditions, generate n_samples_per_condition synthetic samples.

    Args:
        model: Fitted model with .generate() method
        conditions: Conditioning variables (shape: [n_contexts, n_conditions])
        n_samples_per_condition: Number of synthetic samples per context

    Returns:
        Samples array (shape: [n_contexts, n_samples_per_condition, n_targets])
    """
    n_contexts = len(conditions)

    # Repeat conditions n_samples_per_condition times
    repeated_conditions = pd.concat(
        [conditions] * n_samples_per_condition,
        ignore_index=True
    )

    # Generate all samples at once
    all_samples = model.generate(repeated_conditions)

    # Reshape to [n_contexts, n_samples_per_condition, n_targets]
    target_vars = [col for col in all_samples.columns if col not in conditions.columns]

    samples = np.zeros((n_contexts, n_samples_per_condition, len(target_vars)))

    for i in range(n_contexts):
        idx_start = i * n_samples_per_condition
        idx_end = (i + 1) * n_samples_per_condition
        samples[i, :, :] = all_samples.iloc[idx_start:idx_end][target_vars].values

    return samples


def compute_comprehensive_distributional_metrics(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_vars: List[str],
    condition_vars: List[str],
    n_samples_per_condition: int = 50,
) -> Dict[str, any]:
    """
    Compute all distributional quality metrics.

    This is the main function to call for comprehensive evaluation.

    Args:
        model: Fitted model with .generate() method
        train_data: Training data (for reference distributions)
        test_data: Holdout test data (ground truth)
        target_vars: Target variables
        condition_vars: Conditioning variables
        n_samples_per_condition: Samples per context for CRPS/intervals

    Returns:
        Dict with all metrics:
            - quantile_losses: Pinball losses per variable per quantile
            - crps: CRPS per variable
            - interval_coverage: Calibration per variable per coverage level
            - variance_ratios: Variance ratios per variable
            - conditional_variance_errors: Within-group variance errors
    """
    results = {}

    print("Generating single synthetic dataset for basic metrics...")
    test_conditions = test_data[condition_vars]
    synthetic_single = model.generate(test_conditions)

    print("Computing quantile losses...")
    results['quantile_losses'] = compute_quantile_losses(
        test_data, synthetic_single, target_vars
    )

    print("Computing variance ratios...")
    results['variance_ratios'] = compute_variance_ratio(
        test_data, synthetic_single, target_vars
    )

    print("Computing conditional variance checks...")
    results['conditional_variance_errors'] = compute_conditional_variance_check(
        test_data, synthetic_single, target_vars, condition_vars
    )

    print(f"Generating {n_samples_per_condition} samples per condition for CRPS/intervals...")
    samples = generate_conditional_samples(
        model, test_conditions, n_samples_per_condition
    )

    print("Computing CRPS...")
    crps_results = {}
    for i, var in enumerate(target_vars):
        real_values = test_data[var].values
        var_samples = samples[:, :, i]
        crps_results[var] = compute_crps(real_values, var_samples)
    results['crps'] = crps_results

    print("Computing prediction interval coverage...")
    interval_results = {}
    for i, var in enumerate(target_vars):
        real_values = test_data[var].values
        var_samples = samples[:, :, i]
        interval_results[var] = compute_prediction_interval_coverage(real_values, var_samples)
    results['interval_coverage'] = interval_results

    return results


def print_distributional_metrics_report(
    metrics: Dict[str, any],
    target_vars: List[str],
):
    """
    Print a formatted report of distributional metrics.

    Args:
        metrics: Output from compute_comprehensive_distributional_metrics
        target_vars: Target variable names
    """
    print("\n" + "="*80)
    print("DISTRIBUTIONAL QUALITY METRICS")
    print("="*80)

    # 1. Quantile Losses
    print("\n1. QUANTILE LOSSES (Pinball Loss)")
    print("   Measures how well quantiles are preserved (lower is better)")
    print("-" * 80)

    quantile_losses = metrics['quantile_losses']
    for var in target_vars:
        print(f"\n{var}:")
        var_losses = quantile_losses[var]
        for q, loss in var_losses.items():
            print(f"  q={q:.2f}: {loss:.4f}")
        mean_loss = np.mean(list(var_losses.values()))
        print(f"  Mean: {mean_loss:.4f}")

    # 2. CRPS
    print("\n2. CONTINUOUS RANKED PROBABILITY SCORE (CRPS)")
    print("   Proper scoring rule for distributional forecasts (lower is better)")
    print("-" * 80)

    crps = metrics['crps']
    for var in target_vars:
        print(f"{var}: {crps[var]:.4f}")
    print(f"Mean CRPS: {np.mean(list(crps.values())):.4f}")

    # 3. Prediction Intervals
    print("\n3. PREDICTION INTERVAL CALIBRATION")
    print("   Coverage should match target (calibration_error near 0 is best)")
    print("-" * 80)

    intervals = metrics['interval_coverage']
    for var in target_vars:
        print(f"\n{var}:")
        var_intervals = intervals[var]
        for level, stats in var_intervals.items():
            print(f"  {level*100:.0f}% interval:")
            print(f"    Target coverage: {stats['target']:.1%}")
            print(f"    Actual coverage: {stats['actual']:.1%}")
            print(f"    Calibration error: {stats['calibration_error']:.4f}")
            print(f"    Mean width: {stats['interval_width']:.2f}")

    # 4. Variance Ratios
    print("\n4. VARIANCE RATIOS (Synthetic / Real)")
    print("   Should be close to 1.0 (under-dispersed if < 1, over-dispersed if > 1)")
    print("-" * 80)

    var_ratios = metrics['variance_ratios']
    for var in target_vars:
        ratio = var_ratios[var]
        status = "✓" if 0.8 <= ratio <= 1.2 else "✗"
        print(f"{var}: {ratio:.3f} {status}")

    # 5. Conditional Variance
    print("\n5. CONDITIONAL VARIANCE PRESERVATION")
    print("   Within-group variance error (lower is better)")
    print("-" * 80)

    cond_var = metrics['conditional_variance_errors']
    for var in target_vars:
        error = cond_var[var]
        print(f"{var}: {error:.4f}")
    print(f"Mean error: {np.mean(list(cond_var.values())):.4f}")

    print("\n" + "="*80)
    print("END REPORT")
    print("="*80 + "\n")
