"""
Multivariate realism metrics for synthetic data evaluation.

These metrics go beyond univariate distributions to assess whether synthetic
records are realistic in the full joint space. This is critical for methods
that synthesize multiple variables jointly (like microplex).

Key insight: A model can have perfect marginal distributions but still produce
unrealistic records (e.g., 80-year-olds with student debt, billionaires on food stamps).

Metrics:
1. Synthetic→Holdout Distance (Authenticity): How close are synthetic records to real data?
2. Holdout→Synthetic Distance (Coverage): Do we cover the full data manifold?
3. Distance Ratio (Privacy/Overfitting Check): Are we too close to training data?
4. Maximum Mean Discrepancy (MMD): Kernel-based multivariate distribution test
5. Energy Distance: Another multivariate two-sample test
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def normalize_data(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    variables: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Normalize all datasets using training data statistics.

    Critical for distance calculations since variables are on different scales
    (age in [18, 90], income in [0, 1e6], etc.).

    Args:
        train: Training data
        holdout: Holdout/test data
        synthetic: Synthetic data
        variables: Variables to normalize

    Returns:
        Tuple of (train_normalized, holdout_normalized, synthetic_normalized, scaler)
    """
    scaler = StandardScaler()

    train_norm = scaler.fit_transform(train[variables])
    holdout_norm = scaler.transform(holdout[variables])
    synthetic_norm = scaler.transform(synthetic[variables])

    return train_norm, holdout_norm, synthetic_norm, scaler


def compute_nearest_neighbor_distances(
    source: np.ndarray,
    target: np.ndarray,
    metric: str = 'euclidean',
    n_jobs: int = -1,
) -> np.ndarray:
    """
    For each point in source, find distance to nearest point in target.

    Args:
        source: Source points (shape: [n_source, n_features])
        target: Target points (shape: [n_target, n_features])
        metric: Distance metric (euclidean, manhattan, etc.)
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        Array of nearest neighbor distances (shape: [n_source])
    """
    # Compute pairwise distances efficiently
    distances = cdist(source, target, metric=metric)

    # Find minimum distance for each source point
    min_distances = np.min(distances, axis=1)

    return min_distances


def compute_authenticity_distance(
    synthetic: np.ndarray,
    holdout: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Synthetic → Holdout distance (Authenticity metric).

    For each synthetic record, find the distance to its nearest real record
    in the holdout set. Lower distance = more realistic synthetic records.

    But: if TOO low, might indicate privacy issues or memorization.

    Args:
        synthetic: Normalized synthetic data
        holdout: Normalized holdout/test data

    Returns:
        Dict with statistics about the distances:
            - mean: Average nearest-neighbor distance
            - median: Median distance
            - std: Standard deviation of distances
            - min: Minimum distance (privacy concern if very small)
            - max: Maximum distance (outliers in synthetic data)
            - q25, q75: Quartiles
    """
    distances = compute_nearest_neighbor_distances(synthetic, holdout)

    return {
        'mean': float(np.mean(distances)),
        'median': float(np.median(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'q25': float(np.percentile(distances, 25)),
        'q75': float(np.percentile(distances, 75)),
    }


def compute_coverage_distance(
    holdout: np.ndarray,
    synthetic: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Holdout → Synthetic distance (Coverage metric).

    For each real holdout record, find the distance to its nearest synthetic record.
    Lower distance = better coverage of the real data manifold.

    High distances indicate regions of the data space that synthetic data doesn't cover.

    Args:
        holdout: Normalized holdout/test data
        synthetic: Normalized synthetic data

    Returns:
        Dict with statistics about the distances
    """
    distances = compute_nearest_neighbor_distances(holdout, synthetic)

    return {
        'mean': float(np.mean(distances)),
        'median': float(np.median(distances)),
        'std': float(np.std(distances)),
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'q25': float(np.percentile(distances, 25)),
        'q75': float(np.percentile(distances, 75)),
    }


def compute_privacy_distance_ratio(
    synthetic: np.ndarray,
    train: np.ndarray,
    holdout: np.ndarray,
) -> Dict[str, float]:
    """
    Compute distance ratio to check for overfitting/memorization.

    For each synthetic record, compare:
    - Distance to nearest training record
    - Distance to nearest holdout record

    If synthetic records are systematically closer to training than to holdout,
    this indicates overfitting/memorization rather than generalization.

    Ratio > 1: Synthetic closer to holdout than training (good - generalizes)
    Ratio ≈ 1: Equal distance (ideal)
    Ratio < 1: Synthetic closer to training than holdout (overfitting risk)

    Args:
        synthetic: Normalized synthetic data
        train: Normalized training data
        holdout: Normalized holdout/test data

    Returns:
        Dict with ratio statistics
    """
    dist_to_train = compute_nearest_neighbor_distances(synthetic, train)
    dist_to_holdout = compute_nearest_neighbor_distances(synthetic, holdout)

    # Compute ratio (add small epsilon to avoid division by zero)
    ratios = (dist_to_holdout + 1e-10) / (dist_to_train + 1e-10)

    # Also compute how many points are closer to training than holdout
    closer_to_train = np.mean(dist_to_train < dist_to_holdout)

    return {
        'mean_ratio': float(np.mean(ratios)),
        'median_ratio': float(np.median(ratios)),
        'fraction_closer_to_train': float(closer_to_train),
        'mean_dist_to_train': float(np.mean(dist_to_train)),
        'mean_dist_to_holdout': float(np.mean(dist_to_holdout)),
    }


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel between X and Y.

    K(x, y) = exp(-gamma * ||x - y||^2)

    Args:
        X: First dataset (shape: [n_x, n_features])
        Y: Second dataset (shape: [n_y, n_features])
        gamma: Kernel bandwidth (default: 1 / n_features)

    Returns:
        Kernel matrix (shape: [n_x, n_y])
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Compute squared Euclidean distances
    dists_sq = cdist(X, Y, metric='sqeuclidean')

    # Apply RBF kernel
    return np.exp(-gamma * dists_sq)


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = 'rbf',
    gamma: float = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    MMD is a kernel-based two-sample test. It measures the distance between
    the mean embeddings of two distributions in a reproducing kernel Hilbert space.

    MMD^2 = E[k(X, X')] - 2*E[k(X, Y)] + E[k(Y, Y')]

    where k is a kernel function (e.g., RBF).

    MMD = 0 iff distributions are identical.
    Higher MMD = more different distributions.

    Args:
        X: First sample (shape: [n_x, n_features])
        Y: Second sample (shape: [n_y, n_features])
        kernel: Kernel type ('rbf' only for now)
        gamma: RBF kernel bandwidth (default: 1 / n_features)

    Returns:
        MMD statistic (always >= 0)
    """
    if kernel != 'rbf':
        raise NotImplementedError("Only RBF kernel is currently supported")

    n_x = len(X)
    n_y = len(Y)

    # Compute kernel matrices
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    # Compute MMD^2 (unbiased estimator)
    # E[k(X, X')] - diagonal terms don't count (they're always 1)
    term1 = (np.sum(K_XX) - np.trace(K_XX)) / (n_x * (n_x - 1))

    # E[k(Y, Y')]
    term2 = (np.sum(K_YY) - np.trace(K_YY)) / (n_y * (n_y - 1))

    # E[k(X, Y)]
    term3 = np.sum(K_XY) / (n_x * n_y)

    mmd_squared = term1 + term2 - 2 * term3

    # Return MMD (take sqrt, but ensure non-negative due to numerical issues)
    return float(np.sqrt(max(0, mmd_squared)))


def compute_energy_distance(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """
    Compute energy distance between two distributions.

    Energy distance is the MULTIVARIATE GENERALIZATION OF CRPS.

    For univariate case (d=1), Energy Score = 2 * CRPS.

    D(X, Y) = 2*E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

    where X, X' are i.i.d. from distribution P and Y, Y' are i.i.d. from Q.

    Properties:
    - D = 0 iff P = Q (distributions are identical)
    - Based on Euclidean distance (no kernel choice needed)
    - Interpretable: measures expected distance between samples
    - Proper scoring rule: minimized when forecast = true distribution

    Args:
        X: First sample (shape: [n_x, n_features])
        Y: Second sample (shape: [n_y, n_features])

    Returns:
        Energy distance statistic
    """
    n_x = len(X)
    n_y = len(Y)

    # Compute pairwise distances
    dist_XX = cdist(X, X, metric='euclidean')
    dist_YY = cdist(Y, Y, metric='euclidean')
    dist_XY = cdist(X, Y, metric='euclidean')

    # E[||X - Y||]
    term1 = 2 * np.mean(dist_XY)

    # E[||X - X'||] - exclude diagonal (i = i')
    term2 = (np.sum(dist_XX) - np.trace(dist_XX)) / (n_x * (n_x - 1))

    # E[||Y - Y'||]
    term3 = (np.sum(dist_YY) - np.trace(dist_YY)) / (n_y * (n_y - 1))

    energy_dist = term1 - term2 - term3

    return float(energy_dist)


def compute_variogram_score(
    X: np.ndarray,
    Y: np.ndarray,
    p: float = 0.5,
) -> float:
    """
    Compute Variogram Score for multivariate distributional accuracy.

    The Variogram Score focuses specifically on PAIRWISE CORRELATIONS
    between variables. It measures whether the model captures the
    joint structure, not just marginals.

    VS = Σᵢⱼ (|yᵢ - yⱼ|^p - E|Xᵢ - Xⱼ|^p)²

    where:
    - y is a single observation (holdout record)
    - X are samples from the forecast (synthetic records)
    - i, j index VARIABLES (not records)
    - p is typically 0.5 (default) or 1

    Key insight: If the model captures marginals perfectly but breaks
    correlations, the variogram score will be high.

    Args:
        X: Synthetic samples (shape: [n_samples, n_features])
        Y: Holdout observations (shape: [n_obs, n_features])
        p: Power parameter (default 0.5, can also use 1)

    Returns:
        Variogram score (lower = better correlation structure)
    """
    n_features = X.shape[1]
    n_samples = X.shape[0]
    n_obs = Y.shape[0]

    # For each pair of variables, compute expected pairwise distance
    # from synthetic samples
    synthetic_pairwise = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # E[|Xᵢ - Xⱼ|^p] over synthetic samples
            diffs = np.abs(X[:, i] - X[:, j]) ** p
            synthetic_pairwise[i, j] = np.mean(diffs)
            synthetic_pairwise[j, i] = synthetic_pairwise[i, j]

    # For each holdout observation, compute |yᵢ - yⱼ|^p
    # Then average the squared error across all observations
    total_score = 0.0
    for obs_idx in range(n_obs):
        y = Y[obs_idx]
        obs_score = 0.0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                obs_pairwise = np.abs(y[i] - y[j]) ** p
                obs_score += (obs_pairwise - synthetic_pairwise[i, j]) ** 2
        total_score += obs_score

    # Normalize by number of observations and pairs
    n_pairs = n_features * (n_features - 1) / 2
    variogram_score = total_score / (n_obs * n_pairs) if n_pairs > 0 else 0.0

    return float(variogram_score)


def compute_multivariate_metrics(
    train_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    variables: List[str],
    kernel_gamma: float = None,
) -> Dict[str, any]:
    """
    Compute all multivariate realism metrics.

    This is the main entry point for multivariate evaluation.

    Args:
        train_data: Training data (for privacy check)
        holdout_data: Holdout/test data (ground truth)
        synthetic_data: Synthetic data to evaluate
        variables: Variables to include in multivariate space
        kernel_gamma: RBF kernel bandwidth for MMD (default: 1 / n_features)

    Returns:
        Dict with all multivariate metrics:
            - authenticity: Synthetic → Holdout distance stats
            - coverage: Holdout → Synthetic distance stats
            - privacy_ratio: Overfitting check
            - mmd: Maximum Mean Discrepancy
            - energy_distance: Energy distance (multivariate CRPS)
            - variogram_score: Pairwise correlation quality
    """
    # Normalize data
    train_norm, holdout_norm, synthetic_norm, scaler = normalize_data(
        train_data, holdout_data, synthetic_data, variables
    )

    results = {}

    # 1. Authenticity: How realistic are synthetic records?
    print("Computing authenticity distance (Synthetic → Holdout)...")
    results['authenticity'] = compute_authenticity_distance(synthetic_norm, holdout_norm)

    # 2. Coverage: Do we cover the full data manifold?
    print("Computing coverage distance (Holdout → Synthetic)...")
    results['coverage'] = compute_coverage_distance(holdout_norm, synthetic_norm)

    # 3. Privacy/Overfitting check
    print("Computing privacy distance ratio...")
    results['privacy_ratio'] = compute_privacy_distance_ratio(
        synthetic_norm, train_norm, holdout_norm
    )

    # 4. Maximum Mean Discrepancy
    print("Computing Maximum Mean Discrepancy (MMD)...")
    results['mmd'] = compute_mmd(holdout_norm, synthetic_norm, gamma=kernel_gamma)

    # 5. Energy Distance (Multivariate CRPS)
    print("Computing Energy Distance (Multivariate CRPS)...")
    results['energy_distance'] = compute_energy_distance(holdout_norm, synthetic_norm)

    # 6. Variogram Score (Correlation Structure)
    print("Computing Variogram Score (pairwise correlation quality)...")
    results['variogram_score'] = compute_variogram_score(synthetic_norm, holdout_norm)

    return results


def print_multivariate_metrics_report(
    metrics: Dict[str, any],
    method_name: str = "Model",
):
    """
    Print a formatted report of multivariate metrics.

    Args:
        metrics: Output from compute_multivariate_metrics
        method_name: Name of the method being evaluated
    """
    print("\n" + "=" * 80)
    print(f"MULTIVARIATE REALISM METRICS: {method_name.upper()}")
    print("=" * 80)

    # 1. Authenticity
    print("\n1. AUTHENTICITY (Synthetic → Holdout Distance)")
    print("   Measures how close synthetic records are to real data")
    print("   Lower = more realistic, but too low = privacy risk")
    print("-" * 80)
    auth = metrics['authenticity']
    print(f"  Mean distance:   {auth['mean']:.4f}")
    print(f"  Median distance: {auth['median']:.4f}")
    print(f"  Std dev:         {auth['std']:.4f}")
    print(f"  Min distance:    {auth['min']:.4f}  ⚠️  (privacy check)")
    print(f"  Max distance:    {auth['max']:.4f}  (outliers)")
    print(f"  25th percentile: {auth['q25']:.4f}")
    print(f"  75th percentile: {auth['q75']:.4f}")

    # 2. Coverage
    print("\n2. COVERAGE (Holdout → Synthetic Distance)")
    print("   Measures how well synthetic data covers the real data manifold")
    print("   Lower = better coverage")
    print("-" * 80)
    cov = metrics['coverage']
    print(f"  Mean distance:   {cov['mean']:.4f}")
    print(f"  Median distance: {cov['median']:.4f}")
    print(f"  Std dev:         {cov['std']:.4f}")
    print(f"  Max distance:    {cov['max']:.4f}  (coverage gaps)")

    # 3. Privacy Ratio
    print("\n3. PRIVACY / OVERFITTING CHECK")
    print("   Compares synthetic distances to train vs holdout")
    print("   Ratio > 1: Good generalization")
    print("   Ratio ≈ 1: Ideal")
    print("   Ratio < 1: Overfitting risk")
    print("-" * 80)
    priv = metrics['privacy_ratio']
    print(f"  Mean distance ratio (holdout/train): {priv['mean_ratio']:.4f}")
    print(f"  Median ratio:                        {priv['median_ratio']:.4f}")
    print(f"  Fraction closer to train:            {priv['fraction_closer_to_train']:.1%}")
    print(f"  Mean dist to train:                  {priv['mean_dist_to_train']:.4f}")
    print(f"  Mean dist to holdout:                {priv['mean_dist_to_holdout']:.4f}")

    status = "✓ Good" if priv['mean_ratio'] >= 0.9 else "⚠️  Check overfitting"
    print(f"  Status: {status}")

    # 4. MMD
    print("\n4. MAXIMUM MEAN DISCREPANCY (MMD)")
    print("   Kernel-based two-sample test (RBF kernel)")
    print("   MMD = 0 iff distributions identical, higher = more different")
    print("-" * 80)
    print(f"  MMD: {metrics['mmd']:.6f}")

    # 5. Energy Distance
    print("\n5. ENERGY DISTANCE")
    print("   Multivariate two-sample test based on Euclidean distance")
    print("   D = 0 iff distributions identical")
    print("-" * 80)
    print(f"  Energy Distance: {metrics['energy_distance']:.6f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("• Authenticity mean should be small (realistic records)")
    print("  but not too small (privacy risk if < 0.1)")
    print("• Coverage mean should be similar to authenticity (no gaps)")
    print("• Privacy ratio should be ≥ 1.0 (no overfitting)")
    print("• MMD and Energy Distance: compare across methods (lower = better)")
    print("=" * 80 + "\n")


def compare_methods_multivariate(
    train_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    synthetic_datasets: Dict[str, pd.DataFrame],
    variables: List[str],
) -> pd.DataFrame:
    """
    Compare multiple methods using multivariate metrics.

    Args:
        train_data: Training data
        holdout_data: Holdout/test data
        synthetic_datasets: Dict mapping method_name -> synthetic_data
        variables: Variables to evaluate

    Returns:
        DataFrame with comparison across methods
    """
    rows = []

    for method_name, synthetic_data in synthetic_datasets.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {method_name}")
        print('=' * 80)

        metrics = compute_multivariate_metrics(
            train_data, holdout_data, synthetic_data, variables
        )

        # Extract key metrics for comparison table
        row = {
            'Method': method_name,
            'Authenticity (mean)': metrics['authenticity']['mean'],
            'Authenticity (min)': metrics['authenticity']['min'],
            'Coverage (mean)': metrics['coverage']['mean'],
            'Coverage (max)': metrics['coverage']['max'],
            'Privacy Ratio': metrics['privacy_ratio']['mean_ratio'],
            'Closer to Train (%)': metrics['privacy_ratio']['fraction_closer_to_train'] * 100,
            'MMD': metrics['mmd'],
            'Energy Distance': metrics['energy_distance'],
        }
        rows.append(row)

        # Print detailed report
        print_multivariate_metrics_report(metrics, method_name)

    df = pd.DataFrame(rows)

    # Print comparison table
    print("\n" + "=" * 80)
    print("MULTIVARIATE METRICS COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print("\nBest performers (lower is better for all except Privacy Ratio):")
    print(f"  Best authenticity:     {df.loc[df['Authenticity (mean)'].idxmin(), 'Method']}")
    print(f"  Best coverage:         {df.loc[df['Coverage (mean)'].idxmin(), 'Method']}")
    print(f"  Best privacy ratio:    {df.loc[df['Privacy Ratio'].idxmax(), 'Method']}")
    print(f"  Lowest MMD:            {df.loc[df['MMD'].idxmin(), 'Method']}")
    print(f"  Lowest energy dist:    {df.loc[df['Energy Distance'].idxmin(), 'Method']}")
    print("=" * 80 + "\n")

    return df
