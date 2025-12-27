"""
Synthetic microdata generation matching CPS marginal distributions.

Uses copulas to generate synthetic data that:
1. Matches univariate distributions (means, quantiles) from CPS
2. Preserves correlations between variables
3. Handles zero-inflated continuous variables
4. Generates discrete variables with correct proportions

The approach uses a Gaussian copula to model dependencies while
allowing flexible marginal distributions.

Example:
    >>> from microplex.cps_synthetic import CPSSummaryStats, CPSSyntheticGenerator
    >>> stats = CPSSummaryStats.from_dataframe(cps_data)
    >>> gen = CPSSyntheticGenerator(stats)
    >>> synthetic = gen.generate(n=10000, seed=42)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.interpolate import interp1d


@dataclass
class CPSSummaryStats:
    """
    Summary statistics for CPS microdata.

    Captures the distributional properties needed to generate
    synthetic data that matches CPS marginals and correlations.

    Attributes:
        variables: List of variable names
        means: Mean values for continuous variables
        stds: Standard deviations for continuous variables
        quantiles: Decile quantiles for continuous variables
        zero_fractions: Fraction of zeros for zero-inflated variables
        discrete_vars: List of discrete variable names
        discrete_distributions: Category proportions for discrete variables
        correlation_matrix: Correlation matrix between variables
        continuous_vars: List of continuous variable names
    """

    variables: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    quantiles: Dict[str, np.ndarray]
    zero_fractions: Dict[str, float]
    discrete_vars: List[str]
    discrete_distributions: Dict[str, Dict[int, float]]
    correlation_matrix: np.ndarray
    continuous_vars: List[str] = field(default_factory=list)
    quantile_values: Dict[str, np.ndarray] = field(default_factory=dict)
    min_values: Dict[str, float] = field(default_factory=dict)
    max_values: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        weight_col: Optional[str] = None,
        discrete_threshold: int = 10,
    ) -> "CPSSummaryStats":
        """
        Create summary statistics from a DataFrame.

        Args:
            data: DataFrame with CPS-like variables
            weight_col: Name of weight column (optional)
            discrete_threshold: Max unique values to consider discrete

        Returns:
            CPSSummaryStats instance
        """
        variables = [col for col in data.columns if col != weight_col]

        # Identify discrete vs continuous variables
        discrete_vars = []
        continuous_vars = []
        for var in variables:
            nunique = data[var].nunique()
            if nunique <= discrete_threshold and data[var].dtype in [np.int64, np.int32, int, "int64", "int32"]:
                discrete_vars.append(var)
            else:
                continuous_vars.append(var)

        # Get weights
        if weight_col and weight_col in data.columns:
            weights = data[weight_col].values
        else:
            weights = np.ones(len(data))
        weights = weights / weights.sum()

        # Compute means and stds for continuous variables
        means = {}
        stds = {}
        quantiles = {}
        quantile_values = {}
        zero_fractions = {}
        min_values = {}
        max_values = {}

        for var in continuous_vars:
            values = data[var].values.astype(float)

            # Weighted mean
            means[var] = np.sum(weights * values)

            # Weighted std
            variance = np.sum(weights * (values - means[var]) ** 2)
            stds[var] = np.sqrt(variance)

            # Zero fraction (count-based, not weighted for accuracy)
            zero_fractions[var] = float(np.mean(values == 0))

            # Get positive values for quantile computation
            positive_mask = values > 0
            positive_values = values[positive_mask]

            # Min/max of positive values (for interpolation)
            if len(positive_values) > 0:
                min_values[var] = float(np.min(positive_values))
                max_values[var] = float(np.max(positive_values))
            else:
                min_values[var] = 0.0
                max_values[var] = 1.0

            # Quantiles of POSITIVE values only (for zero-inflated vars)
            # This is key: we model P(X | X > 0) separately from P(X = 0)
            q_probs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
            if len(positive_values) > 0:
                q_vals = np.quantile(positive_values, q_probs)
            else:
                q_vals = np.zeros_like(q_probs)
            quantiles[var] = q_probs
            quantile_values[var] = q_vals

        # Compute distributions for discrete variables
        discrete_distributions = {}
        for var in discrete_vars:
            values = data[var].values
            unique_vals = np.unique(values)
            dist = {}
            for val in unique_vals:
                mask = values == val
                dist[int(val)] = float(np.sum(weights[mask]))
            discrete_distributions[var] = dist

        # Compute correlation matrix
        # Use rank correlation (Spearman) for robustness
        numeric_data = data[variables].apply(pd.to_numeric, errors="coerce")
        correlation_matrix = numeric_data.corr(method="spearman").fillna(0).values

        return cls(
            variables=variables,
            means=means,
            stds=stds,
            quantiles=quantiles,
            quantile_values=quantile_values,
            zero_fractions=zero_fractions,
            discrete_vars=discrete_vars,
            discrete_distributions=discrete_distributions,
            correlation_matrix=correlation_matrix,
            continuous_vars=continuous_vars,
            min_values=min_values,
            max_values=max_values,
        )


class CPSSyntheticGenerator:
    """
    Generate synthetic microdata matching CPS summary statistics.

    Uses a Gaussian copula approach:
    1. Generate correlated standard normal samples
    2. Transform to uniform using normal CDF
    3. Transform uniform to target marginal using inverse CDF

    This preserves the correlation structure while matching
    arbitrary marginal distributions.

    Example:
        >>> stats = CPSSummaryStats.from_dataframe(cps_data)
        >>> gen = CPSSyntheticGenerator(stats)
        >>> synthetic = gen.generate(n=10000)
    """

    def __init__(self, stats: CPSSummaryStats):
        """
        Initialize generator with summary statistics.

        Args:
            stats: CPSSummaryStats object with target distributions
        """
        self.stats = stats

        # Pre-compute Cholesky decomposition for correlation matrix
        # Use slight regularization for numerical stability
        corr = stats.correlation_matrix.copy()
        np.fill_diagonal(corr, 1.0)  # Ensure diagonal is exactly 1

        # Regularize if needed
        min_eig = np.min(np.linalg.eigvalsh(corr))
        if min_eig < 1e-6:
            corr = corr + (1e-6 - min_eig) * np.eye(corr.shape[0])

        self.cholesky = np.linalg.cholesky(corr)

        # Build quantile interpolators for continuous variables
        self._build_marginal_transforms()

    def _build_marginal_transforms(self):
        """Build inverse CDF transforms for marginal distributions."""
        self.marginal_transforms = {}

        for var in self.stats.continuous_vars:
            q_probs = self.stats.quantiles[var]
            q_vals = self.stats.quantile_values[var]

            # Add endpoints
            probs = np.concatenate([[0.0], q_probs, [1.0]])
            vals = np.concatenate([
                [self.stats.min_values[var]],
                q_vals,
                [self.stats.max_values[var]]
            ])

            # Remove duplicates (in case min/max equals a quantile)
            unique_mask = np.concatenate([[True], np.diff(probs) > 1e-10])
            probs = probs[unique_mask]
            vals = vals[unique_mask]

            # Create interpolator
            self.marginal_transforms[var] = interp1d(
                probs, vals,
                kind="linear",
                bounds_error=False,
                fill_value=(vals[0], vals[-1])
            )

    def generate(
        self,
        n: int,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic microdata.

        Args:
            n: Number of records to generate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with synthetic data matching CPS distributions
        """
        if seed is not None:
            np.random.seed(seed)

        # Step 1: Generate correlated standard normal samples
        z = np.random.standard_normal((n, len(self.stats.variables)))
        z_correlated = z @ self.cholesky.T

        # Step 2: Transform to uniform using normal CDF
        u = scipy_stats.norm.cdf(z_correlated)

        # Step 3: Transform to target marginals
        result = {}
        for i, var in enumerate(self.stats.variables):
            if var in self.stats.discrete_vars:
                # Discrete: use inverse CDF sampling
                result[var] = self._sample_discrete(var, u[:, i])
            else:
                # Continuous: use quantile transform
                result[var] = self._transform_continuous(var, u[:, i])

        return pd.DataFrame(result)

    def _sample_discrete(self, var: str, u: np.ndarray) -> np.ndarray:
        """
        Sample discrete variable using inverse CDF.

        Args:
            var: Variable name
            u: Uniform [0,1] samples

        Returns:
            Discrete samples
        """
        dist = self.stats.discrete_distributions[var]
        categories = sorted(dist.keys())
        probs = np.array([dist[c] for c in categories])

        # Normalize probabilities
        probs = probs / probs.sum()

        # Compute CDF
        cdf = np.cumsum(probs)

        # Inverse CDF transform
        result = np.zeros(len(u), dtype=int)
        for i, cat in enumerate(categories):
            if i == 0:
                mask = u <= cdf[i]
            else:
                mask = (u > cdf[i - 1]) & (u <= cdf[i])
            result[mask] = cat

        # Handle edge case (u exactly at boundaries)
        result[u > cdf[-1]] = categories[-1]

        return result

    def _transform_continuous(self, var: str, u: np.ndarray) -> np.ndarray:
        """
        Transform uniform samples to target continuous distribution.

        Handles zero-inflation by:
        1. Checking if sample falls in zero region
        2. Otherwise, using quantile transform for positive values

        Args:
            var: Variable name
            u: Uniform [0,1] samples

        Returns:
            Samples from target distribution
        """
        zero_frac = self.stats.zero_fractions.get(var, 0)

        if zero_frac > 0:
            # Zero-inflated: split into zero and positive regions
            result = np.zeros(len(u))

            # Zero region: u < zero_frac
            zero_mask = u < zero_frac

            # Positive region: rescale u to [0,1] for quantile transform
            positive_mask = ~zero_mask
            if positive_mask.any():
                # Rescale uniform to positive range
                u_positive = (u[positive_mask] - zero_frac) / (1 - zero_frac)
                u_positive = np.clip(u_positive, 0, 1)

                # Apply quantile transform
                result[positive_mask] = self.marginal_transforms[var](u_positive)

            return result
        else:
            # No zero inflation: direct quantile transform
            return self.marginal_transforms[var](u)


def validate_synthetic(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    variables: Optional[List[str]] = None,
) -> Dict[str, Union[Dict[str, float], float]]:
    """
    Validate synthetic data against reference data.

    Computes various metrics comparing distributions:
    - KS statistics for continuous variables
    - Mean relative errors
    - Correlation errors
    - Distribution divergence for discrete variables

    Args:
        reference: Reference DataFrame (e.g., CPS data)
        synthetic: Synthetic DataFrame to validate
        variables: Variables to validate (default: all common columns)

    Returns:
        Dictionary of validation metrics
    """
    if variables is None:
        variables = [col for col in reference.columns if col in synthetic.columns]

    metrics = {
        "ks_statistics": {},
        "mean_errors": {},
        "std_errors": {},
        "correlation_errors": {},
    }

    # KS statistics and mean/std errors
    for var in variables:
        ref_vals = reference[var].dropna().values
        syn_vals = synthetic[var].dropna().values

        if len(ref_vals) > 0 and len(syn_vals) > 0:
            # KS statistic
            ks_stat, _ = scipy_stats.ks_2samp(ref_vals, syn_vals)
            metrics["ks_statistics"][var] = float(ks_stat)

            # Mean error
            ref_mean = np.mean(ref_vals)
            syn_mean = np.mean(syn_vals)
            if ref_mean != 0:
                metrics["mean_errors"][var] = abs(syn_mean - ref_mean) / abs(ref_mean)
            else:
                metrics["mean_errors"][var] = abs(syn_mean)

            # Std error
            ref_std = np.std(ref_vals)
            syn_std = np.std(syn_vals)
            if ref_std != 0:
                metrics["std_errors"][var] = abs(syn_std - ref_std) / ref_std
            else:
                metrics["std_errors"][var] = abs(syn_std)

    # Correlation errors
    numeric_vars = [v for v in variables if v in reference.columns and v in synthetic.columns]
    if len(numeric_vars) >= 2:
        ref_corr = reference[numeric_vars].corr(method="spearman").fillna(0)
        syn_corr = synthetic[numeric_vars].corr(method="spearman").fillna(0)

        for i, var1 in enumerate(numeric_vars):
            for j, var2 in enumerate(numeric_vars):
                if i < j:
                    pair = f"{var1}_vs_{var2}"
                    ref_r = ref_corr.loc[var1, var2]
                    syn_r = syn_corr.loc[var1, var2]
                    metrics["correlation_errors"][pair] = abs(syn_r - ref_r)

    # Aggregate metrics
    metrics["mean_ks"] = np.mean(list(metrics["ks_statistics"].values())) if metrics["ks_statistics"] else 0
    metrics["mean_corr_error"] = np.mean(list(metrics["correlation_errors"].values())) if metrics["correlation_errors"] else 0

    return metrics
