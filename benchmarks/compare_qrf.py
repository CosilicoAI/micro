"""
Benchmark comparing microplex against Sequential Quantile Random Forests (QRF).

QRF is PolicyEngine's current approach for microdata enhancement, using
sequential/chained imputation where each target variable is predicted
one at a time using quantile regression forests.

Key differences:
- QRF: Sequential imputation (predict var1, then var2|var1, etc.)
- microplex: Joint distribution modeling via normalizing flows

Critical test areas:
- Marginal distribution matching (QRF should be good)
- Correlation preservation (QRF may break correlations in sequential mode)
- Zero-inflation handling (QRF has no principled zero-inflation modeling)
- Joint distribution quality (microplex advantage)
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class QRFBenchmarkResult:
    """Results from QRF benchmark."""

    method: str
    dataset: str

    # Marginal fidelity
    ks_stats: Dict[str, float]
    mean_ks: float

    # Joint fidelity
    correlation_error: float

    # Zero handling
    zero_fraction_error: Dict[str, float]
    mean_zero_error: float

    # Joint distribution quality (new metric)
    conditional_correlation_error: float

    # Timing
    train_time: float
    generate_time: float

    # Metadata
    n_train: int
    n_generate: int


class SequentialQRF:
    """
    Sequential Quantile Random Forest implementation.

    Mimics PolicyEngine's current approach:
    1. Predict target_var1 | conditions
    2. Predict target_var2 | conditions + target_var1
    3. Predict target_var3 | conditions + target_var1 + target_var2
    ...

    Uses HistGradientBoostingRegressor with quantile loss for fast training.
    """

    def __init__(
        self,
        target_vars: List[str],
        condition_vars: List[str],
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
    ):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # One model per target variable
        self.models = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False):
        """
        Fit sequential QRF models.

        For each target variable, train a quantile regression model
        conditioning on demographics + previously predicted targets.
        """
        features_so_far = self.condition_vars.copy()

        for i, target in enumerate(self.target_vars):
            if verbose:
                print(
                    f"  Training QRF {i+1}/{len(self.target_vars)}: {target} | {features_so_far}"
                )

            # Features = conditions + previously predicted targets
            X = data[features_so_far].values
            y = data[target].values

            # Train quantile regression model (median)
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.5,  # Median
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + i,
                verbose=0,
            )
            model.fit(X, y)

            self.models[target] = {
                "model": model,
                "features": features_so_far.copy(),
            }

            # Add this target to feature set for next variable
            features_so_far.append(target)

    def generate(self, conditions: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic data using sequential prediction.

        For each sample:
        1. Predict target_var1 from conditions
        2. Predict target_var2 from conditions + predicted target_var1
        3. Continue sequentially...

        Uses random quantile sampling to introduce variability.
        """
        n_samples = len(conditions)
        result = conditions.copy()

        for target in self.target_vars:
            model_info = self.models[target]
            model = model_info["model"]
            features = model_info["features"]

            # Get features (includes previously predicted targets)
            X = result[features].values

            # Predict using model
            # Add noise by training multiple quantile models and sampling
            predictions = model.predict(X)

            # Add gaussian noise to simulate quantile sampling
            # (proper implementation would train models at different quantiles)
            noise_scale = np.std(predictions) * 0.3
            predictions += np.random.normal(0, noise_scale, n_samples)

            # Ensure non-negative for economic variables
            predictions = np.maximum(predictions, 0)

            result[target] = predictions

        return result


class SequentialQRFWithZeroInflation:
    """
    Enhanced Sequential QRF with two-stage zero-inflation handling.

    For each target variable:
    1. Train binary classifier for P(positive | features)
    2. Train quantile regression for P(value | positive, features)

    This is a fairer comparison to microplex's two-stage approach.
    """

    def __init__(
        self,
        target_vars: List[str],
        condition_vars: List[str],
        n_estimators: int = 100,
        max_depth: int = 10,
        zero_threshold: float = 1e-6,
        random_state: int = 42,
    ):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.zero_threshold = zero_threshold
        self.random_state = random_state

        self.models = {}

    def fit(self, data: pd.DataFrame, verbose: bool = False):
        """Fit two-stage models for each target variable."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        features_so_far = self.condition_vars.copy()

        for i, target in enumerate(self.target_vars):
            if verbose:
                print(
                    f"  Training QRF+ZI {i+1}/{len(self.target_vars)}: {target} | {features_so_far}"
                )

            X = data[features_so_far].values
            y = data[target].values

            # Stage 1: Binary classifier for P(positive)
            is_positive = (y > self.zero_threshold).astype(int)
            zero_frac = (is_positive == 0).mean()

            classifier = None
            if 0.01 < zero_frac < 0.99:  # Only train if non-trivial
                classifier = HistGradientBoostingClassifier(
                    max_iter=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state + i,
                    verbose=0,
                )
                classifier.fit(X, is_positive)

            # Stage 2: Quantile regression for positive values
            regressor = None
            if is_positive.sum() > 10:  # Need enough positive samples
                X_pos = X[is_positive == 1]
                y_pos = y[is_positive == 1]

                regressor = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.5,
                    max_iter=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state + i,
                    verbose=0,
                )
                regressor.fit(X_pos, y_pos)

            self.models[target] = {
                "classifier": classifier,
                "regressor": regressor,
                "features": features_so_far.copy(),
                "zero_frac": zero_frac,
            }

            features_so_far.append(target)

    def generate(self, conditions: pd.DataFrame) -> pd.DataFrame:
        """Generate using two-stage process."""
        n_samples = len(conditions)
        result = conditions.copy()

        for target in self.target_vars:
            model_info = self.models[target]
            classifier = model_info["classifier"]
            regressor = model_info["regressor"]
            features = model_info["features"]
            zero_frac = model_info["zero_frac"]

            X = result[features].values

            # Stage 1: Predict which samples are positive
            if classifier is not None:
                is_positive_proba = classifier.predict_proba(X)[:, 1]
                is_positive = np.random.random(n_samples) < is_positive_proba
            else:
                # Fallback to observed zero fraction
                is_positive = np.random.random(n_samples) > zero_frac

            # Stage 2: Predict values for positive samples
            predictions = np.zeros(n_samples)

            if regressor is not None and is_positive.sum() > 0:
                X_pos = X[is_positive]
                pred_pos = regressor.predict(X_pos)

                # Add noise
                noise_scale = np.std(pred_pos) * 0.3
                pred_pos += np.random.normal(0, noise_scale, len(pred_pos))
                pred_pos = np.maximum(pred_pos, 0)

                predictions[is_positive] = pred_pos
            elif is_positive.sum() > 0:
                # Fallback: sample from training data
                # This shouldn't happen often
                predictions[is_positive] = 1000.0  # Placeholder

            result[target] = predictions

        return result


def compute_conditional_correlation_error(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_vars: List[str],
    condition_vars: List[str],
) -> float:
    """
    Compute conditional correlation preservation.

    This tests whether correlations between targets are preserved
    WITHIN subgroups defined by condition variables.

    This is where sequential QRF is expected to struggle.

    Note: We use the synthetic data's conditions to group both datasets,
    since the test set is what was used for generation.
    """
    # Reset indices to ensure alignment
    real = real.reset_index(drop=True)
    synthetic = synthetic.reset_index(drop=True)

    # Use synthetic data's condition variables for grouping
    # (since it was generated from test conditions)
    binned_data = synthetic[condition_vars].copy()
    for var in condition_vars:
        if synthetic[var].nunique() <= 5:
            binned_data[var] = synthetic[var]
        else:
            binned_data[var] = pd.qcut(
                synthetic[var], q=3, labels=False, duplicates="drop"
            )

    # Compute conditional correlations per group
    errors = []
    for group_vals, group_df in binned_data.groupby(list(condition_vars)):
        if len(group_df) < 10:
            continue

        # Get position indices for this group
        idx = group_df.index.tolist()

        # For real data, we need to sample a comparable subset
        # Let's use a simple approach: compare against full real data's correlation
        # weighted by group size
        synth_corr = synthetic.iloc[idx][target_vars].corr().values

        # For real data, use the full correlation as baseline
        # (This is a simplification - ideally we'd match demographics)
        real_corr = real[target_vars].corr().values

        # Error
        error = np.sqrt(np.sum((real_corr - synth_corr) ** 2))
        errors.append(error)

    return np.mean(errors) if errors else 0.0


def benchmark_qrf_vs_microplex(
    train_data: pd.DataFrame,
    test_conditions: pd.DataFrame,
    target_vars: List[str],
    condition_vars: List[str],
    epochs: int = 100,
) -> Tuple[List[QRFBenchmarkResult], Dict[str, pd.DataFrame]]:
    """
    Run comprehensive benchmark: QRF vs QRF+ZI vs microplex.

    Returns:
        results: List of benchmark results
        synthetic_data: Dict mapping method name to synthetic data
    """
    results = []
    synthetic_data = {}

    methods = {
        "qrf_sequential": SequentialQRF,
        "qrf_zero_inflation": SequentialQRFWithZeroInflation,
    }

    # Benchmark QRF methods
    for method_name, method_cls in methods.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {method_name.upper()}")
        print(f"{'='*60}")

        model = method_cls(target_vars, condition_vars)

        # Training
        print("Training...")
        start = time.time()
        model.fit(train_data, verbose=True)
        train_time = time.time() - start
        print(f"Training time: {train_time:.1f}s")

        # Generation
        print("Generating synthetic data...")
        start = time.time()
        synthetic = model.generate(test_conditions)
        generate_time = time.time() - start
        print(f"Generation time: {generate_time:.1f}s")

        # Compute metrics
        print("Computing metrics...")

        # Marginal fidelity
        ks_stats = {}
        for var in target_vars:
            stat, _ = stats.ks_2samp(train_data[var], synthetic[var])
            ks_stats[var] = stat
        mean_ks = np.mean(list(ks_stats.values()))

        # Joint fidelity
        real_corr = train_data[target_vars].corr().values
        synth_corr = synthetic[target_vars].corr().values
        corr_error = np.sqrt(np.sum((real_corr - synth_corr) ** 2)) / len(target_vars)

        # Zero fidelity
        zero_errors = {}
        for var in target_vars:
            real_zero = (train_data[var] == 0).mean()
            synth_zero = (synthetic[var] == 0).mean()
            zero_errors[var] = abs(real_zero - synth_zero)
        mean_zero_error = np.mean(list(zero_errors.values()))

        # Conditional correlation (new metric)
        cond_corr_error = compute_conditional_correlation_error(
            train_data, synthetic, target_vars, condition_vars
        )

        result = QRFBenchmarkResult(
            method=method_name,
            dataset="economic_microdata",
            ks_stats=ks_stats,
            mean_ks=mean_ks,
            correlation_error=corr_error,
            zero_fraction_error=zero_errors,
            mean_zero_error=mean_zero_error,
            conditional_correlation_error=cond_corr_error,
            train_time=train_time,
            generate_time=generate_time,
            n_train=len(train_data),
            n_generate=len(test_conditions),
        )

        results.append(result)
        synthetic_data[method_name] = synthetic

        # Print summary
        print(f"\nResults:")
        print(f"  Mean KS: {mean_ks:.4f}")
        print(f"  Correlation error: {corr_error:.4f}")
        print(f"  Conditional correlation error: {cond_corr_error:.4f}")
        print(f"  Zero-fraction error: {mean_zero_error:.4f}")

    # Benchmark microplex for comparison
    print(f"\n{'='*60}")
    print("Benchmarking: MICROPLEX (for comparison)")
    print(f"{'='*60}")

    try:
        from microplex import Synthesizer

        model = Synthesizer(target_vars=target_vars, condition_vars=condition_vars)

        print("Training...")
        start = time.time()
        model.fit(train_data, epochs=epochs, verbose=False)
        train_time = time.time() - start
        print(f"Training time: {train_time:.1f}s")

        print("Generating...")
        start = time.time()
        synthetic = model.generate(test_conditions)
        generate_time = time.time() - start
        print(f"Generation time: {generate_time:.1f}s")

        # Metrics
        ks_stats = {}
        for var in target_vars:
            stat, _ = stats.ks_2samp(train_data[var], synthetic[var])
            ks_stats[var] = stat
        mean_ks = np.mean(list(ks_stats.values()))

        real_corr = train_data[target_vars].corr().values
        synth_corr = synthetic[target_vars].corr().values
        corr_error = np.sqrt(np.sum((real_corr - synth_corr) ** 2)) / len(target_vars)

        zero_errors = {}
        for var in target_vars:
            real_zero = (train_data[var] == 0).mean()
            synth_zero = (synthetic[var] == 0).mean()
            zero_errors[var] = abs(real_zero - synth_zero)
        mean_zero_error = np.mean(list(zero_errors.values()))

        cond_corr_error = compute_conditional_correlation_error(
            train_data, synthetic, target_vars, condition_vars
        )

        result = QRFBenchmarkResult(
            method="microplex",
            dataset="economic_microdata",
            ks_stats=ks_stats,
            mean_ks=mean_ks,
            correlation_error=corr_error,
            zero_fraction_error=zero_errors,
            mean_zero_error=mean_zero_error,
            conditional_correlation_error=cond_corr_error,
            train_time=train_time,
            generate_time=generate_time,
            n_train=len(train_data),
            n_generate=len(test_conditions),
        )

        results.append(result)
        synthetic_data["microplex"] = synthetic

        print(f"\nResults:")
        print(f"  Mean KS: {mean_ks:.4f}")
        print(f"  Correlation error: {corr_error:.4f}")
        print(f"  Conditional correlation error: {cond_corr_error:.4f}")
        print(f"  Zero-fraction error: {mean_zero_error:.4f}")

    except Exception as e:
        print(f"ERROR: microplex benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    return results, synthetic_data


if __name__ == "__main__":
    # Quick test
    from run_benchmarks import generate_realistic_microdata

    print("Generating test data...")
    data = generate_realistic_microdata(n_samples=5000, seed=42)

    train = data.iloc[:4000]
    test = data.iloc[4000:]

    target_vars = ["income", "assets", "debt", "savings"]
    condition_vars = ["age", "education", "region"]

    results, synth = benchmark_qrf_vs_microplex(
        train, test[condition_vars], target_vars, condition_vars, epochs=50
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r.method}:")
        print(f"  KS: {r.mean_ks:.4f}")
        print(f"  Corr: {r.correlation_error:.4f}")
        print(f"  Zero: {r.mean_zero_error:.4f}")
