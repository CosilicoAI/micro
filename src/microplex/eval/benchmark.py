"""Synthesis method comparison benchmark.

Compares multiple synthesis approaches on PRDC coverage outcomes:
- QRF / ZI-QRF (quantile regression forest +/- zero-inflation)
- QDNN / ZI-QDNN (quantile deep neural network +/- zero-inflation)
- MAF / ZI-MAF (masked autoregressive flow +/- zero-inflation)
- CTGAN (conditional tabular GAN)
- TVAE (tabular variational autoencoder)

All methods are evaluated on the same train/holdout splits using
PRDC metrics (Precision, Recall, Density, Coverage) from Naeem et al. (2020).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:
    RandomForestQuantileRegressor = None


# --- Protocol ---


@runtime_checkable
class SynthesisMethod(Protocol):
    """Protocol for synthesis methods in the benchmark."""

    name: str

    def fit(
        self, sources: dict[str, pd.DataFrame], shared_cols: list[str]
    ) -> "SynthesisMethod": ...

    def generate(self, n: int, seed: int = 42) -> pd.DataFrame: ...


# --- Result dataclasses ---


@dataclass
class SourceResult:
    """PRDC metrics for one source's holdout."""

    source_name: str
    precision: float
    recall: float
    density: float
    coverage: float
    n_holdout: int
    n_synthetic: int

    def to_dict(self) -> dict:
        return {
            "source": self.source_name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "density": round(self.density, 4),
            "coverage": round(self.coverage, 4),
            "n_holdout": self.n_holdout,
            "n_synthetic": self.n_synthetic,
        }


@dataclass
class MethodResult:
    """Results for one synthesis method across all sources."""

    method_name: str
    source_results: list[SourceResult]
    elapsed_seconds: float

    @property
    def mean_coverage(self) -> float:
        if not self.source_results:
            return 0.0
        return float(np.mean([s.coverage for s in self.source_results]))

    @property
    def mean_precision(self) -> float:
        if not self.source_results:
            return 0.0
        return float(np.mean([s.precision for s in self.source_results]))

    @property
    def mean_recall(self) -> float:
        if not self.source_results:
            return 0.0
        return float(np.mean([s.recall for s in self.source_results]))

    @property
    def mean_density(self) -> float:
        if not self.source_results:
            return 0.0
        return float(np.mean([s.density for s in self.source_results]))

    def to_dict(self) -> dict:
        return {
            "mean_coverage": round(self.mean_coverage, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_density": round(self.mean_density, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "sources": [s.to_dict() for s in self.source_results],
        }


@dataclass
class BenchmarkResult:
    """Results from comparing all synthesis methods."""

    method_results: list[MethodResult]
    holdout_frac: float
    n_generate: int
    k: int
    seed: int

    def to_dict(self) -> dict:
        return {
            "holdout_frac": self.holdout_frac,
            "n_generate": self.n_generate,
            "k": self.k,
            "seed": self.seed,
            "methods": {mr.method_name: mr.to_dict() for mr in self.method_results},
        }

    def summary(self) -> str:
        lines = [
            "Synthesis Method Benchmark",
            "=" * 75,
            f"{'Method':<12} {'Coverage':>10} {'Precision':>10} {'Recall':>10} "
            f"{'Density':>10} {'Time':>10}",
            "-" * 75,
        ]
        for mr in sorted(self.method_results, key=lambda x: -x.mean_coverage):
            lines.append(
                f"{mr.method_name:<12} {mr.mean_coverage:>10.1%} "
                f"{mr.mean_precision:>10.1%} {mr.mean_recall:>10.1%} "
                f"{mr.mean_density:>10.2f} {mr.elapsed_seconds:>9.1f}s"
            )
        lines.append("=" * 75)
        return "\n".join(lines)


# --- PRDC computation ---


def _compute_prdc(real: np.ndarray, synthetic: np.ndarray, k: int = 5) -> dict[str, float]:
    """Compute Precision, Recall, Density, Coverage via k-NN."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import pairwise_distances

    if len(real) < k + 1 or len(synthetic) < k + 1:
        return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}

    scaler = StandardScaler()
    real_s = scaler.fit_transform(real)
    synth_s = scaler.transform(synthetic)

    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real_s)
    real_dists, _ = nn_real.kneighbors(real_s)
    real_radii = real_dists[:, -1]

    nn_synth = NearestNeighbors(n_neighbors=k + 1).fit(synth_s)
    synth_dists, _ = nn_synth.kneighbors(synth_s)
    synth_radii = synth_dists[:, -1]

    nn_synth_1 = NearestNeighbors(n_neighbors=1).fit(synth_s)
    real_to_synth_dist, _ = nn_synth_1.kneighbors(real_s)
    real_to_synth_dist = real_to_synth_dist[:, 0]

    nn_real_1 = NearestNeighbors(n_neighbors=1).fit(real_s)
    synth_to_real_dist, _ = nn_real_1.kneighbors(synth_s)
    synth_to_real_dist = synth_to_real_dist[:, 0]

    coverage = float((real_to_synth_dist <= real_radii).mean())
    precision = float((synth_to_real_dist <= synth_radii).mean())

    max_density_samples = 2000
    if len(synth_s) > max_density_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(synth_s), max_density_samples, replace=False)
        synth_sample = synth_s[idx]
        radii_sample = synth_radii[idx]
    else:
        synth_sample = synth_s
        radii_sample = synth_radii

    dists = pairwise_distances(synth_sample, real_s)
    counts = (dists <= radii_sample[:, None]).sum(axis=1)
    density = float(counts.mean() / k)

    return {
        "precision": precision,
        "recall": coverage,  # recall = coverage in k-NN formulation
        "density": density,
        "coverage": coverage,
    }


# --- Base class for multi-source synthesis methods ---


class _MultiSourceBase:
    """Base class implementing the multi-source fit/generate pattern.

    Subclasses implement _fit_column() and _generate_column() for each
    non-shared column.
    """

    name: str = "Base"

    def __init__(self, zero_inflated: bool = False, zero_threshold: float = 0.1):
        self.zero_inflated = zero_inflated
        self.zero_threshold = zero_threshold
        self.shared_cols_: list[str] = []
        self.all_cols_: list[str] = []
        self.col_to_survey_: dict[str, str] = {}
        self.shared_data_: Optional[pd.DataFrame] = None
        self._col_models: dict = {}
        self._zero_classifiers: dict = {}
        self._col_stats: dict = {}

    def fit(
        self, sources: dict[str, pd.DataFrame], shared_cols: list[str]
    ) -> "_MultiSourceBase":
        self.shared_cols_ = list(shared_cols)
        all_cols = set(shared_cols)
        for survey_name, df in sources.items():
            for col in df.columns:
                if col not in all_cols:
                    all_cols.add(col)
                    self.col_to_survey_[col] = survey_name

        self.all_cols_ = list(all_cols)

        # Pool shared columns
        shared_dfs = []
        for survey_name, df in sources.items():
            available = [c for c in shared_cols if c in df.columns]
            if len(available) == len(shared_cols):
                shared_dfs.append(df[shared_cols].copy())
        self.shared_data_ = pd.concat(shared_dfs, ignore_index=True) if shared_dfs else \
            list(sources.values())[0][shared_cols].copy()

        # Fit model for each non-shared column
        for col in self.all_cols_:
            if col in shared_cols:
                continue

            survey_name = self.col_to_survey_[col]
            survey_df = sources[survey_name]
            available_shared = [c for c in shared_cols if c in survey_df.columns]
            X = survey_df[available_shared].values
            y = survey_df[col].values

            # Check zero-inflation
            min_val = float(np.nanmin(y))
            at_min = np.isclose(y, min_val, atol=1e-6)
            zero_frac = at_min.sum() / len(y)
            self._col_stats[col] = {"min": min_val, "zero_frac": zero_frac}

            if self.zero_inflated and zero_frac >= self.zero_threshold and at_min.sum() >= 10:
                # Stage 1: Zero classifier
                clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                clf.fit(X, (~at_min).astype(int))
                self._zero_classifiers[col] = clf

                # Stage 2: Model on non-zero values
                if (~at_min).sum() >= 10:
                    self._fit_column(col, X[~at_min], y[~at_min])
            else:
                self._fit_column(col, X, y)

        return self

    def generate(self, n: int, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)

        # Sample shared variables
        sample_idx = rng.choice(len(self.shared_data_), size=n, replace=True)
        shared_values = self.shared_data_.iloc[sample_idx].values.copy()
        shared_values += rng.normal(0, 0.1, shared_values.shape)

        synthetic = pd.DataFrame(shared_values, columns=self.shared_cols_)

        for col in self.all_cols_:
            if col in self.shared_cols_:
                continue

            X = shared_values

            if col in self._zero_classifiers:
                # Two-stage generation
                results = np.full(n, self._col_stats[col]["min"])
                clf = self._zero_classifiers[col]
                proba = clf.predict_proba(X)
                if proba.shape[1] == 1:
                    probs = np.full(n, float(clf.classes_[0]))
                else:
                    probs = proba[:, 1]
                is_nonzero = rng.random(n) < probs

                if col in self._col_models and is_nonzero.sum() > 0:
                    results[is_nonzero] = self._generate_column(col, X[is_nonzero], rng)

                synthetic[col] = results
            elif col in self._col_models:
                synthetic[col] = self._generate_column(col, X, rng)

        return synthetic

    def _fit_column(self, col: str, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def _generate_column(self, col: str, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        raise NotImplementedError


# --- QRF Methods ---


class QRFMethod(_MultiSourceBase):
    """Quantile Regression Forest (no zero-inflation)."""

    name = "QRF"

    def __init__(self, n_estimators: int = 100, quantiles: list[float] = None, **kwargs):
        super().__init__(zero_inflated=False)
        self.n_estimators = n_estimators
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]

    def _fit_column(self, col, X, y):
        if RandomForestQuantileRegressor is None:
            raise ImportError("quantile-forest required: pip install quantile-forest")
        qrf = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators, random_state=42, n_jobs=-1
        )
        qrf.fit(X, y)
        self._col_models[col] = qrf

    def _generate_column(self, col, X, rng):
        qrf = self._col_models[col]
        preds = qrf.predict(X, quantiles=self.quantiles)
        q_choices = rng.choice(len(self.quantiles), size=len(X))
        return preds[np.arange(len(X)), q_choices]


class ZIQRFMethod(QRFMethod):
    """Zero-Inflated Quantile Regression Forest."""

    name = "ZI-QRF"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zero_inflated = True


# --- QDNN Methods ---


class QDNNMethod(_MultiSourceBase):
    """Quantile Deep Neural Network (no zero-inflation).

    Uses a small MLP to predict quantiles of the conditional distribution.
    """

    name = "QDNN"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        quantiles: list[float] = None,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(zero_inflated=False)
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.quantiles = quantiles or [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def _fit_column(self, col, X, y):
        import torch
        import torch.nn as nn

        n_quantiles = len(self.quantiles)
        n_input = X.shape[1]

        # Build MLP
        layers = [nn.Linear(n_input, self.hidden_dim), nn.ReLU()]
        for _ in range(self.n_hidden - 1):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(self.hidden_dim, n_quantiles))
        model = nn.Sequential(*layers)

        # Quantile loss (pinball loss)
        quantiles_t = torch.tensor(self.quantiles, dtype=torch.float32)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)  # [batch, n_quantiles]
                errors = batch_y - pred  # [batch, n_quantiles]
                loss = torch.max(
                    quantiles_t * errors,
                    (quantiles_t - 1) * errors,
                ).mean()
                loss.backward()
                optimizer.step()

        model.eval()
        self._col_models[col] = model

    def _generate_column(self, col, X, rng):
        import torch

        model = self._col_models[col]
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            preds = model(X_t).numpy()  # [n, n_quantiles]

        # Sample a random quantile for each record
        q_choices = rng.choice(len(self.quantiles), size=len(X))
        return preds[np.arange(len(X)), q_choices]


class ZIQDNNMethod(QDNNMethod):
    """Zero-Inflated Quantile Deep Neural Network."""

    name = "ZI-QDNN"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zero_inflated = True


# --- MAF Methods ---


class MAFMethod(_MultiSourceBase):
    """Masked Autoregressive Flow (no zero-inflation).

    Uses the existing microplex ConditionalMAF for each column.
    For multi-source synthesis, we train a separate flow per non-shared column.
    """

    name = "MAF"

    def __init__(
        self,
        n_layers: int = 4,
        hidden_dim: int = 32,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__(zero_inflated=False)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def _fit_column(self, col, X, y):
        import torch
        from microplex.flows import ConditionalMAF

        n_context = X.shape[1]

        # Log-transform positive values for better flow training
        y_transformed = y.copy()
        positive_mask = y > 0
        if positive_mask.any():
            y_transformed[positive_mask] = np.log1p(y[positive_mask])

        # Standardize
        y_mean = y_transformed.mean()
        y_std = max(y_transformed.std(), 1e-6)
        y_norm = (y_transformed - y_mean) / y_std

        self._col_stats[col].update({"y_mean": y_mean, "y_std": y_std})

        flow = ConditionalMAF(
            n_features=1,
            n_context=n_context,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)

        optimizer = torch.optim.Adam(flow.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        flow.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                log_prob = flow.log_prob(batch_y, batch_X)
                loss = -log_prob.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                optimizer.step()

        flow.eval()
        self._col_models[col] = flow

    def _generate_column(self, col, X, rng):
        import torch

        flow = self._col_models[col]
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            samples = flow.sample(X_t).numpy().flatten()

        # Inverse standardization
        y_mean = self._col_stats[col].get("y_mean", 0)
        y_std = self._col_stats[col].get("y_std", 1)
        samples = samples * y_std + y_mean

        # Inverse log transform
        samples = np.expm1(np.clip(samples, -20, 20))
        return np.maximum(samples, 0)


class ZIMAFMethod(MAFMethod):
    """Zero-Inflated Masked Autoregressive Flow."""

    name = "ZI-MAF"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zero_inflated = True


# --- CTGAN / TVAE Methods ---


class CTGANMethod:
    """Conditional Tabular GAN (from SDV library)."""

    name = "CTGAN"

    def __init__(self, epochs: int = 300, batch_size: int = 500, **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None
        self._columns = None

    def fit(
        self, sources: dict[str, pd.DataFrame], shared_cols: list[str]
    ) -> "CTGANMethod":
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            raise ImportError("sdv required: pip install sdv")

        # Stack all sources
        dfs = []
        for name, df in sources.items():
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)

        # Keep only numeric columns with <50% NaN
        numeric_cols = [c for c in combined.columns
                        if combined[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                        and combined[c].isna().mean() < 0.5]
        combined = combined[numeric_cols].dropna()
        self._columns = numeric_cols

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(combined)

        self._model = CTGANSynthesizer(
            metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=False,
        )
        self._model.fit(combined)
        return self

    def generate(self, n: int, seed: int = 42) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Not fitted")
        return self._model.sample(n)


class TVAEMethod:
    """Tabular Variational Autoencoder (from SDV library)."""

    name = "TVAE"

    def __init__(self, epochs: int = 300, batch_size: int = 500, **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None
        self._columns = None

    def fit(
        self, sources: dict[str, pd.DataFrame], shared_cols: list[str]
    ) -> "TVAEMethod":
        try:
            from sdv.single_table import TVAESynthesizer
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            raise ImportError("sdv required: pip install sdv")

        dfs = []
        for name, df in sources.items():
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)

        numeric_cols = [c for c in combined.columns
                        if combined[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                        and combined[c].isna().mean() < 0.5]
        combined = combined[numeric_cols].dropna()
        self._columns = numeric_cols

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(combined)

        self._model = TVAESynthesizer(
            metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        self._model.fit(combined)
        return self

    def generate(self, n: int, seed: int = 42) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Not fitted")
        return self._model.sample(n)


# --- Benchmark Runner ---


def get_default_methods() -> list:
    """Return all available synthesis methods with reasonable defaults."""
    methods = [
        QRFMethod(n_estimators=100),
        ZIQRFMethod(n_estimators=100),
        QDNNMethod(hidden_dim=64, epochs=50),
        ZIQDNNMethod(hidden_dim=64, epochs=50),
        MAFMethod(n_layers=4, hidden_dim=32, epochs=50),
        ZIMAFMethod(n_layers=4, hidden_dim=32, epochs=50),
    ]

    # Add SDV methods if available
    try:
        import sdv
        methods.extend([
            CTGANMethod(epochs=300),
            TVAEMethod(epochs=300),
        ])
    except ImportError:
        pass

    return methods


class BenchmarkRunner:
    """Run synthesis method comparison benchmark.

    Trains each method on the same train/holdout split, generates synthetic
    data, and computes PRDC metrics against holdouts from each source.

    Usage:
        runner = BenchmarkRunner()
        result = runner.run(
            sources={"CPS": cps_df, "SIPP": sipp_df},
            shared_cols=["age", "is_male"],
        )
        print(result.summary())
    """

    def __init__(self, methods: list = None):
        self.methods = methods if methods is not None else get_default_methods()

    def run(
        self,
        sources: dict[str, pd.DataFrame],
        shared_cols: list[str],
        holdout_frac: float = 0.2,
        n_generate: int = None,
        k: int = 5,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run benchmark on all methods.

        Args:
            sources: name -> DataFrame for each source
            shared_cols: Columns present across sources
            holdout_frac: Fraction to hold out for evaluation
            n_generate: Records to generate per method (default: sum of holdouts)
            k: Neighbors for PRDC
            seed: Random seed for reproducible splits

        Returns:
            BenchmarkResult with per-method PRDC
        """
        rng = np.random.RandomState(seed)

        # Create consistent train/holdout splits
        train_sources = {}
        holdouts = {}
        for name, df in sources.items():
            n = len(df)
            n_holdout = max(int(n * holdout_frac), k + 2)
            perm = rng.permutation(n)
            holdouts[name] = df.iloc[perm[:n_holdout]].reset_index(drop=True)
            train_sources[name] = df.iloc[perm[n_holdout:]].reset_index(drop=True)

        if n_generate is None:
            n_generate = sum(len(h) for h in holdouts.values())

        method_results = []

        for method in self.methods:
            print(f"\n--- {method.name} ---")
            t0 = time.time()

            try:
                method.fit(train_sources, shared_cols)
                synthetic = method.generate(n=n_generate, seed=seed)
                elapsed = time.time() - t0

                # Evaluate per source
                source_results = []
                for name, holdout in holdouts.items():
                    eval_cols = [
                        c for c in holdout.columns
                        if c in synthetic.columns
                        and holdout[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                    ]
                    if len(eval_cols) < 1:
                        continue

                    holdout_vals = holdout[eval_cols].values.astype(float)
                    synth_vals = synthetic[eval_cols].dropna().values.astype(float)

                    hold_mask = ~np.isnan(holdout_vals).any(axis=1)
                    synth_mask = ~np.isnan(synth_vals).any(axis=1)
                    holdout_clean = holdout_vals[hold_mask]
                    synth_clean = synth_vals[synth_mask]

                    if len(holdout_clean) < k + 2 or len(synth_clean) < k + 2:
                        continue

                    prdc = _compute_prdc(holdout_clean, synth_clean, k=k)

                    source_results.append(SourceResult(
                        source_name=name,
                        precision=prdc["precision"],
                        recall=prdc["recall"],
                        density=prdc["density"],
                        coverage=prdc["coverage"],
                        n_holdout=len(holdout_clean),
                        n_synthetic=len(synth_clean),
                    ))

                    print(f"  {name}: coverage={prdc['coverage']:.1%} "
                          f"precision={prdc['precision']:.1%}")

                mr = MethodResult(
                    method_name=method.name,
                    source_results=source_results,
                    elapsed_seconds=elapsed,
                )
                method_results.append(mr)
                print(f"  Mean coverage: {mr.mean_coverage:.1%} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  ERROR: {e}")
                method_results.append(MethodResult(
                    method_name=method.name,
                    source_results=[],
                    elapsed_seconds=time.time() - t0,
                ))

        return BenchmarkResult(
            method_results=method_results,
            holdout_frac=holdout_frac,
            n_generate=n_generate,
            k=k,
            seed=seed,
        )
