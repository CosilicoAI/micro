"""
District Microplex Builder.

Generates synthetic tax units at geographic granularity (states, counties,
congressional districts) and calibrates weights to match local targets.

Pipeline:
1. Fit a normalizing flow on seed population (calibrated CPS/PUF)
2. Generate synthetic records conditioned on district demographics
3. Calibrate weights to match district-level targets (IRS SOI, Census)

Example:
    >>> from microplex.district import DistrictMicroplex, build_targets_from_database
    >>> dm = DistrictMicroplex(n_per_district=1000, target_sparsity=0.9)
    >>> dm.fit(seed_data, epochs=100)
    >>> marginal, continuous = build_targets_from_database(year=2021)
    >>> result = dm.build(seed_data, districts=["06", "36"], ...)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from microplex import SparseCalibrator, ConditionalMAF
from microplex.targets import load_all_state_targets, TargetCategory


# State FIPS codes (int -> abbreviation)
STATE_FIPS: Dict[int, str] = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


# Required columns for seed data
REQUIRED_COLUMNS = ["state_fips", "wage_income", "head_age", "weight"]


def build_targets_from_database(
    year: int = 2021,
    verbose: bool = False,
) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    """
    Build calibration targets from the microplex targets database.

    Loads state-level targets and organizes them into the format expected
    by SparseCalibrator.

    Args:
        year: Target year (default 2021)
        verbose: Print progress information

    Returns:
        marginal_targets: Dict for categorical targets {var: {category: count}}
        continuous_targets: Dict for sum targets {var: total}
    """
    targets = load_all_state_targets(year)

    if verbose:
        print(f"   Loaded {len(targets)} state-level targets from database")

    # Build state_fips marginal targets (population-based)
    marginal_targets: Dict[str, Dict] = {"state_fips": {}}
    continuous_targets: Dict[str, float] = {}

    # Track totals for continuous targets
    total_eitc = 0.0
    total_ctc = 0.0
    total_snap = 0.0

    # Organize by state
    for t in targets:
        state = t.geography
        if state == "US":
            continue

        # Get state FIPS from state code
        state_fips = None
        for fips, code in STATE_FIPS.items():
            if code == state:
                state_fips = fips
                break

        if state_fips is None:
            continue

        # State population targets -> tax unit counts
        if t.is_count:
            if t.category == TargetCategory.HOUSEHOLD_STRUCTURE:
                if "total_households" in t.name:
                    # Use households as proxy for tax units
                    marginal_targets["state_fips"][state_fips] = t.value
            elif t.category == TargetCategory.POPULATION:
                if "total_population" in t.name and state_fips not in marginal_targets["state_fips"]:
                    # Fallback: use population / 2.5 if no household target
                    marginal_targets["state_fips"][state_fips] = t.value / 2.5

        # Continuous targets - aggregate nationally for calibration
        if not t.is_count:
            if t.category == TargetCategory.EITC and "amount" in t.name:
                total_eitc += t.value
            elif t.category == TargetCategory.CTC and "amount" in t.name:
                total_ctc += t.value
            elif t.category == TargetCategory.SNAP and "benefits" in t.name:
                total_snap += t.value

    if verbose:
        n_states = len(marginal_targets.get("state_fips", {}))
        print(f"   Built targets for {n_states} states")
        print(f"   National totals: EITC=${total_eitc/1e9:.1f}B, CTC=${total_ctc/1e9:.1f}B, SNAP=${total_snap/1e9:.1f}B")

    return marginal_targets, continuous_targets


def synthesize_district_records(
    seed_data: pd.DataFrame,
    district_id: str,
    n_records: int,
    maf: Optional[ConditionalMAF] = None,
    seed: int = 42,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Generate synthetic records for a specific district.

    Uses bootstrap resampling from state-level seed data with optional noise.
    If a trained normalizing flow is provided, uses it for synthesis.

    Args:
        seed_data: Seed population DataFrame
        district_id: FIPS code for district (e.g., "06" for California)
        n_records: Number of records to generate
        maf: Optional trained ConditionalMAF for flow-based generation
        seed: Random seed for reproducibility
        device: Device for flow generation (if using MAF)

    Returns:
        DataFrame with synthetic records for the district
    """
    np.random.seed(seed)

    # Get state from district (first 2 digits of FIPS)
    if len(district_id) >= 2:
        state_fips = int(district_id[:2])
    else:
        state_fips = int(district_id)

    # Filter to records from same state for context sampling
    state_mask = seed_data["state_fips"].astype(int) == state_fips
    state_records = seed_data[state_mask]

    if len(state_records) == 0:
        state_records = seed_data

    if maf is not None and hasattr(maf, '_X_mean'):
        # Flow-based generation
        context_indices = np.random.choice(len(maf._training_context), n_records, replace=True)
        context = maf._training_context[context_indices].astype(np.float32)

        # Generate from flow with tight clipping
        X_normalized = maf.generate(context, clip_z=2.5, device=device)
        X_normalized = np.clip(X_normalized, -4, 4)

        # Denormalize
        X_log = X_normalized * maf._X_std + maf._X_mean
        X_log = np.clip(X_log, -10, 18)

        # Inverse log transform for income columns
        X = X_log.copy()
        for col_idx in maf._log_transform_cols:
            X[:, col_idx] = np.expm1(np.clip(X_log[:, col_idx], -10, 16))
            X[:, col_idx] = np.clip(X[:, col_idx], 0, 1e8)

        synthetic = pd.DataFrame(X, columns=maf._cont_vars)

        # Add categorical columns by sampling from state records
        cat_cols = ["filing_status", "num_dependents", "num_ctc_children",
                    "num_eitc_children", "is_joint"]
        for col in cat_cols:
            if col in state_records.columns:
                synthetic[col] = state_records[col].sample(n_records, replace=True).values

        synthetic["state_fips"] = state_fips
    else:
        # Bootstrap resampling fallback
        indices = np.random.choice(len(state_records), n_records, replace=True)
        synthetic = state_records.iloc[indices].copy().reset_index(drop=True)

        # Add noise to continuous variables
        noise_cols = ["wage_income", "self_employment_income", "interest_income"]
        for col in noise_cols:
            if col in synthetic.columns:
                noise = np.random.lognormal(0, 0.1, n_records)
                synthetic[col] = synthetic[col] * noise
                synthetic[col] = np.maximum(synthetic[col], 0)

    # Assign district ID
    synthetic["district_id"] = district_id
    synthetic["tax_unit_id"] = range(n_records)
    synthetic["state_fips"] = state_fips

    # Recalculate totals
    income_cols = ["wage_income", "self_employment_income", "interest_income",
                   "dividend_income", "rental_income", "social_security_income",
                   "unemployment_compensation", "other_income"]
    existing_cols = [c for c in income_cols if c in synthetic.columns]
    if existing_cols:
        synthetic["total_income"] = synthetic[existing_cols].sum(axis=1)

    # Compute earned income for EITC
    if "wage_income" in synthetic.columns:
        se_col = synthetic.get("self_employment_income", 0)
        if isinstance(se_col, int):
            se_col = 0
        synthetic["earned_income"] = synthetic["wage_income"] + np.maximum(se_col, 0)

    # Initialize uniform weights
    synthetic["weight"] = 1.0

    return synthetic


@dataclass
class DistrictMicroplex:
    """
    District-level microplex builder.

    Generates synthetic tax units calibrated to geographic targets.

    Args:
        n_per_district: Number of records to synthesize per district
        target_sparsity: Target sparsity for calibration (0-1)
        hidden_dim: Hidden dimension for normalizing flow
        n_layers: Number of flow layers
        device: Device for training/generation ("cpu" or "cuda")

    Example:
        >>> dm = DistrictMicroplex(n_per_district=1000, target_sparsity=0.9)
        >>> dm.fit(seed_data, epochs=100)
        >>> result = dm.generate(districts=["06", "36", "48"])
        >>> calibrated = dm.calibrate(result, marginal_targets, continuous_targets)
    """

    n_per_district: int = 1000
    target_sparsity: float = 0.9
    hidden_dim: int = 128
    n_layers: int = 4
    device: str = "cpu"

    # Private attributes (not in __init__)
    _maf: Optional[ConditionalMAF] = field(default=None, repr=False)
    _seed_data: Optional[pd.DataFrame] = field(default=None, repr=False)
    _calibrator: Optional[SparseCalibrator] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize private attributes."""
        self._maf = None
        self._seed_data = None
        self._calibrator = None

    def _validate_seed_data(self, seed_data: pd.DataFrame) -> None:
        """Validate seed data has required columns."""
        if len(seed_data) == 0:
            raise ValueError("Seed data is empty")

        missing = [col for col in REQUIRED_COLUMNS if col not in seed_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _prepare_training_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Prepare data for normalizing flow training."""
        # Continuous variables to synthesize
        continuous_vars = [
            "wage_income",
            "self_employment_income",
            "interest_income",
            "dividend_income",
            "rental_income",
            "social_security_income",
            "head_age",
        ]

        # Only use columns that exist
        continuous_vars = [v for v in continuous_vars if v in df.columns]

        # Conditioning variables
        condition_vars = [
            "state_fips",
            "filing_status",
            "num_dependents",
        ]
        condition_vars = [v for v in condition_vars if v in df.columns]

        # Filter to records with valid data
        mask = df[continuous_vars].notna().all(axis=1)
        df_valid = df[mask].copy()

        # Extract arrays
        X = df_valid[continuous_vars].values.astype(np.float32)

        # Encode categorical conditioning vars
        cond_data = []
        for var in condition_vars:
            if df_valid[var].dtype == object:
                # One-hot encode
                dummies = pd.get_dummies(df_valid[var], prefix=var)
                cond_data.append(dummies.values)
            else:
                # Normalize numeric
                vals = df_valid[var].values.astype(np.float32)
                cond_data.append((vals - vals.mean()) / (vals.std() + 1e-6))

        C = np.column_stack(cond_data) if cond_data else None

        return X, C, continuous_vars, condition_vars

    def fit(
        self,
        seed_data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> "DistrictMicroplex":
        """
        Train normalizing flow on seed data.

        Args:
            seed_data: Calibrated tax unit data (e.g., from CPS/PUF)
            epochs: Training epochs
            batch_size: Batch size for training
            lr: Learning rate
            verbose: Print training progress

        Returns:
            self for method chaining
        """
        self._validate_seed_data(seed_data)
        self._seed_data = seed_data

        X, C, cont_vars, cond_vars = self._prepare_training_data(seed_data)

        if verbose:
            print(f"Training ConditionalMAF:")
            print(f"  Continuous vars: {cont_vars}")
            print(f"  Condition shape: {C.shape if C is not None else 'None'}")
            print(f"  Data shape: {X.shape}")

        # Log-transform income variables (they're highly skewed)
        X_log = X.copy()
        n_income_cols = min(6, X.shape[1] - 1)  # Last col is age
        X_log[:, :n_income_cols] = np.log1p(np.maximum(X_log[:, :n_income_cols], 0))

        # Standardize for numerical stability
        X_mean = X_log.mean(axis=0)
        X_std = X_log.std(axis=0) + 1e-6
        X_normalized = (X_log - X_mean) / X_std

        # Initialize flow
        n_features = X.shape[1]
        n_context = C.shape[1] if C is not None else 0

        self._maf = ConditionalMAF(
            n_features=n_features,
            n_context=n_context,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        )

        # Train the flow
        if verbose:
            print(f"  Training for {epochs} epochs on {self.device}...")

        self._maf.fit(
            X_normalized, C,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=self.device,
            verbose=verbose,
            verbose_freq=max(1, epochs // 10),
        )

        # Store normalization params for generation
        self._maf._X_mean = X_mean
        self._maf._X_std = X_std
        self._maf._training_context = C
        self._maf._cont_vars = cont_vars
        self._maf._cond_vars = cond_vars
        self._maf._log_transform_cols = list(range(n_income_cols))

        return self

    def generate(
        self,
        districts: List[str],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Generate synthetic records for specified districts.

        Args:
            districts: List of district FIPS codes (e.g., ["06", "36"])
            verbose: Print progress

        Returns:
            DataFrame with synthetic records for all districts
        """
        if self._seed_data is None:
            raise ValueError("Must call fit() before generate()")

        all_synthetic = []
        for i, district_id in enumerate(districts):
            synthetic = synthesize_district_records(
                seed_data=self._seed_data,
                district_id=district_id,
                n_records=self.n_per_district,
                maf=self._maf,
                seed=42 + i,
                device=self.device,
            )
            all_synthetic.append(synthetic)

            if verbose and (i + 1) % 10 == 0:
                print(f"   Generated {i+1}/{len(districts)} districts")

        combined = pd.concat(all_synthetic, ignore_index=True)

        if verbose:
            print(f"   Total synthetic records: {len(combined):,}")

        return combined

    def calibrate(
        self,
        synthetic: pd.DataFrame,
        marginal_targets: Dict[str, Dict],
        continuous_targets: Dict[str, float],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Calibrate synthetic population to targets.

        Args:
            synthetic: Synthetic population DataFrame
            marginal_targets: Categorical targets {var: {category: count}}
            continuous_targets: Sum targets {var: total}
            verbose: Print progress

        Returns:
            Calibrated DataFrame with adjusted weights
        """
        self._calibrator = SparseCalibrator(
            target_sparsity=self.target_sparsity,
            max_iter=2000,
            tol=1e-6,
        )

        if verbose:
            n_cat = sum(len(v) for v in marginal_targets.values())
            n_cont = len(continuous_targets)
            print(f"Calibrating {len(synthetic):,} records to {n_cat + n_cont} targets...")

        start = time.time()
        result = self._calibrator.fit_transform(synthetic, marginal_targets, continuous_targets)
        elapsed = time.time() - start

        if verbose:
            val = self._calibrator.validate(result)
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Sparsity: {self._calibrator.get_sparsity():.1%}")
            print(f"  Max error: {val['max_error']:.2%}")
            print(f"  Mean error: {val['mean_error']:.2%}")

        return result

    def build(
        self,
        seed_data: pd.DataFrame,
        districts: List[str],
        marginal_targets: Dict[str, Dict],
        continuous_targets: Dict[str, float],
        epochs: int = 100,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Build complete district microplex (fit, generate, calibrate).

        Args:
            seed_data: Calibrated tax unit seed data
            districts: List of district FIPS codes
            marginal_targets: Categorical calibration targets
            continuous_targets: Sum calibration targets
            epochs: Training epochs for normalizing flow
            verbose: Print progress

        Returns:
            Calibrated synthetic population DataFrame
        """
        if verbose:
            print("=" * 60)
            print("BUILDING DISTRICT MICROPLEX")
            print("=" * 60)

        # Step 1: Fit
        if verbose:
            print("\n1. Training synthesizer...")
        self.fit(seed_data, epochs=epochs, verbose=verbose)

        # Step 2: Generate
        if verbose:
            print(f"\n2. Synthesizing for {len(districts)} districts...")
        synthetic = self.generate(districts, verbose=verbose)

        # Ensure state_fips is numeric
        synthetic["state_fips"] = synthetic["state_fips"].astype(int)

        # Step 3: Calibrate
        if verbose:
            print("\n3. Calibrating weights...")
        calibrated = self.calibrate(synthetic, marginal_targets, continuous_targets, verbose=verbose)

        if verbose:
            print("\n" + "=" * 60)
            print("DISTRICT MICROPLEX COMPLETE")
            print("=" * 60)
            non_zero = (calibrated["weight"] > 1e-9).sum()
            print(f"Total records: {len(calibrated):,}")
            print(f"Non-zero weights: {non_zero:,} ({non_zero/len(calibrated):.1%})")
            print(f"Weighted population: {calibrated['weight'].sum():,.0f}")

        return calibrated


def load_seed_data(data_source_path: Path) -> pd.DataFrame:
    """Load calibrated tax unit data from cosilico-data-sources."""
    parquet_path = data_source_path / "tax_units_calibrated_gradient_2024.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Calibrated tax units not found at {parquet_path}. "
            "Run gradient_calibrate.py first."
        )

    df = pd.read_parquet(parquet_path)
    return df


def build_district_microplex(
    data_source_path: Path,
    n_per_district: int = 1000,
    target_sparsity: float = 0.9,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build district-level microplex (convenience function).

    This is a convenience wrapper that:
    1. Loads seed data from cosilico-data-sources
    2. Builds targets from the microplex targets database
    3. Trains synthesizer and generates records
    4. Calibrates to state-level targets

    Args:
        data_source_path: Path to cosilico-data-sources
        n_per_district: Records to synthesize per district
        target_sparsity: Target sparsity for calibration
        output_path: Where to save results (optional)
        verbose: Print progress

    Returns:
        Calibrated district microplex DataFrame
    """
    if verbose:
        print("\n1. Loading seed data...")
    seed_df = load_seed_data(data_source_path)
    if verbose:
        print(f"Loaded {len(seed_df):,} tax units")

    # Get districts from seed data
    states = sorted(seed_df["state_fips"].dropna().unique().astype(int))
    districts = [f"{s:02d}" for s in states]

    # Build targets from database
    if verbose:
        print("\n2. Loading targets from microplex database...")
    marginal_targets, continuous_targets = build_targets_from_database(year=2021, verbose=verbose)

    # Add income totals from seed data
    continuous_targets["wage_income"] = float((seed_df["wage_income"] * seed_df["weight"]).sum())
    if "total_income" in seed_df.columns:
        continuous_targets["total_income"] = float((seed_df["total_income"] * seed_df["weight"]).sum())

    # Fill in missing state targets from seed data
    if not marginal_targets.get("state_fips"):
        marginal_targets["state_fips"] = {}

    for state in states:
        if state not in marginal_targets["state_fips"]:
            mask = seed_df["state_fips"].astype(int) == state
            target_count = seed_df.loc[mask, "weight"].sum()
            marginal_targets["state_fips"][state] = float(target_count)

    # Build microplex
    dm = DistrictMicroplex(n_per_district=n_per_district, target_sparsity=target_sparsity)
    result = dm.build(
        seed_data=seed_df,
        districts=districts,
        marginal_targets=marginal_targets,
        continuous_targets=continuous_targets,
        epochs=50,
        verbose=verbose,
    )

    # Save if requested
    if output_path:
        if verbose:
            print(f"\nSaving to {output_path}...")
        result.to_parquet(output_path)

    return result
