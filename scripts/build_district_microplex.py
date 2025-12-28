"""
Build District-Level Microplex: Synthetic tax units calibrated to geographic targets.

Uses normalizing flows to synthesize records that can be assigned to counties/districts,
then calibrates weights to match district-level IRS SOI and Census targets.

Pipeline:
1. Load calibrated CPS tax units (seed population with state-level calibration)
2. Train normalizing flow on income/tax distributions conditional on demographics
3. Generate synthetic records for each district, conditioned on district demographics
4. Calibrate weights to match:
   - State population totals (from seed calibration)
   - County/district AGI distributions
   - Tax credit amounts (EITC, CTC)

Usage:
    python scripts/build_district_microplex.py --n-per-district 1000 --target-sparsity 0.9
"""

import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys

# Add parent for microplex imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from microplex import SparseCalibrator, ConditionalMAF


# State FIPS codes
STATE_FIPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


def load_seed_data(data_source_path: Path) -> pd.DataFrame:
    """Load calibrated tax unit data from cosilico-data-sources."""
    parquet_path = data_source_path / "tax_units_calibrated_gradient_2024.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Calibrated tax units not found at {parquet_path}. "
            "Run gradient_calibrate.py first."
        )

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} tax units from {parquet_path.name}")
    return df


def load_district_targets(data_source_path: Path) -> Dict:
    """Load district-level targets from IRS SOI and Census."""
    targets_dir = data_source_path / "data" / "targets"

    targets = {}

    # State demographics (contains county data if available)
    demo_path = targets_dir / "state_demographics.parquet"
    if demo_path.exists():
        targets["demographics"] = pd.read_parquet(demo_path)

    # State income distribution
    income_path = targets_dir / "state_income_distribution.parquet"
    if income_path.exists():
        targets["income"] = pd.read_parquet(income_path)

    # State tax credits
    credits_path = targets_dir / "state_tax_credits.parquet"
    if credits_path.exists():
        targets["credits"] = pd.read_parquet(credits_path)

    return targets


def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Prepare data for normalizing flow training.

    Continuous vars (to model): income components, ages
    Conditioning vars: state, filing status, num dependents
    """
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

    # Conditioning variables
    condition_vars = [
        "state_fips",
        "filing_status",
        "num_dependents",
    ]

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


def train_synthesizer(
    df: pd.DataFrame,
    hidden_dim: int = 128,
    n_layers: int = 4,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> ConditionalMAF:
    """Train normalizing flow on tax unit data."""
    X, C, cont_vars, cond_vars = prepare_training_data(df)

    if verbose:
        print(f"\nTraining ConditionalMAF:")
        print(f"  Continuous vars: {cont_vars}")
        print(f"  Condition shape: {C.shape if C is not None else 'None'}")
        print(f"  Data shape: {X.shape}")

    # Log-transform income variables (they're highly skewed)
    X_log = X.copy()
    X_log[:, :6] = np.log1p(np.maximum(X_log[:, :6], 0))  # First 6 are incomes

    # Standardize for numerical stability
    X_mean = X_log.mean(axis=0)
    X_std = X_log.std(axis=0) + 1e-6
    X_normalized = (X_log - X_mean) / X_std

    # Initialize flow
    n_features = X.shape[1]
    n_context = C.shape[1] if C is not None else 0

    maf = ConditionalMAF(
        n_features=n_features,
        n_context=n_context,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )

    # Train the flow
    if verbose:
        print(f"  Training for {epochs} epochs on {device}...")

    maf.fit(
        X_normalized, C,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        verbose=verbose,
        verbose_freq=max(1, epochs // 10),
    )

    # Store normalization params for generation
    maf._X_mean = X_mean
    maf._X_std = X_std
    maf._training_context = C
    maf._cont_vars = cont_vars
    maf._cond_vars = cond_vars
    maf._log_transform_cols = list(range(6))

    return maf


def synthesize_for_district(
    maf: ConditionalMAF,
    seed_df: pd.DataFrame,
    district_id: str,
    n_records: int,
    district_demographics: Optional[Dict] = None,
    seed: int = 42,
    use_flow: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Generate synthetic records for a specific district.

    Uses trained normalizing flow to generate income distributions
    conditioned on district demographics.

    Args:
        maf: Trained ConditionalMAF
        seed_df: Seed data for demographics
        district_id: FIPS code for district
        n_records: Number of records to generate
        district_demographics: Optional override demographics
        seed: Random seed
        use_flow: If True, use flow; if False, use bootstrap
        device: Device for flow generation
    """
    np.random.seed(seed)

    # Get state from district (first 2 digits of FIPS)
    if len(district_id) >= 2:
        state_fips = int(district_id[:2])
    else:
        state_fips = int(district_id)

    # Filter to records from same state for context sampling
    state_mask = seed_df["state_fips"].astype(int) == state_fips
    state_records = seed_df[state_mask]

    if len(state_records) == 0:
        state_records = seed_df

    if use_flow and hasattr(maf, '_X_mean'):
        # Sample conditioning vectors from state's distribution
        context_indices = np.random.choice(len(maf._training_context), n_records, replace=True)

        # Filter context to state-relevant records if possible
        # For now, sample from all contexts
        context = maf._training_context[context_indices].astype(np.float32)

        # Generate from flow with tighter clipping
        X_normalized = maf.generate(context, clip_z=2.5, device=device)

        # Clip normalized values to avoid extreme outputs
        X_normalized = np.clip(X_normalized, -4, 4)

        # Denormalize
        X_log = X_normalized * maf._X_std + maf._X_mean

        # Clip log values before exp to avoid overflow (log($10M) ≈ 16)
        X_log = np.clip(X_log, -10, 18)

        # Inverse log transform for income columns
        X = X_log.copy()
        for col_idx in maf._log_transform_cols:
            X[:, col_idx] = np.expm1(np.clip(X_log[:, col_idx], -10, 16))
            X[:, col_idx] = np.clip(X[:, col_idx], 0, 1e8)  # Cap at $100M

        # Build dataframe
        synthetic = pd.DataFrame(X, columns=maf._cont_vars)

        # Add categorical columns by sampling from state records
        cat_cols = ["filing_status", "num_dependents", "num_ctc_children",
                    "num_eitc_children", "is_joint"]
        for col in cat_cols:
            if col in state_records.columns:
                synthetic[col] = state_records[col].sample(n_records, replace=True).values

        synthetic["state_fips"] = state_fips

    else:
        # Fallback to bootstrap resampling
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
        synthetic["earned_income"] = synthetic["wage_income"] + np.maximum(se_col, 0)

    # Initialize uniform weights
    synthetic["weight"] = 1.0

    return synthetic


def build_marginal_targets(
    target_data: Dict,
    districts: List[str],
    year: int = 2021,
) -> Dict:
    """Build marginal targets for calibration from loaded target data."""
    marginal_targets = {}

    # State-level targets from income distribution
    if "income" in target_data:
        income_df = target_data["income"]
        income_df = income_df[income_df["year"] == year]

        # AGI bracket targets by state
        for state_code in income_df["state_code"].unique():
            state_data = income_df[income_df["state_code"] == state_code]

            for _, row in state_data.iterrows():
                bracket = row["agi_bracket"]
                target_var = f"agi_bracket_{state_code}"

                if target_var not in marginal_targets:
                    marginal_targets[target_var] = {}

                marginal_targets[target_var][bracket] = row["target_returns"]

    # District-level population targets (if available)
    if "demographics" in target_data:
        demo_df = target_data["demographics"]
        demo_df = demo_df[demo_df["year"] == year]

        # State population totals
        marginal_targets["state"] = {}
        for _, row in demo_df.iterrows():
            state_code = row["state_code"]
            # Use household count as proxy for tax unit count
            marginal_targets["state"][state_code] = row["total_households"]

    return marginal_targets


def build_continuous_targets(
    target_data: Dict,
    districts: List[str],
    year: int = 2021,
) -> Dict:
    """Build continuous targets (totals) from loaded target data."""
    continuous_targets = {}

    if "income" in target_data:
        income_df = target_data["income"]
        income_df = income_df[income_df["year"] == year]

        # Total AGI by state
        for state_code in income_df["state_code"].unique():
            state_data = income_df[income_df["state_code"] == state_code]
            total_agi = state_data["target_agi"].sum()
            continuous_targets[f"agi_{state_code}"] = total_agi

    if "credits" in target_data:
        credits_df = target_data["credits"]
        credits_df = credits_df[credits_df["year"] == year]

        # EITC totals by state
        for _, row in credits_df.iterrows():
            state_code = row["state_code"]
            continuous_targets[f"eitc_{state_code}"] = row["eitc_amount"]
            continuous_targets[f"ctc_{state_code}"] = row["ctc_amount"]

    return continuous_targets


def calibrate_district_population(
    synthetic: pd.DataFrame,
    marginal_targets: Dict,
    continuous_targets: Dict,
    target_sparsity: float = 0.9,
    verbose: bool = True,
) -> pd.DataFrame:
    """Calibrate synthetic population to district targets."""
    calibrator = SparseCalibrator(
        target_sparsity=target_sparsity,
        max_iter=2000,
        tol=1e-6,
    )

    if verbose:
        n_cat = sum(len(v) for v in marginal_targets.values())
        n_cont = len(continuous_targets)
        print(f"Calibrating {len(synthetic):,} records to {n_cat + n_cont} targets...")

    start = time.time()
    result = calibrator.fit_transform(synthetic, marginal_targets, continuous_targets)
    elapsed = time.time() - start

    if verbose:
        val = calibrator.validate(result)
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Sparsity: {calibrator.get_sparsity():.1%}")
        print(f"  Max error: {val['max_error']:.2%}")
        print(f"  Mean error: {val['mean_error']:.2%}")

    return result


def build_district_microplex(
    data_source_path: Path,
    n_per_district: int = 1000,
    target_sparsity: float = 0.9,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build district-level microplex.

    Args:
        data_source_path: Path to cosilico-data-sources
        n_per_district: Records to synthesize per district
        target_sparsity: Target sparsity for calibration
        output_path: Where to save results
        verbose: Print progress
    """
    if verbose:
        print("=" * 60)
        print("BUILDING DISTRICT MICROPLEX")
        print("=" * 60)

    # Step 1: Load seed data
    if verbose:
        print("\n1. Loading seed data...")
    seed_df = load_seed_data(data_source_path)

    # Step 2: Load targets
    if verbose:
        print("\n2. Loading district targets...")
    target_data = load_district_targets(data_source_path)
    for name, df in target_data.items():
        if verbose:
            print(f"   {name}: {len(df)} rows")

    # Step 3: Train synthesizer (or use bootstrap fallback)
    if verbose:
        print("\n3. Training synthesizer...")
    maf = train_synthesizer(seed_df, epochs=50, verbose=verbose)

    # Step 4: Get list of districts (states for now, counties later)
    states = sorted(seed_df["state_fips"].dropna().unique().astype(int))
    districts = [f"{s:02d}" for s in states]
    if verbose:
        print(f"\n4. Synthesizing for {len(districts)} districts...")

    # Step 5: Synthesize records per district
    all_synthetic = []
    for i, district_id in enumerate(districts):
        synthetic = synthesize_for_district(
            maf, seed_df, district_id, n_per_district, seed=42+i
        )
        all_synthetic.append(synthetic)
        if verbose and (i + 1) % 10 == 0:
            print(f"   Generated {i+1}/{len(districts)} districts")

    combined = pd.concat(all_synthetic, ignore_index=True)
    if verbose:
        print(f"   Total synthetic records: {len(combined):,}")

    # Step 6: Build calibration targets
    if verbose:
        print("\n5. Building calibration targets...")

    # For now, use simple state-level targets
    marginal_targets = {"state_fips": {}}
    for state in states:
        mask = seed_df["state_fips"].astype(int) == state
        target_count = seed_df.loc[mask, "weight"].sum()
        marginal_targets["state_fips"][state] = float(target_count)

    # Income totals
    continuous_targets = {
        "wage_income": float((seed_df["wage_income"] * seed_df["weight"]).sum()),
        "total_income": float((seed_df["total_income"] * seed_df["weight"]).sum()),
    }

    if verbose:
        print(f"   {len(marginal_targets['state_fips'])} state targets")
        print(f"   {len(continuous_targets)} continuous targets")

    # Step 7: Calibrate
    if verbose:
        print(f"\n6. Calibrating with SparseCalibrator...")

    # Ensure state_fips is numeric for calibration
    combined["state_fips"] = combined["state_fips"].astype(int)

    calibrated = calibrate_district_population(
        combined,
        marginal_targets,
        continuous_targets,
        target_sparsity=target_sparsity,
        verbose=verbose,
    )

    # Step 8: Save
    if output_path:
        if verbose:
            print(f"\n7. Saving to {output_path}...")
        calibrated.to_parquet(output_path)

    if verbose:
        print("\n" + "=" * 60)
        print("DISTRICT MICROPLEX COMPLETE")
        print("=" * 60)
        non_zero = (calibrated["weight"] > 1e-9).sum()
        print(f"Total records: {len(calibrated):,}")
        print(f"Non-zero weights: {non_zero:,} ({non_zero/len(calibrated):.1%})")
        print(f"Weighted population: {calibrated['weight'].sum():,.0f}")

    return calibrated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build District Microplex")
    parser.add_argument(
        "--n-per-district", type=int, default=1000,
        help="Records to synthesize per district"
    )
    parser.add_argument(
        "--target-sparsity", type=float, default=0.9,
        help="Target sparsity for calibration"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output parquet path"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Find cosilico-data-sources
    script_dir = Path(__file__).parent
    data_source_path = script_dir.parent.parent / "cosilico-data-sources"

    if not data_source_path.exists():
        print(f"Error: cosilico-data-sources not found at {data_source_path}")
        exit(1)

    output_path = Path(args.output) if args.output else None

    build_district_microplex(
        data_source_path=data_source_path,
        n_per_district=args.n_per_district,
        target_sparsity=args.target_sparsity,
        output_path=output_path,
        verbose=not args.quiet,
    )
