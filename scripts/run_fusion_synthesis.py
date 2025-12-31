"""
Multi-survey fusion synthesis: CPS + PUF → Complete synthetic population.

Pipeline:
1. Load CPS ASEC (demographics, household structure, geography, some income)
2. Load IRS PUF (detailed income, capital gains, deductions)
3. Harmonize both to common schema
4. Stack surveys with missing value mask
5. Train Masked MAF on stacked data
6. Generate complete synthetic population
7. (Optional) Calibrate to IRS SOI targets

Usage:
    python scripts/run_fusion_synthesis.py --n-synthetic 200000 --device mps
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_cps_for_fusion(year: int = 2023) -> pd.DataFrame:
    """Load CPS and convert to pandas with harmonization-ready columns."""
    from microplex.data_sources.cps import load_cps_asec

    print(f"\n1. Loading CPS ASEC {year}...")
    dataset = load_cps_asec(year=year, download=True)

    # Convert polars to pandas
    persons = dataset.persons.to_pandas()
    households = dataset.households.to_pandas()

    # Merge person + household
    df = persons.merge(
        households[["household_id", "state_fips", "household_weight"]],
        on="household_id",
        how="left",
    )

    # Rename to match common schema
    rename_map = {
        "wage_income": "employment_income",
        "self_employment_income": "self_employment_income",
        "interest_income": "interest_income",
        "dividend_income": "dividend_income",
        "rental_income": "rental_income",
        "social_security": "social_security",
        "unemployment_compensation": "unemployment_compensation",
        "weight": "person_weight",
    }
    df = df.rename(columns=rename_map)

    # Add derived columns for harmonization
    if "sex" in df.columns:
        df["is_male"] = (df["sex"] == 1).astype(float)

    if "marital_status" in df.columns:
        # Married includes married-spouse present (1) and married-spouse absent (2)
        df["is_married"] = df["marital_status"].isin([1, 2]).astype(float)

    # Use person weight
    df["weight"] = df["person_weight"]

    print(f"   Loaded {len(df):,} person records")
    print(f"   Weighted population: {df['weight'].sum():,.0f}")

    return df


def load_puf_for_fusion(target_year: int = 2024) -> pd.DataFrame:
    """Load PUF with uprating to target year."""
    from microplex.data_sources.puf import load_puf

    print(f"\n2. Loading IRS PUF (uprated to {target_year})...")
    try:
        df = load_puf(target_year=target_year, expand_persons=True)
        print(f"   Loaded {len(df):,} person-level records from PUF")
        print(f"   Weighted filers: {df['weight'].sum():,.0f}")
        return df
    except Exception as e:
        print(f"   Warning: Could not load PUF: {e}")
        print("   Continuing with CPS only...")
        return None


def run_fusion_synthesis(
    n_synthetic: int = 200_000,
    cps_year: int = 2023,
    puf_target_year: int = 2024,
    n_layers: int = 6,
    hidden_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    output_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run full multi-survey fusion synthesis pipeline.

    Args:
        n_synthetic: Number of synthetic records to generate
        cps_year: Year of CPS ASEC to use
        puf_target_year: Year to uprate PUF to
        n_layers: Number of MAF layers
        hidden_dim: Hidden dimension in MADE networks
        epochs: Training epochs
        batch_size: Training batch size
        lr: Learning rate
        device: Device for training (cpu, cuda, mps)
        output_path: Path to save synthetic population
        model_path: Path to save trained model
        verbose: Print progress

    Returns:
        Synthetic population DataFrame
    """
    from microplex.fusion.harmonize import (
        harmonize_surveys,
        stack_surveys,
        COMMON_SCHEMA,
    )
    from microplex.fusion.masked_maf import (
        fit_masked_maf,
        generate_complete_population,
    )

    start_time = time.time()

    if verbose:
        print("=" * 70)
        print("MULTI-SURVEY FUSION SYNTHESIS")
        print("=" * 70)

    # Step 1: Load CPS
    cps = load_cps_for_fusion(year=cps_year)

    # Step 2: Load PUF (optional - may require HuggingFace access)
    puf = load_puf_for_fusion(target_year=puf_target_year)

    # Step 3: Harmonize surveys
    if verbose:
        print("\n3. Harmonizing surveys to common schema...")

    surveys = {"cps": cps}
    if puf is not None:
        surveys["puf"] = puf

    harmonized = harmonize_surveys(surveys)

    # Step 4: Stack surveys with mask
    if verbose:
        print("\n4. Stacking surveys with observation mask...")

    stacked, mask = stack_surveys(harmonized, normalize_weights=True)

    # Report survey composition
    if verbose:
        survey_counts = stacked["_survey"].value_counts()
        print("\n   Survey composition:")
        for survey, count in survey_counts.items():
            pct = 100 * count / len(stacked)
            print(f"     {survey}: {count:,} ({pct:.1f}%)")

    # Get variable names
    variable_names = list(COMMON_SCHEMA.keys())

    # Step 5: Train Masked MAF
    if verbose:
        print(f"\n5. Training Masked MAF ({n_layers} layers, {hidden_dim} hidden)...")
        print(f"   Device: {device}")

    model = fit_masked_maf(
        stacked=stacked,
        mask=mask,
        variable_names=variable_names,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        verbose=verbose,
    )

    # Save model if path provided
    if model_path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        if verbose:
            print(f"   Saved model to {model_path}.pkl")

    # Step 6: Generate synthetic population
    if verbose:
        print(f"\n6. Generating {n_synthetic:,} synthetic records...")

    synthetic = generate_complete_population(
        model=model,
        n_samples=n_synthetic,
        variable_names=variable_names,
        clip_z=3.0,
        device=device,
    )

    # Add uniform weights (to be calibrated later)
    synthetic["weight"] = 1.0

    # Report summary statistics
    if verbose:
        print("\n   Synthetic population summary:")
        print(f"     Records: {len(synthetic):,}")

        # Compare key variables
        print("\n   Variable comparison (Original vs Synthetic):")
        key_vars = [
            "age",
            "employment_income",
            "self_employment_income",
            "long_term_capital_gains",
            "partnership_s_corp_income",
        ]
        for var in key_vars:
            if var in synthetic.columns and var in stacked.columns:
                orig_mean = stacked[var].dropna().mean()
                synth_mean = synthetic[var].mean()
                if orig_mean != 0:
                    pct_diff = 100 * (synth_mean - orig_mean) / abs(orig_mean)
                    print(f"     {var}: orig={orig_mean:,.0f}, synth={synth_mean:,.0f} ({pct_diff:+.1f}%)")

    # Save output if path provided
    if output_path:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        synthetic.to_parquet(output_path)
        if verbose:
            print(f"\n   Saved synthetic population to {output_path}")

    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "=" * 70)
        print(f"FUSION SYNTHESIS COMPLETE ({elapsed:.1f}s)")
        print("=" * 70)

    return synthetic


def main():
    parser = argparse.ArgumentParser(
        description="Multi-survey fusion synthesis: CPS + PUF → synthetic population"
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=200_000,
        help="Number of synthetic records to generate",
    )
    parser.add_argument(
        "--cps-year",
        type=int,
        default=2023,
        help="Year of CPS ASEC to use",
    )
    parser.add_argument(
        "--puf-target-year",
        type=int,
        default=2024,
        help="Year to uprate PUF to",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        help="Number of MAF layers",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension in MADE networks",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file path",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save trained model (without extension)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else DATA_DIR / "fusion_synthetic.parquet"
    model_path = Path(args.save_model) if args.save_model else MODELS_DIR / "masked_maf_fusion"

    run_fusion_synthesis(
        n_synthetic=args.n_synthetic,
        cps_year=args.cps_year,
        puf_target_year=args.puf_target_year,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_path=output_path,
        model_path=model_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
