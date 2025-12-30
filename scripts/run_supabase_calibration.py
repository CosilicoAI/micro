#!/usr/bin/env python3
"""
Run calibration using targets from Supabase.

Loads PE calibration targets from Supabase and calibrates CPS microdata
to match official Census/IRS/agency targets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

# Direct import to avoid torch dependency
import importlib.util

# Import calibration module
cal_path = Path(__file__).parent.parent / "src" / "microplex" / "calibration.py"
spec = importlib.util.spec_from_file_location("calibration", cal_path)
cal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal_module)
Calibrator = cal_module.Calibrator

# Import supabase_targets module
sb_path = Path(__file__).parent.parent / "src" / "microplex" / "supabase_targets.py"
spec = importlib.util.spec_from_file_location("supabase_targets", sb_path)
sb_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb_module)
SupabaseTargetLoader = sb_module.SupabaseTargetLoader


# State FIPS to abbreviation
FIPS_TO_STATE = {
    1: "al", 2: "ak", 4: "az", 5: "ar", 6: "ca", 8: "co", 9: "ct", 10: "de",
    11: "dc", 12: "fl", 13: "ga", 15: "hi", 16: "id", 17: "il", 18: "in",
    19: "ia", 20: "ks", 21: "ky", 22: "la", 23: "me", 24: "md", 25: "ma",
    26: "mi", 27: "mn", 28: "ms", 29: "mo", 30: "mt", 31: "ne", 32: "nv",
    33: "nh", 34: "nj", 35: "nm", 36: "ny", 37: "nc", 38: "nd", 39: "oh",
    40: "ok", 41: "or", 42: "pa", 44: "ri", 45: "sc", 46: "sd", 47: "tn",
    48: "tx", 49: "ut", 50: "vt", 51: "va", 53: "wa", 54: "wv", 55: "wi",
    56: "wy",
}


def load_cps_data(data_dir: Path) -> pd.DataFrame:
    """Load enhanced CPS person-level data."""
    print("=" * 70)
    print("LOADING CPS DATA")
    print("=" * 70)

    df = pd.read_parquet(data_dir / "cps_enhanced_persons.parquet")
    print(f"Records: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Filter territories
    territory_fips = {3, 7, 14, 43, 52}
    df = df[~df['state_fips'].isin(territory_fips)].copy()
    print(f"After filtering territories: {len(df):,}")

    # Add state abbreviation
    df['state'] = df['state_fips'].map(FIPS_TO_STATE)

    return df


def load_supabase_targets() -> dict:
    """Load calibration targets from Supabase."""
    print("\n" + "=" * 70)
    print("LOADING TARGETS FROM SUPABASE")
    print("=" * 70)

    loader = SupabaseTargetLoader()

    # Get summary
    summary = loader.get_summary()
    print(f"Total targets in Supabase: {summary['total']:,}")
    print("\nBy institution:")
    for inst, count in sorted(summary['by_institution'].items(), key=lambda x: -x[1]):
        print(f"  {inst}: {count:,}")

    # Build calibration constraints
    constraints = loader.build_calibration_constraints(period=2024)
    print(f"\nMapped to {len(constraints)} CPS constraints")

    return constraints


def aggregate_to_persons(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare person-level data for calibration."""
    print("\n" + "=" * 70)
    print("PREPARING PERSON-LEVEL DATA")
    print("=" * 70)

    # Use person weight as the calibration weight
    df = df.copy()
    df['weight'] = df['person_weight']

    # Convert boolean benefits to binary
    for col in ['snap', 'tanf', 'wic', 'medicaid']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)

    print(f"Persons: {len(df):,}")
    print(f"Total weighted population: {df['weight'].sum():,.0f}")

    return df


def filter_targets_to_data(targets: dict, df: pd.DataFrame) -> dict:
    """Keep only targets that match available CPS columns."""
    available = set(df.columns)
    filtered = {}

    for var, value in targets.items():
        if var in available:
            filtered[var] = value
        else:
            # Check if it's a state-level target we need to build
            pass  # For now, skip state-level

    print(f"\nFiltered to {len(filtered)} targets matching CPS columns")

    if filtered:
        print("\nTarget variables:")
        for var, value in sorted(filtered.items()):
            if value > 1e12:
                print(f"  {var}: ${value/1e12:.2f}T")
            elif value > 1e9:
                print(f"  {var}: ${value/1e9:.2f}B")
            elif value > 1e6:
                print(f"  {var}: ${value/1e6:.2f}M")
            else:
                print(f"  {var}: {value:,.0f}")

    return filtered


def run_calibration(df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """Run IPF calibration with the targets."""
    print("\n" + "=" * 70)
    print("RUNNING CALIBRATION")
    print("=" * 70)

    if not targets:
        print("ERROR: No targets available for calibration")
        return df

    print(f"Calibrating with {len(targets)} continuous targets")

    calibrator = Calibrator(
        method="ipf",
        max_iter=200,
        tol=1e-6,
    )

    calibrator.fit(
        df,
        marginal_targets={},
        continuous_targets=targets,
        weight_col="weight",
    )

    df = df.copy()
    df['calibrated_weight'] = calibrator.weights_

    print(f"\nCalibration complete!")
    print(f"  Converged: {calibrator.converged_}")
    print(f"  Iterations: {calibrator.n_iterations_}")

    # Weight adjustment statistics
    weight_ratio = df['calibrated_weight'] / df['weight']
    print(f"\nWeight adjustment statistics:")
    print(f"  Min ratio: {weight_ratio.min():.4f}")
    print(f"  Max ratio: {weight_ratio.max():.4f}")
    print(f"  Mean ratio: {weight_ratio.mean():.4f}")
    print(f"  Std ratio: {weight_ratio.std():.4f}")

    return df


def validate_and_report(df: pd.DataFrame, targets: dict):
    """Validate calibration results and report errors."""
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    if 'calibrated_weight' not in df.columns:
        print("No calibrated weights - skipping validation")
        return

    results = []
    for var, target in sorted(targets.items()):
        # Before calibration
        before = (df[var] * df['weight']).sum()
        before_error = abs(before - target) / target * 100 if target != 0 else 0

        # After calibration
        after = (df[var] * df['calibrated_weight']).sum()
        after_error = abs(after - target) / target * 100 if target != 0 else 0

        results.append({
            'variable': var,
            'target': target,
            'before': before,
            'after': after,
            'before_error': before_error,
            'after_error': after_error,
            'improvement': before_error - after_error,
        })

    results_df = pd.DataFrame(results)

    print("\nPer-target results:")
    print("-" * 90)
    print(f"{'Variable':<30} {'Target':>15} {'Before Error':>12} {'After Error':>12} {'Improvement':>12}")
    print("-" * 90)

    for _, row in results_df.iterrows():
        target_str = f"${row['target']/1e9:.1f}B" if row['target'] > 1e9 else f"{row['target']:,.0f}"
        print(f"{row['variable']:<30} {target_str:>15} {row['before_error']:>11.2f}% {row['after_error']:>11.2f}% {row['improvement']:>11.2f}%")

    print("-" * 90)

    # Summary
    print("\nSummary:")
    print(f"  Targets: {len(results)}")
    print(f"  Before calibration - Mean error: {results_df['before_error'].mean():.2f}%")
    print(f"  After calibration  - Mean error: {results_df['after_error'].mean():.2f}%")
    print(f"  Mean improvement: {results_df['improvement'].mean():.2f}%")

    # Best and worst
    best = results_df.loc[results_df['after_error'].idxmin()]
    worst = results_df.loc[results_df['after_error'].idxmax()]
    print(f"\n  Best target: {best['variable']} ({best['after_error']:.4f}% error)")
    print(f"  Worst target: {worst['variable']} ({worst['after_error']:.2f}% error)")


def main():
    """Run full calibration pipeline using Supabase targets."""
    print("=" * 70)
    print("MICROPLEX CALIBRATION WITH SUPABASE TARGETS")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"

    # Step 1: Load CPS data
    df = load_cps_data(data_dir)

    # Step 2: Load targets from Supabase
    targets = load_supabase_targets()

    # Step 3: Prepare person-level data
    df = aggregate_to_persons(df)

    # Step 4: Filter targets to available columns
    targets = filter_targets_to_data(targets, df)

    # Step 5: Run calibration
    df = run_calibration(df, targets)

    # Step 6: Validate
    validate_and_report(df, targets)

    # Step 7: Summary
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)

    if 'calibrated_weight' in df.columns:
        print(f"\nFinal statistics:")
        print(f"  Persons: {len(df):,}")
        print(f"  Original weighted pop: {df['weight'].sum():,.0f}")
        print(f"  Calibrated weighted pop: {df['calibrated_weight'].sum():,.0f}")
        print(f"  Targets matched: {len(targets)}")

        # Save output
        output_path = data_dir / "cps_supabase_calibrated.parquet"
        df.to_parquet(output_path, index=False)
        print(f"\nSaved to {output_path}")

    return df, targets


if __name__ == "__main__":
    df, targets = main()
