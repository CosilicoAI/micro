#!/usr/bin/env python3
"""
Full Hierarchical Calibration Pipeline

Runs the complete microplex pipeline on real CPS data:
1. Load CPS households and persons
2. Aggregate person-level features to household level
3. Build state-level and demographic targets
4. Run IPF calibration
5. Propagate weights to persons
6. Validate against targets
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from microplex.calibration import Calibrator


def load_cps_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CPS household and person data."""
    print("=" * 70)
    print("LOADING CPS DATA")
    print("=" * 70)

    hh = pd.read_parquet(data_dir / "cps_asec_households.parquet")
    persons = pd.read_parquet(data_dir / "cps_asec_persons.parquet")

    print(f"Households: {len(hh):,}")
    print(f"Persons: {len(persons):,}")
    print(f"Avg HH size: {len(persons) / len(hh):.2f}")
    print(f"Total weighted HH: {hh['hh_weight'].sum():,.0f}")

    # Show state distribution
    print("\nState distribution (top 10):")
    state_counts = hh.groupby('state_fips')['hh_weight'].sum().sort_values(ascending=False)
    for state, count in state_counts.head(10).items():
        print(f"  {state}: {count:,.0f}")

    return hh, persons


def aggregate_person_features(
    hh: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate person-level features to household level for calibration."""
    print("\n" + "=" * 70)
    print("AGGREGATING PERSON FEATURES TO HOUSEHOLD LEVEL")
    print("=" * 70)

    hh = hh.copy()

    # Age groups
    age_groups = [
        ("n_age_0_17", (0, 18)),
        ("n_age_18_64", (18, 65)),
        ("n_age_65_plus", (65, 200)),
    ]

    for col_name, (age_min, age_max) in age_groups:
        mask = (persons["age"] >= age_min) & (persons["age"] < age_max)
        counts = persons[mask].groupby("household_id").size()
        hh[col_name] = hh["household_id"].map(counts).fillna(0).astype(int)
        print(f"  {col_name}: {hh[col_name].sum():,} persons across {(hh[col_name] > 0).sum():,} HHs")

    # Employment status
    employed = persons[persons["employment_status"] == 1]
    employed_per_hh = employed.groupby("household_id").size()
    hh["n_employed"] = hh["household_id"].map(employed_per_hh).fillna(0).astype(int)
    print(f"  n_employed: {hh['n_employed'].sum():,} persons across {(hh['n_employed'] > 0).sum():,} HHs")

    # Total income per household
    income_per_hh = persons.groupby("household_id")["income"].sum()
    hh["hh_income"] = hh["household_id"].map(income_per_hh).fillna(0)
    print(f"  hh_income: ${hh['hh_income'].sum():,.0f} total")

    return hh


def build_targets(hh: pd.DataFrame) -> dict:
    """Build calibration targets based on actual weighted totals.

    In production, these would come from administrative data (Census, IRS, etc.)
    For demo purposes, we use the original survey weights as targets.
    """
    print("\n" + "=" * 70)
    print("BUILDING CALIBRATION TARGETS")
    print("=" * 70)

    # State-level household targets
    state_targets = {}
    state_weighted = hh.groupby("state_fips")["hh_weight"].sum()
    for state, weight in state_weighted.items():
        state_targets[state] = weight

    print(f"State targets: {len(state_targets)} states")
    print(f"  Total: {sum(state_targets.values()):,.0f}")

    # Tenure targets (1=own, 2=rent)
    tenure_targets = {}
    tenure_weighted = hh.groupby("tenure")["hh_weight"].sum()
    for tenure, weight in tenure_weighted.items():
        tenure_targets[tenure] = weight

    print(f"Tenure targets: {tenure_targets}")

    return {
        "state_fips": state_targets,
        "tenure": tenure_targets,
    }


def build_continuous_targets(hh: pd.DataFrame) -> dict:
    """Build continuous calibration targets (sums)."""
    # Person counts by age group
    targets = {}
    for col in ["n_age_0_17", "n_age_18_64", "n_age_65_plus", "n_employed"]:
        if col in hh.columns:
            targets[col] = (hh[col] * hh["hh_weight"]).sum()
            print(f"  {col}: {targets[col]:,.0f}")

    # Total income
    targets["hh_income"] = (hh["hh_income"] * hh["hh_weight"]).sum()
    print(f"  hh_income: ${targets['hh_income']:,.0f}")

    return targets


def run_calibration(
    hh: pd.DataFrame,
    marginal_targets: dict,
    continuous_targets: dict,
) -> pd.DataFrame:
    """Run IPF calibration on household data."""
    print("\n" + "=" * 70)
    print("RUNNING IPF CALIBRATION")
    print("=" * 70)

    # Initialize with original weights
    hh = hh.copy()
    hh["weight"] = hh["hh_weight"]

    calibrator = Calibrator(
        method="ipf",
        max_iter=100,
        tol=1e-6,
    )

    print(f"Marginal constraints: {len(marginal_targets)} dimensions")
    for dim, targets in marginal_targets.items():
        print(f"  {dim}: {len(targets)} categories")

    print(f"Continuous constraints: {len(continuous_targets)}")

    calibrator.fit(
        hh,
        marginal_targets=marginal_targets,
        continuous_targets=continuous_targets,
        weight_col="weight",
    )

    hh["calibrated_weight"] = calibrator.weights_

    print(f"\nCalibration complete!")
    print(f"  Converged: {calibrator.converged_}")
    print(f"  Iterations: {calibrator.n_iterations_}")

    # Weight statistics
    weight_ratio = hh["calibrated_weight"] / hh["weight"]
    print(f"\nWeight adjustment statistics:")
    print(f"  Min ratio: {weight_ratio.min():.4f}")
    print(f"  Max ratio: {weight_ratio.max():.4f}")
    print(f"  Mean ratio: {weight_ratio.mean():.4f}")
    print(f"  Std ratio: {weight_ratio.std():.4f}")

    return hh


def propagate_weights_to_persons(
    hh: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    """Propagate household weights to all persons."""
    print("\n" + "=" * 70)
    print("PROPAGATING WEIGHTS TO PERSONS")
    print("=" * 70)

    persons = persons.copy()
    weight_map = hh.set_index("household_id")["calibrated_weight"]
    persons["weight"] = persons["household_id"].map(weight_map)

    print(f"Total weighted persons: {persons['weight'].sum():,.0f}")

    return persons


def validate_calibration(
    hh: pd.DataFrame,
    marginal_targets: dict,
    continuous_targets: dict,
) -> dict:
    """Validate calibrated weights against targets."""
    print("\n" + "=" * 70)
    print("VALIDATING CALIBRATION")
    print("=" * 70)

    results = {}

    # Check marginal targets
    print("\nMarginal target validation:")
    for dim, targets in marginal_targets.items():
        dim_results = {}
        actual = hh.groupby(dim)["calibrated_weight"].sum()

        total_error = 0
        for cat, target in targets.items():
            actual_val = actual.get(cat, 0)
            error = abs(actual_val - target) / target if target > 0 else 0
            dim_results[cat] = {
                "target": target,
                "actual": actual_val,
                "error_pct": error * 100,
            }
            total_error += error

        avg_error = total_error / len(targets) * 100
        print(f"  {dim}: avg error = {avg_error:.4f}%")
        results[dim] = dim_results

    # Check continuous targets
    print("\nContinuous target validation:")
    for var, target in continuous_targets.items():
        actual = (hh[var] * hh["calibrated_weight"]).sum()
        error = abs(actual - target) / target * 100 if target > 0 else 0
        print(f"  {var}: target={target:,.0f}, actual={actual:,.0f}, error={error:.4f}%")
        results[var] = {"target": target, "actual": actual, "error_pct": error}

    return results


def save_outputs(
    hh: pd.DataFrame,
    persons: pd.DataFrame,
    output_dir: Path,
):
    """Save calibrated outputs."""
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    hh_path = output_dir / "microplex_hh.parquet"
    persons_path = output_dir / "microplex_persons.parquet"

    hh.to_parquet(hh_path, index=False)
    persons.to_parquet(persons_path, index=False)

    print(f"  Saved {len(hh):,} households to {hh_path}")
    print(f"  Saved {len(persons):,} persons to {persons_path}")


def main():
    """Run full hierarchical calibration pipeline."""
    print("=" * 70)
    print("MICROPLEX HIERARCHICAL CALIBRATION PIPELINE")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data"

    # Step 1: Load data
    hh, persons = load_cps_data(data_dir)

    # Step 2: Aggregate person features to household level
    hh = aggregate_person_features(hh, persons)

    # Step 3: Build targets
    marginal_targets = build_targets(hh)
    continuous_targets = build_continuous_targets(hh)

    # Step 4: Run calibration
    hh = run_calibration(hh, marginal_targets, continuous_targets)

    # Step 5: Propagate weights to persons
    persons = propagate_weights_to_persons(hh, persons)

    # Step 6: Validate
    validation = validate_calibration(hh, marginal_targets, continuous_targets)

    # Step 7: Save outputs
    save_outputs(hh, persons, data_dir)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nFinal statistics:")
    print(f"  Households: {len(hh):,}")
    print(f"  Persons: {len(persons):,}")
    print(f"  Total weighted HHs: {hh['calibrated_weight'].sum():,.0f}")
    print(f"  Total weighted persons: {persons['weight'].sum():,.0f}")

    return hh, persons, validation


if __name__ == "__main__":
    hh, persons, validation = main()
