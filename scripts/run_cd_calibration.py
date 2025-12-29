"""
Run calibration with congressional district targets.

Since CPS only has state FIPS (no PUMA), we use probabilistic CD assignment:
1. Assign each household to a CD based on state CD population shares
2. Create CD indicators for calibration
3. Run IPF calibration to match CD population targets

This enables CD-level analysis while acknowledging the assignment is probabilistic.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

# Import calibration
from microplex.calibration import Calibrator

# State FIPS to abbreviation
FIPS_TO_STATE = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}


def assign_cds_probabilistically(
    hh: pd.DataFrame,
    cd_probs: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """Assign each household to a congressional district.

    For each household in a state, we assign a CD based on the CD population
    shares within that state.

    Args:
        hh: Household dataframe with 'state_fips' column
        cd_probs: CD probability dataframe with 'state_fips', 'cd_id', 'prob'
        seed: Random seed for reproducibility

    Returns:
        Household dataframe with 'cd_id' column added
    """
    np.random.seed(seed)

    # Create cd_id column
    hh = hh.copy()
    hh["cd_id"] = None

    # For each state, assign CDs
    for state_fips in hh["state_fips"].unique():
        state_cds = cd_probs[cd_probs["state_fips"] == state_fips]
        if len(state_cds) == 0:
            print(f"  Warning: No CDs for state FIPS {state_fips}")
            continue

        # Get households in this state
        state_mask = hh["state_fips"] == state_fips
        n_hh = state_mask.sum()

        # Sample CDs with replacement based on probabilities
        cd_assignments = np.random.choice(
            state_cds["cd_id"].values,
            size=n_hh,
            replace=True,
            p=state_cds["prob"].values
        )

        hh.loc[state_mask, "cd_id"] = cd_assignments

    return hh


def build_cd_indicators(hh: pd.DataFrame) -> pd.DataFrame:
    """Create indicator columns for each congressional district.

    For calibration, we need a column for each CD that indicates whether
    the household is in that CD (actually, how many persons in that CD).
    """
    hh = hh.copy()

    for cd_id in hh["cd_id"].dropna().unique():
        # Indicator: n_persons if household is in this CD, 0 otherwise
        col_name = f"n_persons_{cd_id}"
        hh[col_name] = np.where(hh["cd_id"] == cd_id, hh["n_persons"], 0)

    return hh


def load_cd_targets(targets_path: Path) -> dict:
    """Load CD population targets as a dictionary."""
    targets = pd.read_parquet(targets_path)

    # Filter to CD targets (geography matches "XX-YY" or "XX-AL")
    cd_mask = targets["geography"].str.match(r"^[A-Z]{2}-\d{2}$|^[A-Z]{2}-AL$", na=False)
    cd_targets = targets[cd_mask].copy()

    # Create target dictionary: {cd_id: population}
    return dict(zip(cd_targets["geography"], cd_targets["value"]))


def main():
    data_dir = Path(__file__).parent.parent / "data"

    print("=" * 60)
    print("CD-LEVEL CALIBRATION")
    print("=" * 60)

    # 1. Load household data
    print("\n1. Loading CPS household data...")
    hh = pd.read_parquet(data_dir / "cps_asec_households.parquet")
    print(f"   Loaded {len(hh):,} households")

    # Filter out territories (not in our CD data)
    valid_states = set(FIPS_TO_STATE.keys())
    hh = hh[hh["state_fips"].isin(valid_states)].copy()
    print(f"   After filtering territories: {len(hh):,} households")

    # 2. Load CD probability mapping
    print("\n2. Loading CD probability mapping...")
    cd_probs = pd.read_parquet(data_dir / "state_cd_probabilities.parquet")
    print(f"   CDs: {len(cd_probs)}")
    print(f"   States: {cd_probs['state_fips'].nunique()}")

    # 3. Assign CDs probabilistically
    print("\n3. Assigning households to CDs...")
    hh = assign_cds_probabilistically(hh, cd_probs)
    cd_counts = hh["cd_id"].value_counts()
    print(f"   Assigned to {len(cd_counts)} CDs")
    print(f"   Households per CD: {cd_counts.mean():.1f} (mean), {cd_counts.min()}-{cd_counts.max()} (range)")

    # 4. Build CD indicators
    print("\n4. Building CD indicators...")
    hh = build_cd_indicators(hh)
    cd_cols = [c for c in hh.columns if c.startswith("n_persons_")]
    print(f"   Created {len(cd_cols)} indicator columns")

    # 5. Load CD targets
    print("\n5. Loading CD targets...")
    cd_targets_dict = load_cd_targets(data_dir / "targets.parquet")
    print(f"   Loaded {len(cd_targets_dict)} CD targets")

    # Filter to CDs that exist in our data
    valid_cds = set(hh["cd_id"].dropna().unique())
    cd_targets_dict = {k: v for k, v in cd_targets_dict.items() if k in valid_cds and v > 0}
    print(f"   Valid CD targets: {len(cd_targets_dict)}")

    # 6. Set up calibration
    print("\n6. Setting up calibration...")

    # For IPF, we need to use continuous_targets for the person-count columns
    # Format: {column_name: target_total}
    continuous_targets = {}
    for cd_id, target_pop in cd_targets_dict.items():
        col_name = f"n_persons_{cd_id}"
        if col_name in hh.columns:
            continuous_targets[col_name] = target_pop

    print(f"   Continuous targets: {len(continuous_targets)}")

    # 7. Run calibration
    print("\n7. Running IPF calibration...")

    # Use initial CPS weights as starting point
    hh["weight"] = hh["hh_weight"]

    calibrator = Calibrator(method="ipf", max_iter=500, tol=1e-6)

    # Run calibration with empty marginal targets but continuous targets
    calibrated_hh = calibrator.fit_transform(
        hh,
        marginal_targets={},  # No categorical constraints
        continuous_targets=continuous_targets,
        weight_col="weight"
    )

    # 8. Evaluate results
    print("\n8. Evaluating calibration...")
    print(f"   Converged: {calibrator.converged_}")
    print(f"   Iterations: {calibrator.n_iterations_}")

    # Check total weighted persons
    total_persons = (calibrated_hh["weight"] * calibrated_hh["n_persons"]).sum()
    print(f"   Total weighted persons: {total_persons:,.0f}")

    # Calculate errors
    errors = []
    for cd_id, target_pop in cd_targets_dict.items():
        col_name = f"n_persons_{cd_id}"
        if col_name in calibrated_hh.columns:
            calibrated_pop = (calibrated_hh["weight"] * calibrated_hh[col_name]).sum()
            error = abs(calibrated_pop - target_pop) / target_pop if target_pop > 0 else 0
            errors.append({
                "cd_id": cd_id,
                "calibrated": calibrated_pop,
                "target": target_pop,
                "error": error
            })

    errors_df = pd.DataFrame(errors)
    print(f"\n   Mean absolute error: {errors_df['error'].mean()*100:.2f}%")
    print(f"   Max error: {errors_df['error'].max()*100:.2f}%")
    print(f"   Median error: {errors_df['error'].median()*100:.2f}%")

    # Sample of CD-level results (sorted by error)
    print("\n   Top 10 CDs by error:")
    worst = errors_df.nlargest(10, "error")
    for _, row in worst.iterrows():
        print(f"   {row['cd_id']}: {row['calibrated']:,.0f} / {row['target']:,.0f} ({row['error']*100:.1f}% error)")

    print("\n   Sample best CDs:")
    best = errors_df.nsmallest(5, "error")
    for _, row in best.iterrows():
        print(f"   {row['cd_id']}: {row['calibrated']:,.0f} / {row['target']:,.0f} ({row['error']*100:.1f}% error)")

    # 9. Save results
    print("\n9. Saving results...")
    output_path = data_dir / "cps_calibrated_cd.parquet"
    calibrated_hh.to_parquet(output_path, index=False)
    print(f"   Saved to {output_path}")

    print("\n" + "=" * 60)
    print("CD CALIBRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
