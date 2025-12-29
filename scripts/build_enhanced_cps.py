#!/usr/bin/env python3
"""Build enhanced CPS microdata with full income/benefit columns from PE-US."""

import pandas as pd
import numpy as np
from pathlib import Path
from policyengine_us import Microsimulation


def build_enhanced_cps(year: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build enhanced CPS person and household data from PE-US.

    Returns:
        Tuple of (persons_df, households_df)
    """
    print(f"Loading PE-US microsimulation for {year}...")
    sim = Microsimulation()

    # Person-level variables
    person_vars = [
        # IDs and weights
        'person_id', 'household_id', 'tax_unit_id', 'spm_unit_id',
        'person_weight',

        # Demographics
        'age', 'is_male', 'race', 'is_hispanic',
        'marital_status', 'is_disabled',

        # Employment
        'employment_income', 'self_employment_income',
        'is_employed', 'hours_worked_per_week',

        # Investment income
        'dividend_income', 'interest_income', 'rental_income',
        'short_term_capital_gains', 'long_term_capital_gains',

        # Retirement income
        'social_security', 'pension_income',
        'taxable_pension_income', 'tax_exempt_pension_income',

        # Other income
        'ssi', 'unemployment_compensation', 'alimony_income',
        'farm_income', 'partnership_s_corp_income',

        # Benefits
        'snap', 'ssi', 'tanf', 'wic',
        'medicaid', 'medicare',

        # Tax credits
        'eitc', 'ctc', 'cdcc',

        # Geography
        'state_fips',
    ]

    # Household-level variables
    household_vars = [
        'household_id', 'household_weight',
        'household_size', 'household_income',
        'state_fips',
    ]

    # Load person data
    print("Loading person variables...")
    person_data = {}
    for var in person_vars:
        try:
            vals = sim.calculate(var, year)
            person_data[var] = vals
        except Exception as e:
            print(f"  Warning: {var} not available: {e}")

    persons_df = pd.DataFrame(person_data)
    print(f"  Persons: {len(persons_df):,} rows, {len(persons_df.columns)} columns")

    # Load household data
    print("Loading household variables...")
    household_data = {}
    for var in household_vars:
        try:
            vals = sim.calculate(var, year)
            household_data[var] = vals
        except Exception as e:
            print(f"  Warning: {var} not available: {e}")

    households_df = pd.DataFrame(household_data)
    print(f"  Households: {len(households_df):,} rows, {len(households_df.columns)} columns")

    return persons_df, households_df


def compute_calibration_totals(persons_df: pd.DataFrame) -> dict:
    """Compute weighted totals for calibration targets.

    Returns:
        Dict of target_name -> (computed_value, pe_target_value, error_pct)
    """
    from microplex.pe_targets import PETargets

    pe = PETargets()
    pe_national = pe.get_national_targets()

    # Weight column
    weight = persons_df.get('person_weight', pd.Series([1] * len(persons_df)))

    # Mapping of PE target names to our columns
    income_map = {
        'employment_income': 'employment_income',
        'self_employment_income': 'self_employment_income',
        'social_security': 'social_security',
        'dividend_income': 'dividend_income',
        'interest_income': 'interest_income',
        'rental_income': 'rental_income',
        'pension_income': 'pension_income',
        'ssi': 'ssi',
        'unemployment_compensation': 'unemployment_compensation',
    }

    results = {}

    for pe_name, col_name in income_map.items():
        if col_name in persons_df.columns:
            computed = (persons_df[col_name] * weight).sum()

            # Find PE target
            pe_row = pe_national[pe_national['name'] == pe_name]
            if not pe_row.empty:
                target = pe_row.iloc[0]['value']
                error = abs(computed - target) / target * 100
                results[pe_name] = {
                    'computed': computed,
                    'target': target,
                    'error_pct': error
                }

    return results


def main():
    # Build enhanced CPS
    persons_df, households_df = build_enhanced_cps(2024)

    # Save to parquet
    out_dir = Path("data")
    persons_df.to_parquet(out_dir / "cps_enhanced_persons.parquet", index=False)
    households_df.to_parquet(out_dir / "cps_enhanced_households.parquet", index=False)

    print(f"\n✅ Saved enhanced CPS data")
    print(f"   Persons: {out_dir / 'cps_enhanced_persons.parquet'}")
    print(f"   Households: {out_dir / 'cps_enhanced_households.parquet'}")

    # Compute and compare calibration totals
    print("\n=== CALIBRATION COMPARISON ===")
    results = compute_calibration_totals(persons_df)

    print(f"\n{'Variable':<30} {'Computed':>15} {'Target':>15} {'Error':>10}")
    print("-" * 75)

    for name, vals in sorted(results.items(), key=lambda x: -x[1]['target']):
        computed = vals['computed']
        target = vals['target']
        error = vals['error_pct']

        comp_str = f"${computed/1e9:.1f}B"
        tgt_str = f"${target/1e9:.1f}B"
        err_str = f"{error:.1f}%"

        print(f"{name:<30} {comp_str:>15} {tgt_str:>15} {err_str:>10}")


if __name__ == "__main__":
    main()
