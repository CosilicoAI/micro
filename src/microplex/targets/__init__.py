"""
Microplex Calibration Targets Database

Maps calibration targets from authoritative sources to RAC variables.
Provides parity with PolicyEngine and other microsimulation frameworks.

Target counts (2021):
- 241 national targets (IRS SOI, Census, SSA, Fed, admin data)
- 2,142 state-level targets (51 states × 40+ variables)
- Total: 2,383 targets across 17 categories
- 89% mapped to RAC statute references
"""

from microplex.targets.database import TargetsDatabase, Target, TargetCategory
from microplex.targets.loaders import (
    # National IRS/SOI targets
    load_soi_targets,
    load_filing_status_targets,
    load_eitc_targets,
    load_ctc_targets,
    load_aca_targets,
    # Benefit programs
    load_snap_targets,
    load_medicaid_targets,
    load_ssi_targets,
    load_tanf_targets,
    load_housing_targets,
    load_wic_targets,
    load_other_benefit_targets,
    # Social Security / Medicare
    load_social_security_targets,
    # Demographics
    load_demographics_targets,
    load_race_ethnicity_targets,
    load_education_targets,
    load_disability_targets,
    load_household_composition_targets,
    load_poverty_targets,
    load_wealth_targets,
    # Employment
    load_employment_industry_targets,
    # State-level targets
    load_state_demographics_targets,
    load_state_income_targets,
    load_state_tax_credit_targets,
    load_state_unemployment_targets,
    load_state_snap_targets,
    load_state_medicaid_targets,
    load_state_ssi_targets,
    load_state_tanf_targets,
    load_state_housing_targets,
    load_all_state_targets,
    # Combined
    load_all_targets,
)
from microplex.targets.rac_mapping import RAC_VARIABLE_MAP

__all__ = [
    # Core classes
    "TargetsDatabase",
    "Target",
    "TargetCategory",
    # National loaders
    "load_soi_targets",
    "load_filing_status_targets",
    "load_eitc_targets",
    "load_ctc_targets",
    "load_aca_targets",
    "load_snap_targets",
    "load_medicaid_targets",
    "load_ssi_targets",
    "load_tanf_targets",
    "load_housing_targets",
    "load_wic_targets",
    "load_other_benefit_targets",
    "load_social_security_targets",
    "load_demographics_targets",
    "load_race_ethnicity_targets",
    "load_education_targets",
    "load_disability_targets",
    "load_household_composition_targets",
    "load_poverty_targets",
    "load_wealth_targets",
    "load_employment_industry_targets",
    # State loaders
    "load_state_demographics_targets",
    "load_state_income_targets",
    "load_state_tax_credit_targets",
    "load_state_unemployment_targets",
    "load_state_snap_targets",
    "load_state_medicaid_targets",
    "load_state_ssi_targets",
    "load_state_tanf_targets",
    "load_state_housing_targets",
    "load_all_state_targets",
    # Combined
    "load_all_targets",
    # RAC mapping
    "RAC_VARIABLE_MAP",
]
