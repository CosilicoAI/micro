"""
Microplex Calibration Targets Database

Maps calibration targets from authoritative sources to RAC variables.
Provides parity with PolicyEngine and other microsimulation frameworks.
"""

from microplex.targets.database import TargetsDatabase, Target, TargetCategory
from microplex.targets.loaders import (
    load_soi_targets,
    load_snap_targets,
    load_medicaid_targets,
    load_eitc_targets,
    load_ctc_targets,
    load_ssi_targets,
    load_tanf_targets,
    load_housing_targets,
    load_aca_targets,
    load_wic_targets,
    load_other_benefit_targets,
    load_demographics_targets,
    load_all_targets,
)
from microplex.targets.rac_mapping import RAC_VARIABLE_MAP

__all__ = [
    "TargetsDatabase",
    "Target",
    "TargetCategory",
    "load_soi_targets",
    "load_snap_targets",
    "load_medicaid_targets",
    "load_eitc_targets",
    "load_ctc_targets",
    "load_ssi_targets",
    "load_tanf_targets",
    "load_housing_targets",
    "load_aca_targets",
    "load_wic_targets",
    "load_other_benefit_targets",
    "load_demographics_targets",
    "load_all_targets",
    "RAC_VARIABLE_MAP",
]
