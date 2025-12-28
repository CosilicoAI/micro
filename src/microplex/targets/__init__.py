"""
Microplex Calibration Targets Framework

General-purpose framework for calibration targets.
Country-specific targets and loaders are in microplex-sources.

Classes:
    Target: A single calibration target
    TargetCategory: Categories of targets (income, benefits, demographics)
    TargetsDatabase: Collection of targets with indexing

RAC Mapping:
    RACVariable: Variable definition linked to statute
    RAC_VARIABLE_MAP: Mapping from variable names to RAC definitions
"""

from microplex.targets.database import TargetsDatabase, Target, TargetCategory
from microplex.targets.rac_mapping import (
    RACVariable,
    RAC_VARIABLE_MAP,
    POLICYENGINE_TO_RAC,
    MICRODATA_TO_RAC,
    get_rac_for_target,
    get_rac_for_pe_variable,
    get_rac_for_microdata_column,
)

__all__ = [
    # Core classes
    "TargetsDatabase",
    "Target",
    "TargetCategory",
    # RAC mapping
    "RACVariable",
    "RAC_VARIABLE_MAP",
    "POLICYENGINE_TO_RAC",
    "MICRODATA_TO_RAC",
    "get_rac_for_target",
    "get_rac_for_pe_variable",
    "get_rac_for_microdata_column",
]
