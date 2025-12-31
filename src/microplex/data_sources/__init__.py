"""
Data sources for microplex.

This module provides loaders for various microdata sources:
- CPS ASEC (Census Bureau's primary income/poverty survey)
- CPS to Cosilico variable mappings with legal references
- Data transformation utilities
"""

from microplex.data_sources.cps import (
    CPSDataset,
    download_cps_asec,
    load_cps_asec as load_cps_asec_polars,
    get_available_years,
    PERSON_VARIABLES,
    HOUSEHOLD_VARIABLES,
)
from microplex.data_sources.cps_mappings import (
    CoverageLevel,
    CoverageGap,
    VariableMapping,
    map_age,
    map_earned_income,
    map_filing_status,
    map_is_blind,
    map_is_dependent,
    map_ctc_qualifying_children,
    map_agi_proxy,
    map_household_size,
    get_mapping_metadata,
    get_all_mappings,
    coverage_summary,
)
from microplex.data_sources.cps_transform import (
    TransformedDataset,
    transform_cps_to_cosilico,
)

__all__ = [
    # CPS loading
    "CPSDataset",
    "download_cps_asec",
    "load_cps_asec_polars",
    "get_available_years",
    "PERSON_VARIABLES",
    "HOUSEHOLD_VARIABLES",
    # Mappings
    "CoverageLevel",
    "CoverageGap",
    "VariableMapping",
    "map_age",
    "map_earned_income",
    "map_filing_status",
    "map_is_blind",
    "map_is_dependent",
    "map_ctc_qualifying_children",
    "map_agi_proxy",
    "map_household_size",
    "get_mapping_metadata",
    "get_all_mappings",
    "coverage_summary",
    # Transform
    "TransformedDataset",
    "transform_cps_to_cosilico",
]
