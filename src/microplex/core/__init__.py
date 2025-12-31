"""
Core data models for microplex.

This module provides the foundational data structures for microdata representation:
- Entity types (Person, TaxUnit, Household, Family, SPMUnit, Record)
- Variable definitions with legal references
- Period arithmetic
- Multi-resolution dataset generation
"""

from microplex.core.entities import (
    EntityType,
    FilingStatus,
    RecordType,
    Entity,
    Person,
    TaxUnit,
    Household,
    Family,
    SPMUnit,
    Record,
)
from microplex.core.variables import (
    DataType,
    VariableRole,
    LegalReference,
    Variable,
    VariableRegistry,
)
from microplex.core.periods import (
    PeriodType,
    Period,
)
from microplex.core.resolution import (
    ResolutionLevel,
    ResolutionConfig,
    HardConcreteGate,
    compress_dataset,
    for_browser,
    for_api,
    for_research,
)

__all__ = [
    # Entities
    "EntityType",
    "FilingStatus",
    "RecordType",
    "Entity",
    "Person",
    "TaxUnit",
    "Household",
    "Family",
    "SPMUnit",
    "Record",
    # Variables
    "DataType",
    "VariableRole",
    "LegalReference",
    "Variable",
    "VariableRegistry",
    # Periods
    "PeriodType",
    "Period",
    # Resolution
    "ResolutionLevel",
    "ResolutionConfig",
    "HardConcreteGate",
    "compress_dataset",
    "for_browser",
    "for_api",
    "for_research",
]
