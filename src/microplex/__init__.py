"""
microplex: Microdata synthesis and reweighting using normalizing flows.

A library for creating rich, calibrated microdata through:
- Conditional synthesis (demographics → outcomes)
- Reweighting to population targets
- Zero-inflated distributions (common in economic/health data)
- Joint correlations between variables
- Hierarchical structures (households, firms, etc.)

Example:
    >>> from microplex import Synthesizer
    >>> synth = Synthesizer(
    ...     target_vars=["income", "expenditure"],
    ...     condition_vars=["age", "education"],
    ... )
    >>> synth.fit(training_data)
    >>> synthetic = synth.generate(new_demographics)
"""

from microplex.synthesizer import Synthesizer
from microplex.reweighting import Reweighter
from microplex.calibration import (
    Calibrator,
    SparseCalibrator,
    HardConcreteCalibrator,
)

# Default sparse calibrator: Cross-Category + IPF achieves exact target matching
# with controllable sparsity. HardConcreteCalibrator available for differentiable
# pipelines or custom loss functions.
DefaultSparseCalibrator = SparseCalibrator
from microplex.hierarchical import (
    HierarchicalSynthesizer,
    HouseholdSchema,
    prepare_cps_for_hierarchical,
)
from microplex.transforms import (
    ZeroInflatedTransform,
    LogTransform,
    Standardizer,
    VariableTransformer,
    MultiVariableTransformer,
)
from microplex.flows import ConditionalMAF, MADE, AffineCouplingLayer
from microplex.discrete import (
    BinaryModel,
    CategoricalModel,
    DiscreteModelCollection,
)
from microplex.data import (
    load_cps_asec,
    load_cps_for_synthesis,
    create_sample_data,
    get_data_info,
)
from microplex.cps_synthetic import (
    CPSSummaryStats,
    CPSSyntheticGenerator,
    validate_synthetic,
)
from microplex.geography import (
    BlockGeography,
    load_block_probabilities,
    derive_geographies,
    STATE_LEN,
    COUNTY_LEN,
    TRACT_LEN,
    BLOCK_LEN,
)
from microplex.transitions import (
    Mortality,
    DisabilityOnset,
    DisabilityRecovery,
    DisabilityTransitionModel,
    MarriageTransition,
    DivorceTransition,
)
from microplex.statmatch_backend import (
    StatMatchSynthesizer,
    create_synthesizer,
    HAS_STATMATCH,
)
from microplex.pe_targets import (
    PETargets,
    get_pe_targets,
    create_calibration_targets,
)
from microplex.unified_calibration import (
    UnifiedCalibrator,
    CalibrationTarget,
    calibrate_to_pe_targets,
)
from microplex.target_registry import (
    TargetRegistry,
    TargetSpec,
    TargetCategory,
    TargetLevel,
    TargetGroup,
    get_registry,
    print_registry_summary,
)
from microplex.calibration_harness import (
    CalibrationHarness,
    CalibrationResult,
    run_pe_parity_suite,
)

# Core data models (from cosilico-microdata merge)
from microplex.core import (
    # Entities
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
    # Variables
    DataType,
    VariableRole,
    LegalReference,
    Variable,
    VariableRegistry,
    # Periods
    PeriodType,
    Period,
    # Resolution
    ResolutionLevel,
    ResolutionConfig,
    HardConcreteGate,
    compress_dataset,
    for_browser,
    for_api,
    for_research,
)

# Data sources (Polars-based CPS loading + PUF)
from microplex.data_sources import (
    CPSDataset,
    download_cps_asec,
    load_cps_asec_polars,
    get_available_years,
    PERSON_VARIABLES,
    HOUSEHOLD_VARIABLES,
    # Mappings
    CoverageLevel,
    CoverageGap,
    VariableMapping,
    get_mapping_metadata,
    get_all_mappings,
    coverage_summary,
    # Transform
    TransformedDataset,
    transform_cps_to_cosilico,
    # PUF
    load_puf,
    download_puf,
    PUF_VARIABLE_MAP,
    UPRATING_FACTORS,
    PUF_EXCLUSIVE_VARS,
    SHARED_VARS,
)

# Fusion (multi-survey synthesis)
from microplex.fusion import (
    harmonize_surveys,
    stack_surveys,
    COMMON_SCHEMA,
    MaskedMAF,
    FusionConfig,
    FusionResult,
    FusionSynthesizer,
    synthesize_from_surveys,
)

# Validation
from microplex.validation import (
    AGI_BRACKETS,
    FILING_STATUSES,
    SOITargets,
    get_soi_years,
    load_soi_targets,
    compute_validation_metrics,
    ValidationResult,
    validate_against_soi,
    MetricComparison,
    BaselineComparison,
    compute_baseline_comparison,
    export_comparison_json,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "Synthesizer",
    "Reweighter",
    "Calibrator",
    "SparseCalibrator",
    "HardConcreteCalibrator",
    "DefaultSparseCalibrator",
    # Statistical matching (optional backend)
    "StatMatchSynthesizer",
    "create_synthesizer",
    "HAS_STATMATCH",
    # Hierarchical
    "HierarchicalSynthesizer",
    "HouseholdSchema",
    "prepare_cps_for_hierarchical",
    # CPS Synthetic
    "CPSSummaryStats",
    "CPSSyntheticGenerator",
    "validate_synthetic",
    # Data loading
    "load_cps_asec",
    "load_cps_for_synthesis",
    "create_sample_data",
    "get_data_info",
    # Geography
    "BlockGeography",
    "load_block_probabilities",
    "derive_geographies",
    "STATE_LEN",
    "COUNTY_LEN",
    "TRACT_LEN",
    "BLOCK_LEN",
    # Transforms
    "ZeroInflatedTransform",
    "LogTransform",
    "Standardizer",
    "VariableTransformer",
    "MultiVariableTransformer",
    # Flows
    "ConditionalMAF",
    "MADE",
    "AffineCouplingLayer",
    # Discrete
    "BinaryModel",
    "CategoricalModel",
    "DiscreteModelCollection",
    # Transitions
    "Mortality",
    "DisabilityOnset",
    "DisabilityRecovery",
    "DisabilityTransitionModel",
    "MarriageTransition",
    "DivorceTransition",
    # PE Parity
    "PETargets",
    "get_pe_targets",
    "create_calibration_targets",
    "UnifiedCalibrator",
    "CalibrationTarget",
    "calibrate_to_pe_targets",
    # Target Registry
    "TargetRegistry",
    "TargetSpec",
    "TargetCategory",
    "TargetLevel",
    "TargetGroup",
    "get_registry",
    "print_registry_summary",
    # Calibration Harness
    "CalibrationHarness",
    "CalibrationResult",
    "run_pe_parity_suite",
    # Core entities (from cosilico-microdata)
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
    # Data sources (Polars-based CPS + PUF)
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
    "get_mapping_metadata",
    "get_all_mappings",
    "coverage_summary",
    # Transform
    "TransformedDataset",
    "transform_cps_to_cosilico",
    # PUF
    "load_puf",
    "download_puf",
    "PUF_VARIABLE_MAP",
    "UPRATING_FACTORS",
    "PUF_EXCLUSIVE_VARS",
    "SHARED_VARS",
    # Fusion (multi-survey synthesis)
    "harmonize_surveys",
    "stack_surveys",
    "COMMON_SCHEMA",
    "MaskedMAF",
    "FusionConfig",
    "FusionResult",
    "FusionSynthesizer",
    "synthesize_from_surveys",
    # Validation
    "AGI_BRACKETS",
    "FILING_STATUSES",
    "SOITargets",
    "get_soi_years",
    "load_soi_targets",
    "compute_validation_metrics",
    "ValidationResult",
    "validate_against_soi",
    "MetricComparison",
    "BaselineComparison",
    "compute_baseline_comparison",
    "export_comparison_json",
]
