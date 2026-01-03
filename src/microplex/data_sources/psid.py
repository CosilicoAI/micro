"""PSID (Panel Study of Income Dynamics) data source for microplex.

Provides loading, processing, and transition rate extraction from PSID panel data.
PSID is the longest-running longitudinal household survey in the world (1968-present),
making it ideal for calibrating demographic transition models.

Key features:
- Load PSID panel data via the `psid` Python package
- Extract empirical transition rates (marriage, divorce, etc.)
- Calibrate microplex transition models with PSID data
- Use as a source in MultiSourceFusion for coverage evaluation

Example:
    >>> from microplex.data_sources.psid import load_psid_panel, extract_transition_rates
    >>>
    >>> # Load PSID data
    >>> dataset = load_psid_panel(data_dir="./psid_data")
    >>>
    >>> # Extract transition rates for model calibration
    >>> transitions = psid.get_household_transitions(dataset.panel)
    >>> rates = extract_transition_rates(transitions)
    >>>
    >>> # Calibrate marriage model
    >>> from microplex.transitions import MarriageTransition
    >>> marriage_rates = calibrate_marriage_rates(rates["marriage_by_age"])
    >>> model = MarriageTransition(base_rates={"male": 0.05, "female": 0.06})
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Variable mapping from PSID names to microplex conventions
PSID_TO_MICROPLEX_VARS = {
    # Demographics
    "age": "age",
    "sex": "is_male",  # Will need transformation (1=male in PSID)
    "marital_status": "marital_status",
    "education": "education",
    "race": "race",

    # Income
    "total_family_income": "total_income",
    "head_labor_income": "head_labor_income",
    "wife_labor_income": "spouse_labor_income",

    # Wealth
    "total_wealth": "total_wealth",

    # Identifiers
    "person_id": "person_id",
    "interview_number": "household_id",
    "year": "year",
    "relationship": "relationship",
}


@dataclass
class PSIDDataset:
    """Container for PSID panel data.

    Attributes:
        persons: DataFrame with person-year observations
        source: Data source identifier (path or "mock")
        panel: Optional Panel object from psid package
    """

    persons: pd.DataFrame
    source: str
    panel: Optional["psid.Panel"] = None  # Forward reference

    @property
    def n_persons(self) -> int:
        """Number of unique persons in dataset."""
        if "person_id" in self.persons.columns:
            return self.persons["person_id"].nunique()
        return 0

    @property
    def n_observations(self) -> int:
        """Total number of person-year observations."""
        return len(self.persons)

    @property
    def years(self) -> List[int]:
        """List of years in the dataset."""
        if "year" in self.persons.columns:
            return sorted(self.persons["year"].unique().tolist())
        return []

    def summary(self) -> Dict:
        """Return summary statistics."""
        return {
            "n_persons": self.n_persons,
            "n_observations": self.n_observations,
            "years": self.years,
            "source": self.source,
        }


def load_psid_panel(
    data_dir: Union[str, Path],
    years: Optional[List[int]] = None,
    family_vars: Optional[Dict[str, str]] = None,
    individual_vars: Optional[Dict[str, str]] = None,
    sample: Optional[str] = None,
) -> PSIDDataset:
    """Load PSID panel data using the psid package.

    Args:
        data_dir: Directory containing PSID data files
        years: List of survey years to include (None = all available)
        family_vars: Dict mapping variable names to crosswalk lookups
        individual_vars: Dict mapping individual-level variables
        sample: Sample type filter ("SRC", "SEO", "IMMIGRANT", or None for all)

    Returns:
        PSIDDataset with loaded panel data

    Raises:
        FileNotFoundError: If data_dir doesn't exist
        ValueError: If psid package is not installed
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"PSID data directory not found: {data_dir}")

    try:
        import psid
    except ImportError:
        raise ValueError(
            "psid package not installed. Install with: pip install psid"
        )

    # Build panel using psid package
    panel = psid.build_panel(
        data_dir=str(data_dir),
        years=years,
        family_vars=family_vars,
        individual_vars=individual_vars,
        sample=sample,
    )

    # Convert to DataFrame with microplex variable names
    df = panel.data.copy()

    # Rename columns to microplex conventions
    rename_map = {}
    for psid_name, microplex_name in PSID_TO_MICROPLEX_VARS.items():
        if psid_name in df.columns:
            rename_map[psid_name] = microplex_name

    df = df.rename(columns=rename_map)

    # Transform sex to is_male boolean
    if "is_male" in df.columns:
        df["is_male"] = df["is_male"] == 1

    return PSIDDataset(
        persons=df,
        source=str(data_dir),
        panel=panel,
    )


def extract_transition_rates(
    transitions_df: pd.DataFrame,
    transition_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Extract overall transition rates from PSID transition data.

    Args:
        transitions_df: DataFrame from psid.get_household_transitions()
        transition_types: Types to extract (None = all available)

    Returns:
        Dict mapping transition type to annual probability
    """
    if "type" not in transitions_df.columns:
        raise ValueError("transitions_df must have 'type' column")

    total = len(transitions_df)
    if total == 0:
        return {}

    counts = transitions_df["type"].value_counts()

    if transition_types is None:
        transition_types = counts.index.tolist()

    rates = {}
    for t_type in transition_types:
        if t_type in counts.index:
            rates[t_type] = counts[t_type] / total
        else:
            rates[t_type] = 0.0

    return rates


def get_age_specific_rates(
    transitions_df: pd.DataFrame,
    transition_type: str,
    age_bins: List[Tuple[int, int]],
    age_col: str = "age_from",
) -> Dict[Tuple[int, int], float]:
    """Extract age-specific transition rates.

    Args:
        transitions_df: DataFrame from psid.get_household_transitions()
        transition_type: Type of transition (e.g., "marriage", "divorce")
        age_bins: List of (age_min, age_max) tuples
        age_col: Column name for age

    Returns:
        Dict mapping age range to rate
    """
    if age_col not in transitions_df.columns:
        return {}

    rates = {}

    for age_min, age_max in age_bins:
        # Filter to age bin
        mask = (transitions_df[age_col] >= age_min) & (transitions_df[age_col] <= age_max)
        bin_data = transitions_df[mask]

        if len(bin_data) == 0:
            rates[(age_min, age_max)] = 0.0
            continue

        # Count transitions of specified type
        type_count = (bin_data["type"] == transition_type).sum()
        rates[(age_min, age_max)] = type_count / len(bin_data)

    return rates


def calibrate_marriage_rates(
    psid_rates: Dict[Tuple[int, int], float],
    gender_adjustment: Optional[Dict[str, float]] = None,
) -> Dict[Tuple[int, int], float]:
    """Convert PSID-derived rates to MarriageTransition format.

    Args:
        psid_rates: Dict from get_age_specific_rates() for marriage
        gender_adjustment: Optional {"male": factor, "female": factor}

    Returns:
        Dict compatible with MarriageTransition base_rates
    """
    # PSID rates are already in the right format: (age_min, age_max) -> rate
    calibrated = {}

    for age_range, rate in psid_rates.items():
        # Ensure rate is a valid probability
        calibrated[age_range] = float(np.clip(rate, 0.0, 1.0))

    return calibrated


def calibrate_divorce_rates(
    psid_rates: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """Convert PSID-derived rates to DivorceTransition format.

    Args:
        psid_rates: Dict from get_age_specific_rates() for divorce

    Returns:
        Dict compatible with DivorceTransition age_effects
    """
    # Same format as marriage rates
    calibrated = {}

    for age_range, rate in psid_rates.items():
        calibrated[age_range] = float(np.clip(rate, 0.0, 1.0))

    return calibrated


def create_psid_fusion_source(
    dataset: PSIDDataset,
    source_vars: Optional[List[str]] = None,
) -> Dict:
    """Create configuration for adding PSID to MultiSourceFusion.

    Args:
        dataset: PSIDDataset from load_psid_panel()
        source_vars: Variables to include (None = common set)

    Returns:
        Dict with parameters for fusion.add_source()
    """
    if source_vars is None:
        # Default to variables commonly available in PSID
        source_vars = ["age", "total_income"]

        # Add others if present
        optional = ["is_male", "marital_status", "education", "total_wealth"]
        for var in optional:
            if var in dataset.persons.columns:
                source_vars.append(var)

    # Determine number of periods per person
    if "year" in dataset.persons.columns and "person_id" in dataset.persons.columns:
        periods_per_person = dataset.persons.groupby("person_id")["year"].nunique()
        n_periods = int(periods_per_person.median())
    else:
        n_periods = 1

    return {
        "name": "psid",
        "data": dataset.persons,
        "source_vars": source_vars,
        "n_periods": n_periods,
        "person_id_col": "person_id",
        "period_col": "year",
    }
