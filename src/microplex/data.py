"""
Data loading utilities for microplex.

Provides functions for loading CPS ASEC microdata and preparing it
for use with HierarchicalSynthesizer.

Example:
    >>> from microplex.data import load_cps_asec
    >>> households, persons = load_cps_asec()
    >>> from microplex import HierarchicalSynthesizer
    >>> synth = HierarchicalSynthesizer()
    >>> synth.fit(households, persons)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


# Default data directory (relative to package root)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_cps_asec(
    data_dir: Optional[Union[str, Path]] = None,
    households_only: bool = False,
    persons_only: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load CPS ASEC microdata.

    Loads pre-processed household and person level data from parquet files.
    If data files don't exist, raises an error with instructions to download.

    Args:
        data_dir: Directory containing parquet files (default: package data/)
        households_only: If True, only return household DataFrame
        persons_only: If True, only return person DataFrame

    Returns:
        (households, persons) DataFrames, or single DataFrame if *_only=True

    Raises:
        FileNotFoundError: If data files don't exist

    Example:
        >>> households, persons = load_cps_asec()
        >>> print(f"Loaded {len(households)} households, {len(persons)} persons")
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)

    hh_path = data_dir / "cps_asec_households.parquet"
    person_path = data_dir / "cps_asec_persons.parquet"

    # Check if files exist
    if not hh_path.exists() or not person_path.exists():
        raise FileNotFoundError(
            f"CPS ASEC data files not found in {data_dir}.\n"
            "Run the download script first:\n"
            "  python scripts/download_cps_asec.py\n"
            "Or generate sample data:\n"
            "  python scripts/download_cps_asec.py --sample"
        )

    if households_only:
        return pd.read_parquet(hh_path)

    if persons_only:
        return pd.read_parquet(person_path)

    households = pd.read_parquet(hh_path)
    persons = pd.read_parquet(person_path)

    return households, persons


def load_cps_for_synthesis(
    data_dir: Optional[Union[str, Path]] = None,
    sample_fraction: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare CPS ASEC data for HierarchicalSynthesizer.

    This function loads the data and ensures all required columns exist
    with proper types for the synthesizer.

    Args:
        data_dir: Directory containing parquet files
        sample_fraction: If provided, sample this fraction of households
        random_state: Random seed for sampling

    Returns:
        (households, persons) tuple ready for HierarchicalSynthesizer.fit()

    Example:
        >>> from microplex import HierarchicalSynthesizer
        >>> from microplex.data import load_cps_for_synthesis
        >>> hh, persons = load_cps_for_synthesis(sample_fraction=0.1)
        >>> synth = HierarchicalSynthesizer()
        >>> synth.fit(hh, persons, epochs=50)
    """
    households, persons = load_cps_asec(data_dir)

    # Ensure required columns exist with proper types
    households = _prepare_household_data(households)
    persons = _prepare_person_data(persons)

    # Sample if requested
    if sample_fraction is not None and 0 < sample_fraction < 1:
        np.random.seed(random_state)
        sampled_hh_ids = np.random.choice(
            households["household_id"].unique(),
            size=int(len(households) * sample_fraction),
            replace=False,
        )
        households = households[households["household_id"].isin(sampled_hh_ids)]
        persons = persons[persons["household_id"].isin(sampled_hh_ids)]

    return households, persons


def _prepare_household_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare household data for synthesis.

    Ensures all required columns exist with proper types.
    """
    result = df.copy()

    # Required columns with defaults
    required_cols = {
        "household_id": lambda: np.arange(len(result)),
        "n_persons": lambda: np.ones(len(result)),
        "n_adults": lambda: np.ones(len(result)),
        "n_children": lambda: np.zeros(len(result)),
        "state_fips": lambda: np.zeros(len(result)),
        "tenure": lambda: np.ones(len(result)),
        "hh_weight": lambda: np.ones(len(result)),
    }

    for col, default_fn in required_cols.items():
        if col not in result.columns:
            result[col] = default_fn()

    # Ensure numeric types
    for col in ["n_persons", "n_adults", "n_children", "state_fips", "tenure"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(int)

    for col in ["hh_weight"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(1).astype(float)

    # Ensure at least 1 person per household
    result["n_persons"] = result["n_persons"].clip(lower=1)
    result["n_adults"] = result["n_adults"].clip(lower=1)

    return result


def _prepare_person_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare person data for synthesis.

    Ensures all required columns exist with proper types.
    """
    result = df.copy()

    # Required columns with defaults
    required_cols = {
        "person_id": lambda: np.arange(len(result)),
        "household_id": lambda: np.zeros(len(result), dtype=int),
        "age": lambda: np.full(len(result), 30),
        "sex": lambda: np.ones(len(result)),
        "income": lambda: np.zeros(len(result)),
        "employment_status": lambda: np.zeros(len(result)),
        "education": lambda: np.ones(len(result)),
        "relationship_to_head": lambda: np.ones(len(result)),
    }

    for col, default_fn in required_cols.items():
        if col not in result.columns:
            result[col] = default_fn()

    # Ensure numeric types
    for col in ["age", "sex", "employment_status", "education", "relationship_to_head"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(int)

    for col in ["income"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(float)

    # Clip age to reasonable range
    result["age"] = result["age"].clip(0, 120)

    return result


def create_sample_data(
    n_households: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create synthetic sample data for testing.

    Generates realistic-looking household and person data based on
    CPS ASEC distributions. Useful for testing when actual CPS data
    is not available.

    Args:
        n_households: Number of households to generate
        seed: Random seed for reproducibility

    Returns:
        (households, persons) tuple

    Example:
        >>> hh, persons = create_sample_data(n_households=5000)
        >>> print(f"Generated {len(hh)} households with {len(persons)} persons")
    """
    np.random.seed(seed)

    # Generate household composition based on CPS ASEC distributions
    # Source: 2024 CPS ASEC household size distribution
    n_persons = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7],
        n_households,
        p=[0.28, 0.34, 0.16, 0.12, 0.06, 0.03, 0.01],
    )

    households = pd.DataFrame({
        "household_id": np.arange(n_households),
        "n_persons": n_persons,
        # State distribution roughly proportional to population
        "state_fips": np.random.choice(
            [6, 48, 12, 36, 42, 17, 39, 13, 37, 26, 4, 34, 51, 53, 25,
             47, 29, 18, 55, 21, 24, 41, 8, 22, 5, 28, 20, 31, 35, 23],
            n_households,
        ),
        "tenure": np.random.choice([1, 2, 3], n_households, p=[0.65, 0.34, 0.01]),
        "hh_weight": np.random.lognormal(8, 0.5, n_households),
    })

    # Derive n_adults and n_children from n_persons
    households["n_children"] = np.minimum(
        np.random.binomial(households["n_persons"], 0.25),
        households["n_persons"] - 1,  # At least 1 adult
    )
    households["n_adults"] = households["n_persons"] - households["n_children"]

    # Generate persons
    persons_list = []
    person_id = 0

    for _, hh in households.iterrows():
        hh_id = hh["household_id"]
        n_adults_hh = int(hh["n_adults"])
        n_children_hh = int(hh["n_children"])

        # Generate adults
        for i in range(n_adults_hh):
            age = np.random.randint(18, 85)
            education = np.random.choice([1, 2, 3, 4], p=[0.10, 0.28, 0.30, 0.32])

            # Income model: log-normal base with age and education factors
            if np.random.random() < 0.15:  # 15% zero income
                income = 0
            else:
                base_income = np.random.lognormal(10.5, 1.0)
                age_factor = 1 + 0.02 * min(age - 18, 30) - 0.01 * max(age - 55, 0)
                edu_factor = 1 + 0.3 * education
                income = base_income * age_factor * edu_factor

            persons_list.append({
                "person_id": person_id,
                "household_id": hh_id,
                "age": age,
                "sex": np.random.choice([1, 2]),  # 1=Male, 2=Female
                "income": max(0, income),
                "employment_status": np.random.choice([0, 1, 2], p=[0.35, 0.60, 0.05]),
                "education": education,
                "relationship_to_head": 1 if i == 0 else (2 if i == 1 else 3),
            })
            person_id += 1

        # Generate children
        for i in range(n_children_hh):
            persons_list.append({
                "person_id": person_id,
                "household_id": hh_id,
                "age": np.random.randint(0, 18),
                "sex": np.random.choice([1, 2]),
                "income": 0,
                "employment_status": 0,
                "education": 1,
                "relationship_to_head": 4,  # Child
            })
            person_id += 1

    persons = pd.DataFrame(persons_list)

    return households, persons


def get_data_info(data_dir: Optional[Union[str, Path]] = None) -> dict:
    """
    Get information about available CPS ASEC data files.

    Returns:
        Dictionary with file info (exists, size, record count, columns)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    else:
        data_dir = Path(data_dir)

    info = {
        "data_dir": str(data_dir),
        "households": {"exists": False},
        "persons": {"exists": False},
    }

    hh_path = data_dir / "cps_asec_households.parquet"
    person_path = data_dir / "cps_asec_persons.parquet"

    if hh_path.exists():
        hh = pd.read_parquet(hh_path)
        info["households"] = {
            "exists": True,
            "path": str(hh_path),
            "size_mb": hh_path.stat().st_size / 1e6,
            "n_records": len(hh),
            "columns": list(hh.columns),
        }

    if person_path.exists():
        persons = pd.read_parquet(person_path)
        info["persons"] = {
            "exists": True,
            "path": str(person_path),
            "size_mb": person_path.stat().st_size / 1e6,
            "n_records": len(persons),
            "columns": list(persons.columns),
        }

    return info
