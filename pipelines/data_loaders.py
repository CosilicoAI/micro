"""Data loaders for US survey data.

Loads and harmonizes CPS, PUF, and SIPP for multi-survey fusion.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Try to import HuggingFace hub for downloading data
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Data paths
COSILICO_DATA = Path("/Users/maxghenis/CosilicoAI/cosilico-data-sources")
STORAGE_FOLDER = COSILICO_DATA / "storage"


# Variable harmonization mapping
# Maps survey-specific variable names to common names
VARIABLE_MAPPING = {
    # Demographics
    "age": {
        "cps": "age",
        "puf": "age",
        "sipp": "TAGE",
    },
    "is_male": {
        "cps": "sex",  # 1=male, 2=female in CPS
        "puf": "is_male",
        "sipp": "ESEX",  # 1=male, 2=female
    },
    "state_fips": {
        "cps": "state_fips",
        "puf": None,  # PUF doesn't have state
        "sipp": None,
    },
    "marital_status": {
        "cps": "marital_status",
        "puf": "filing_status",
        "sipp": "EMS",
    },
    # Income
    "wage_income": {
        "cps": "wage_salary_income",
        "puf": "employment_income",
        "sipp": "TPTOTINC",  # Total person income
    },
    "self_employment_income": {
        "cps": "self_employment_income",
        "puf": "self_employment_income",
        "sipp": None,
    },
    "interest_income": {
        "cps": "interest_income",
        "puf": "taxable_interest_income",
        "sipp": None,
    },
    "dividend_income": {
        "cps": "dividend_income",
        "puf": "qualified_dividend_income",
        "sipp": None,
    },
    "social_security_income": {
        "cps": "social_security_income",
        "puf": "social_security",
        "sipp": None,
    },
    "unemployment_compensation": {
        "cps": "unemployment_compensation",
        "puf": "taxable_unemployment_compensation",
        "sipp": None,
    },
    # PUF-specific income (detailed tax items)
    "rental_income": {
        "cps": None,
        "puf": "rental_income",
        "sipp": None,
    },
    "capital_gains": {
        "cps": None,
        "puf": "long_term_capital_gains",
        "sipp": None,
    },
    # SIPP-specific
    "tip_income": {
        "cps": None,
        "puf": None,
        "sipp": "tip_income",  # Derived column
    },
}


def load_cps(
    path: Optional[Path] = None,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Load CPS data."""
    if path is None:
        path = COSILICO_DATA / "micro/us/cps_2024.parquet"

    print(f"Loading CPS from {path}...")
    df = pd.read_parquet(path)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    # Add survey identifier
    df["_survey"] = "cps"

    # Weight column
    if "march_supplement_weight" in df.columns:
        df["weight"] = df["march_supplement_weight"]
    elif "weight" not in df.columns:
        df["weight"] = 1.0

    print(f"  Loaded {len(df):,} CPS records")
    return df


def load_puf(
    year: int = 2024,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Load PUF data from HuggingFace or local cache."""
    if not HF_AVAILABLE:
        print("  Warning: huggingface_hub not installed, skipping PUF")
        return pd.DataFrame()

    filename = f"puf_{year}.h5"
    local_path = STORAGE_FOLDER / filename

    if not local_path.exists():
        print(f"Downloading PUF {year} from HuggingFace...")
        STORAGE_FOLDER.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="policyengine/irs-soi-puf",
                filename=filename,
                repo_type="model",
                local_dir=STORAGE_FOLDER,
            )
        except Exception as e:
            print(f"  Warning: Could not download PUF: {e}")
            return pd.DataFrame()

    print(f"Loading PUF from {local_path}...")
    try:
        df = pd.read_hdf(local_path)
    except Exception as e:
        print(f"  Warning: Could not load PUF: {e}")
        return pd.DataFrame()

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    # Add survey identifier
    df["_survey"] = "puf"

    # Weight column
    if "household_weight" in df.columns:
        df["weight"] = df["household_weight"]
    elif "weight" not in df.columns:
        df["weight"] = 1.0

    print(f"  Loaded {len(df):,} PUF records")
    return df


def load_sipp(
    year: int = 2023,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Load SIPP data from HuggingFace or local cache."""
    if not HF_AVAILABLE:
        print("  Warning: huggingface_hub not installed, skipping SIPP")
        return pd.DataFrame()

    filename = f"pu{year}_slim.csv"
    local_path = STORAGE_FOLDER / filename

    if not local_path.exists():
        print(f"Downloading SIPP {year} from HuggingFace...")
        STORAGE_FOLDER.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="PolicyEngine/policyengine-us-data",
                filename=filename,
                repo_type="model",
                local_dir=STORAGE_FOLDER,
            )
        except Exception as e:
            print(f"  Warning: Could not download SIPP: {e}")
            return pd.DataFrame()

    print(f"Loading SIPP from {local_path}...")
    try:
        df = pd.read_csv(local_path)
    except Exception as e:
        print(f"  Warning: Could not load SIPP: {e}")
        return pd.DataFrame()

    # Derive tip income if columns exist
    tip_cols = [c for c in df.columns if "TXAMT" in c]
    if tip_cols:
        df["tip_income"] = df[tip_cols].fillna(0).sum(axis=1) * 12

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)

    # Add survey identifier
    df["_survey"] = "sipp"

    # Weight column
    if "WPFINWGT" in df.columns:
        df["weight"] = df["WPFINWGT"]
    elif "weight" not in df.columns:
        df["weight"] = 1.0

    print(f"  Loaded {len(df):,} SIPP records")
    return df


def harmonize_variable(
    df: pd.DataFrame,
    common_name: str,
    survey: str,
) -> pd.Series:
    """Extract and harmonize a variable from survey data."""
    mapping = VARIABLE_MAPPING.get(common_name, {})
    survey_name = mapping.get(survey)

    if survey_name is None or survey_name not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)

    values = df[survey_name].copy()

    # Apply survey-specific transformations
    if common_name == "is_male":
        if survey == "cps":
            # CPS: 1=male, 2=female
            values = (values == 1).astype(float)
        elif survey == "sipp":
            # SIPP: 1=male, 2=female
            values = (values == 1).astype(float)

    return values


def stack_surveys(
    surveys: Dict[str, pd.DataFrame],
    variables: List[str],
) -> pd.DataFrame:
    """Stack multiple surveys into single DataFrame with NaN for missing vars.

    This is the key function for multi-survey fusion. Variables not present
    in a survey are set to NaN, and microplex's masked loss will only
    compute loss on observed (non-NaN) values.
    """
    print("\nStacking surveys for multi-survey fusion...")

    stacked_rows = []
    for survey_name, df in surveys.items():
        if len(df) == 0:
            continue

        print(f"  Processing {survey_name}: {len(df):,} records")

        # Create harmonized DataFrame
        harmonized = pd.DataFrame(index=df.index)

        for var in variables:
            harmonized[var] = harmonize_variable(df, var, survey_name)

        # Add weight and survey identifier
        harmonized["weight"] = df["weight"]
        harmonized["_survey"] = survey_name

        stacked_rows.append(harmonized)

    result = pd.concat(stacked_rows, ignore_index=True)
    print(f"\nStacked total: {len(result):,} records")

    # Report missing data pattern
    print("\nMissing data pattern:")
    for var in variables:
        n_observed = result[var].notna().sum()
        pct = 100 * n_observed / len(result)
        print(f"  {var}: {n_observed:,} observed ({pct:.1f}%)")

    return result


def load_all_surveys(
    cps_path: Optional[Path] = None,
    sample_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load and stack all available surveys.

    Returns:
        stacked: Combined DataFrame with harmonized variables
        surveys: Dict of individual survey DataFrames
    """
    surveys = {}

    # Load CPS
    cps = load_cps(cps_path, sample_frac=sample_frac, seed=seed)
    if len(cps) > 0:
        surveys["cps"] = cps

    # Load PUF
    puf = load_puf(sample_frac=sample_frac, seed=seed)
    if len(puf) > 0:
        surveys["puf"] = puf

    # Load SIPP
    sipp = load_sipp(sample_frac=sample_frac, seed=seed)
    if len(sipp) > 0:
        surveys["sipp"] = sipp

    # Define common variables to harmonize
    common_vars = [
        "age",
        "is_male",
        "state_fips",
        "marital_status",
        "wage_income",
        "self_employment_income",
        "interest_income",
        "dividend_income",
        "social_security_income",
        "unemployment_compensation",
        "rental_income",
        "capital_gains",
        "tip_income",
    ]

    # Stack surveys
    stacked = stack_surveys(surveys, common_vars)

    return stacked, surveys


if __name__ == "__main__":
    # Test loading
    stacked, surveys = load_all_surveys(sample_frac=0.01)
    print(f"\nLoaded {len(surveys)} surveys")
    print(f"Total stacked records: {len(stacked):,}")
