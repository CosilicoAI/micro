"""
CPS ASEC (Annual Social and Economic Supplement) data loading.

The CPS ASEC is the primary source for income and poverty statistics in the US.
Released annually in March, it contains detailed income, employment, and
demographic information for ~100K households.

Data source: https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "microplex"

# CPS ASEC data URLs by year
CPS_URLS = {
    2023: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
}

# Key variable mappings (Census variable name -> our name)
PERSON_VARIABLES = {
    # Demographics
    "A_AGE": "age",
    "A_SEX": "sex",
    "PRDTRACE": "race",
    "PEHSPNON": "hispanic",
    "A_HGA": "education",
    # Employment
    "A_CLSWKR": "class_of_worker",
    "A_WKSTAT": "work_status",
    "A_HRS1": "hours_worked",
    # Income (annual)
    "WSAL_VAL": "wage_income",
    "SEMP_VAL": "self_employment_income",
    "INT_VAL": "interest_income",
    "DIV_VAL": "dividend_income",
    "RNT_VAL": "rental_income",
    "SS_VAL": "social_security",
    "SSI_VAL": "ssi",
    "UC_VAL": "unemployment_compensation",
    "PTOTVAL": "total_person_income",
    # Benefits
    "PAW_VAL": "public_assistance",
    "MCARE": "has_medicare",
    "MCAID": "has_medicaid",
    # Identifiers
    "PH_SEQ": "household_id",
    "PF_SEQ": "family_id",
    "A_LINENO": "person_number",
    "A_FAMREL": "family_relationship",
    "A_MARITL": "marital_status",
    # Weights
    "A_FNLWGT": "weight",
    "MARSUPWT": "march_supplement_weight",
}

HOUSEHOLD_VARIABLES = {
    "H_SEQ": "household_id",
    "GESTFIPS": "state_fips",
    "GTCBSA": "cbsa",
    "HRHTYPE": "household_type",
    "H_NUMPER": "household_size",
    "HHINC": "household_income_bracket",
    "HTOTVAL": "household_total_income",
    "HSUP_WGT": "household_weight",
}


@dataclass
class CPSDataset:
    """Container for CPS ASEC data."""

    persons: pl.DataFrame
    households: pl.DataFrame
    year: int
    source: str

    @property
    def n_persons(self) -> int:
        return len(self.persons)

    @property
    def n_households(self) -> int:
        return len(self.households)

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "year": self.year,
            "n_persons": self.n_persons,
            "n_households": self.n_households,
            "states": self.households["state_fips"].n_unique(),
            "total_weight": float(self.persons["weight"].sum()),
        }


def download_cps_asec(
    year: int,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download CPS ASEC data for a given year.

    Args:
        year: Year of CPS ASEC (e.g., 2023)
        cache_dir: Directory to cache downloads
        force: Re-download even if cached

    Returns:
        Path to downloaded/cached zip file
    """
    import httpx
    import zipfile

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    if year not in CPS_URLS:
        available = ", ".join(str(y) for y in sorted(CPS_URLS.keys()))
        raise ValueError(f"CPS ASEC for {year} not available. Available: {available}")

    url = CPS_URLS[year]
    filename = f"cps_asec_{year}.zip"
    cache_path = cache_dir / filename

    if cache_path.exists() and not force:
        print(f"Using cached CPS ASEC {year} from {cache_path}")
        return cache_path

    print(f"Downloading CPS ASEC {year} from {url}...")

    with httpx.Client(follow_redirects=True, timeout=300) as client:
        response = client.get(url)
        response.raise_for_status()

        with open(cache_path, "wb") as f:
            f.write(response.content)

    print(f"Downloaded {len(response.content) / 1_000_000:.1f} MB to {cache_path}")
    return cache_path


def load_cps_asec(
    year: int = 2023,
    cache_dir: Path | None = None,
    download: bool = True,
) -> CPSDataset:
    """
    Load CPS ASEC data for a given year.

    Args:
        year: Year of CPS ASEC (e.g., 2023)
        cache_dir: Directory for cached data
        download: Whether to download if not cached

    Returns:
        CPSDataset with persons and households DataFrames
    """
    import zipfile

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Check for processed parquet first
    processed_path = cache_dir / f"cps_asec_{year}_processed.parquet"
    if processed_path.exists():
        print(f"Loading processed CPS ASEC {year} from {processed_path}")
        persons = pl.read_parquet(processed_path)
        # Derive households from persons
        households = _derive_households(persons)
        return CPSDataset(
            persons=persons,
            households=households,
            year=year,
            source=str(processed_path),
        )

    # Download if needed
    zip_path = cache_dir / f"cps_asec_{year}.zip"
    if not zip_path.exists():
        if not download:
            raise FileNotFoundError(
                f"CPS ASEC {year} not found at {zip_path}. "
                "Set download=True to fetch from Census."
            )
        zip_path = download_cps_asec(year, cache_dir)

    # Extract and parse
    print(f"Parsing CPS ASEC {year}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the person file (pppub*.csv)
        person_file = None
        household_file = None

        for name in zf.namelist():
            lower = name.lower()
            if "pppub" in lower and lower.endswith(".csv"):
                person_file = name
            elif "hhpub" in lower and lower.endswith(".csv"):
                household_file = name

        if person_file is None:
            raise ValueError(f"Could not find person file in {zip_path}")

        # Schema overrides for columns with large IDs that overflow int64
        schema_overrides = {
            "PERIDNUM": pl.Utf8,  # Person ID - too large for int64
            "H_IDNUM": pl.Utf8,  # Household ID - too large for int64
            "OCCURNUM": pl.Utf8,  # Occurrence number
            "QSTNUM": pl.Utf8,  # Questionnaire number
        }

        # Read person data
        with zf.open(person_file) as f:
            persons_raw = pl.read_csv(
                f,
                infer_schema_length=10000,
                schema_overrides=schema_overrides,
            )

        # Read household data if available
        if household_file:
            with zf.open(household_file) as f:
                households_raw = pl.read_csv(
                    f,
                    infer_schema_length=10000,
                    schema_overrides=schema_overrides,
                )
        else:
            households_raw = None

    # Process person data
    persons = _process_persons(persons_raw, year)

    # Process or derive household data
    if households_raw is not None:
        households = _process_households(households_raw, year)
    else:
        households = _derive_households(persons)

    # Cache processed data
    persons.write_parquet(processed_path)
    print(f"Cached processed data to {processed_path}")

    return CPSDataset(
        persons=persons,
        households=households,
        year=year,
        source=str(zip_path),
    )


def _process_persons(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Process raw person file into clean format."""
    # Select and rename available columns
    available = set(df.columns)
    selected = {}

    for census_name, our_name in PERSON_VARIABLES.items():
        if census_name in available:
            selected[census_name] = our_name

    if not selected:
        raise ValueError("No recognized variables found in person file")

    result = df.select([
        pl.col(census_name).alias(our_name)
        for census_name, our_name in selected.items()
    ])

    # Scale weights: CPS ASEC weights have 2 implied decimal places
    # See CPS documentation: A_FNLWGT is expressed in units of 1/100
    # Divide by 100 to get actual population representation
    if "weight" in result.columns:
        result = result.with_columns(
            (pl.col("weight") / 100).alias("weight")
        )
    if "march_supplement_weight" in result.columns:
        result = result.with_columns(
            (pl.col("march_supplement_weight") / 100).alias("march_supplement_weight")
        )

    # Convert income values (negative values indicate no income or missing)
    income_cols = [
        "wage_income", "self_employment_income", "interest_income",
        "dividend_income", "rental_income", "social_security", "ssi",
        "unemployment_compensation", "public_assistance", "total_person_income"
    ]

    for col in income_cols:
        if col in result.columns:
            result = result.with_columns(
                pl.when(pl.col(col) < 0)
                .then(0)
                .otherwise(pl.col(col))
                .alias(col)
            )

    # Add derived columns
    if "age" in result.columns:
        result = result.with_columns([
            (pl.col("age") >= 18).alias("is_adult"),
            (pl.col("age") < 18).alias("is_child"),
            (pl.col("age") >= 65).alias("is_senior"),
        ])

    # Add year
    result = result.with_columns(pl.lit(year).alias("year"))

    return result


def _process_households(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Process raw household file into clean format."""
    available = set(df.columns)
    selected = {}

    for census_name, our_name in HOUSEHOLD_VARIABLES.items():
        if census_name in available:
            selected[census_name] = our_name

    if not selected:
        raise ValueError("No recognized variables found in household file")

    result = df.select([
        pl.col(census_name).alias(our_name)
        for census_name, our_name in selected.items()
    ])

    # Scale weights: CPS ASEC weights have 2 implied decimal places
    if "household_weight" in result.columns:
        result = result.with_columns(
            (pl.col("household_weight") / 100).alias("household_weight")
        )

    result = result.with_columns(pl.lit(year).alias("year"))

    return result


def _derive_households(persons: pl.DataFrame) -> pl.DataFrame:
    """Derive household-level data from person records."""
    if "household_id" not in persons.columns:
        raise ValueError("Cannot derive households without household_id")

    households = persons.group_by("household_id").agg([
        pl.len().alias("household_size"),
        pl.col("weight").first().alias("household_weight"),
        pl.col("state_fips").first() if "state_fips" in persons.columns else pl.lit(None).alias("state_fips"),
        pl.col("total_person_income").sum().alias("household_total_income") if "total_person_income" in persons.columns else pl.lit(0).alias("household_total_income"),
        pl.col("is_child").sum().alias("num_children") if "is_child" in persons.columns else pl.lit(0).alias("num_children"),
        pl.col("is_adult").sum().alias("num_adults") if "is_adult" in persons.columns else pl.lit(0).alias("num_adults"),
    ])

    if "year" in persons.columns:
        year_val = persons.select("year").unique().to_series()[0]
        households = households.with_columns(
            pl.lit(year_val).alias("year")
        )

    return households


def get_available_years() -> list[int]:
    """Return list of available CPS ASEC years."""
    return sorted(CPS_URLS.keys())
