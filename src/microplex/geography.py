"""
Block-based geography derivation utilities for microplex.

This module provides tools for working with Census block GEOIDs and deriving
higher-level geographies (tract, county, state, congressional district).

Census GEOID Structure (15 characters total):
- State FIPS:     2 chars (positions 0-1)
- County FIPS:    3 chars (positions 2-4)
- Tract:          6 chars (positions 5-10)
- Block:          4 chars (positions 11-14)

Example:
    >>> from microplex.geography import BlockGeography
    >>> geo = BlockGeography()
    >>> geo.get_state("010010201001000")
    '01'
    >>> geo.get_county("010010201001000")
    '01001'
    >>> geo.get_tract("010010201001000")
    '01001020100'
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# GEOID structure constants
STATE_LEN = 2
COUNTY_LEN = 3  # County portion after state (total 5 chars for state+county)
TRACT_LEN = 6   # Tract portion after county (total 11 chars for tract GEOID)
BLOCK_LEN = 4   # Block portion after tract (total 15 chars for full GEOID)

# Full length constants for convenience
STATE_GEOID_LEN = STATE_LEN  # 2
COUNTY_GEOID_LEN = STATE_LEN + COUNTY_LEN  # 5
TRACT_GEOID_LEN = STATE_LEN + COUNTY_LEN + TRACT_LEN  # 11
BLOCK_GEOID_LEN = STATE_LEN + COUNTY_LEN + TRACT_LEN + BLOCK_LEN  # 15

# Default data directory (relative to package root)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_BLOCK_PROBABILITIES_PATH = DEFAULT_DATA_DIR / "block_probabilities.parquet"


def load_block_probabilities(
    path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load block probabilities from parquet file.

    Args:
        path: Path to parquet file. If None, uses default package data location.

    Returns:
        DataFrame with columns: geoid, state_fips, county, tract, block,
        population, tract_geoid, cd_id, state_total, prob, national_prob

    Raises:
        FileNotFoundError: If the parquet file doesn't exist.

    Example:
        >>> df = load_block_probabilities()
        >>> print(f"Loaded {len(df):,} blocks")
    """
    if path is None:
        path = DEFAULT_BLOCK_PROBABILITIES_PATH
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Block probabilities file not found at {path}.\n"
            "Run the data preparation script to generate this file."
        )

    return pd.read_parquet(path)


def derive_geographies(
    block_geoids: Union[List[str], np.ndarray, pd.Series],
    include_cd: bool = False,
    include_sld: bool = False,
    block_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Derive all higher-level geographies from block GEOIDs.

    This is a convenience function for batch processing. For repeated lookups,
    use BlockGeography class with caching.

    Args:
        block_geoids: List/array of 15-character block GEOIDs
        include_cd: If True, include congressional district lookup (requires block_data)
        include_sld: If True, include state legislative district lookup (requires block_data)
        block_data: Block probabilities DataFrame for CD/SLD lookup

    Returns:
        DataFrame with columns: block_geoid, state_fips, county_fips, tract_geoid
        If include_cd=True, also includes cd_id column.
        If include_sld=True, also includes sldu_id and sldl_id columns.

    Example:
        >>> geoids = ["010010201001000", "060372073021001"]
        >>> result = derive_geographies(geoids)
        >>> print(result)
    """
    geoids = pd.Series(block_geoids).astype(str)

    result = pd.DataFrame({
        "block_geoid": geoids,
        "state_fips": geoids.str[:STATE_GEOID_LEN],
        "county_fips": geoids.str[:COUNTY_GEOID_LEN],
        "tract_geoid": geoids.str[:TRACT_GEOID_LEN],
    })

    if include_cd or include_sld:
        if block_data is None:
            block_data = load_block_probabilities()

    if include_cd:
        # Create lookup dict for CD
        cd_lookup = dict(zip(block_data["geoid"], block_data["cd_id"]))
        result["cd_id"] = geoids.map(cd_lookup)

    if include_sld:
        # Create lookup dicts for SLD
        if "sldu_id" in block_data.columns:
            sldu_lookup = dict(zip(block_data["geoid"], block_data["sldu_id"]))
            result["sldu_id"] = geoids.map(sldu_lookup)
        if "sldl_id" in block_data.columns:
            sldl_lookup = dict(zip(block_data["geoid"], block_data["sldl_id"]))
            result["sldl_id"] = geoids.map(sldl_lookup)

    return result


class BlockGeography:
    """
    Geography derivation from Census block GEOIDs.

    Provides efficient methods for deriving higher-level geographies
    (tract, county, state, congressional district) from block GEOIDs,
    with caching for performance.

    Attributes:
        data: Block probabilities DataFrame (loaded lazily)

    Example:
        >>> geo = BlockGeography()
        >>> block = "060372073021001"  # A block in Los Angeles County, CA
        >>> geo.get_state(block)
        '06'
        >>> geo.get_county(block)
        '06037'
        >>> geo.get_tract(block)
        '06037207302'
        >>> geo.get_cd(block)
        'CA-37'
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        lazy_load: bool = True,
    ):
        """
        Initialize BlockGeography.

        Args:
            data_path: Path to block probabilities parquet. If None, uses default.
            lazy_load: If True, defer loading data until needed. If False, load immediately.
        """
        self._data_path = data_path
        self._data: Optional[pd.DataFrame] = None
        self._cd_lookup: Optional[Dict[str, str]] = None
        self._sldu_lookup: Optional[Dict[str, str]] = None
        self._sldl_lookup: Optional[Dict[str, str]] = None
        self._state_blocks: Optional[Dict[str, pd.DataFrame]] = None

        if not lazy_load:
            self._load_data()

    def _load_data(self) -> None:
        """Load block probabilities data if not already loaded."""
        if self._data is None:
            self._data = load_block_probabilities(self._data_path)

    @property
    def data(self) -> pd.DataFrame:
        """Block probabilities DataFrame (loaded lazily)."""
        if self._data is None:
            self._load_data()
        return self._data

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_state(block_geoid: str) -> str:
        """
        Get state FIPS from block GEOID.

        Args:
            block_geoid: 15-character Census block GEOID

        Returns:
            2-character state FIPS code

        Example:
            >>> BlockGeography.get_state("060372073021001")
            '06'
        """
        return block_geoid[:STATE_GEOID_LEN]

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_county(block_geoid: str) -> str:
        """
        Get county FIPS (state + county) from block GEOID.

        Args:
            block_geoid: 15-character Census block GEOID

        Returns:
            5-character county FIPS (state + county)

        Example:
            >>> BlockGeography.get_county("060372073021001")
            '06037'
        """
        return block_geoid[:COUNTY_GEOID_LEN]

    @staticmethod
    @lru_cache(maxsize=100000)
    def get_tract(block_geoid: str) -> str:
        """
        Get tract GEOID from block GEOID.

        Args:
            block_geoid: 15-character Census block GEOID

        Returns:
            11-character tract GEOID

        Example:
            >>> BlockGeography.get_tract("060372073021001")
            '06037207302'
        """
        return block_geoid[:TRACT_GEOID_LEN]

    def get_cd(self, block_geoid: str) -> Optional[str]:
        """
        Get congressional district ID from block GEOID.

        Unlike state/county/tract (which can be derived from the GEOID string),
        congressional district requires a lookup in the block data.

        Args:
            block_geoid: 15-character Census block GEOID

        Returns:
            Congressional district ID (e.g., "CA-37") or None if not found

        Example:
            >>> geo = BlockGeography()
            >>> geo.get_cd("060372073021001")
            'CA-37'
        """
        if self._cd_lookup is None:
            self._build_lookups()

        return self._cd_lookup.get(block_geoid)

    def get_sldu(self, block_geoid: str) -> Optional[str]:
        """
        Get State Senate (upper chamber) district ID from block GEOID.

        Args:
            block_geoid: 15-character Census block GEOID

        Returns:
            SLDU ID (e.g., "CA-SLDU-01") or None if not found

        Example:
            >>> geo = BlockGeography()
            >>> geo.get_sldu("060372073021001")
            'CA-SLDU-22'
        """
        if self._sldu_lookup is None:
            self._build_lookups()

        return self._sldu_lookup.get(block_geoid)

    def get_sldl(self, block_geoid: str) -> Optional[str]:
        """
        Get State House (lower chamber) district ID from block GEOID.

        Note: Nebraska has a unicameral legislature, so SLDL will be None
        for Nebraska blocks.

        Args:
            block_geoid: 15-character Census block GEOID

        Returns:
            SLDL ID (e.g., "CA-SLDL-40") or None if not found

        Example:
            >>> geo = BlockGeography()
            >>> geo.get_sldl("060372073021001")
            'CA-SLDL-46'
        """
        if self._sldl_lookup is None:
            self._build_lookups()

        return self._sldl_lookup.get(block_geoid)

    def _build_lookups(self) -> None:
        """Build lookup dictionaries for CD and SLD."""
        self._cd_lookup = dict(zip(self.data["geoid"], self.data["cd_id"]))

        # SLD lookups (may not exist in older data)
        if "sldu_id" in self.data.columns:
            self._sldu_lookup = dict(zip(self.data["geoid"], self.data["sldu_id"]))
        else:
            self._sldu_lookup = {}

        if "sldl_id" in self.data.columns:
            self._sldl_lookup = dict(zip(self.data["geoid"], self.data["sldl_id"]))
        else:
            self._sldl_lookup = {}

    def get_all_geographies(self, block_geoid: str) -> Dict[str, Optional[str]]:
        """
        Get all derived geographies for a block GEOID.

        Args:
            block_geoid: 15-character Census block GEOID

        Returns:
            Dictionary with keys: state_fips, county_fips, tract_geoid, cd_id,
            sldu_id, sldl_id

        Example:
            >>> geo = BlockGeography()
            >>> geo.get_all_geographies("060372073021001")
            {'state_fips': '06', 'county_fips': '06037',
             'tract_geoid': '06037207302', 'cd_id': 'CA-37',
             'sldu_id': 'CA-SLDU-22', 'sldl_id': 'CA-SLDL-46'}
        """
        return {
            "state_fips": self.get_state(block_geoid),
            "county_fips": self.get_county(block_geoid),
            "tract_geoid": self.get_tract(block_geoid),
            "cd_id": self.get_cd(block_geoid),
            "sldu_id": self.get_sldu(block_geoid),
            "sldl_id": self.get_sldl(block_geoid),
        }

    def sample_blocks(
        self,
        state_fips: str,
        n: int,
        replace: bool = True,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample blocks from a state using population-weighted probabilities.

        Args:
            state_fips: 2-character state FIPS code
            n: Number of blocks to sample
            replace: Sample with replacement (default True)
            random_state: Random seed for reproducibility

        Returns:
            Array of sampled block GEOIDs

        Raises:
            ValueError: If state_fips not found in data

        Example:
            >>> geo = BlockGeography()
            >>> blocks = geo.sample_blocks("06", n=100, random_state=42)
            >>> print(f"Sampled {len(blocks)} blocks from California")
        """
        # Build state index if needed
        if self._state_blocks is None:
            self._build_state_index()

        if state_fips not in self._state_blocks:
            raise ValueError(
                f"State FIPS '{state_fips}' not found in block data. "
                f"Available states: {sorted(self._state_blocks.keys())}"
            )

        state_df = self._state_blocks[state_fips]

        if random_state is not None:
            np.random.seed(random_state)

        # Use within-state probabilities (prob column)
        sampled_indices = np.random.choice(
            len(state_df),
            size=n,
            replace=replace,
            p=state_df["prob"].values,
        )

        return state_df["geoid"].values[sampled_indices]

    def _build_state_index(self) -> None:
        """Build index of blocks by state for efficient sampling."""
        self._state_blocks = {}
        for state_fips, group in self.data.groupby("state_fips"):
            # Store as a copy to ensure prob column is contiguous
            self._state_blocks[state_fips] = group[["geoid", "prob"]].copy()

    def sample_blocks_national(
        self,
        n: int,
        replace: bool = True,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample blocks nationally using population-weighted probabilities.

        Args:
            n: Number of blocks to sample
            replace: Sample with replacement (default True)
            random_state: Random seed for reproducibility

        Returns:
            Array of sampled block GEOIDs

        Example:
            >>> geo = BlockGeography()
            >>> blocks = geo.sample_blocks_national(n=1000, random_state=42)
            >>> print(f"Sampled {len(blocks)} blocks from US")
        """
        if random_state is not None:
            np.random.seed(random_state)

        sampled_indices = np.random.choice(
            len(self.data),
            size=n,
            replace=replace,
            p=self.data["national_prob"].values,
        )

        return self.data["geoid"].values[sampled_indices]

    def get_blocks_in_state(self, state_fips: str) -> pd.DataFrame:
        """
        Get all blocks in a state.

        Args:
            state_fips: 2-character state FIPS code

        Returns:
            DataFrame with block data for the state

        Example:
            >>> geo = BlockGeography()
            >>> ca_blocks = geo.get_blocks_in_state("06")
            >>> print(f"California has {len(ca_blocks):,} blocks")
        """
        return self.data[self.data["state_fips"] == state_fips].copy()

    def get_blocks_in_county(self, county_fips: str) -> pd.DataFrame:
        """
        Get all blocks in a county.

        Args:
            county_fips: 5-character county FIPS (state + county)

        Returns:
            DataFrame with block data for the county

        Example:
            >>> geo = BlockGeography()
            >>> la_blocks = geo.get_blocks_in_county("06037")
            >>> print(f"Los Angeles County has {len(la_blocks):,} blocks")
        """
        state = county_fips[:STATE_GEOID_LEN]
        county = county_fips[STATE_GEOID_LEN:]
        return self.data[
            (self.data["state_fips"] == state) &
            (self.data["county"] == county)
        ].copy()

    def get_blocks_in_tract(self, tract_geoid: str) -> pd.DataFrame:
        """
        Get all blocks in a tract.

        Args:
            tract_geoid: 11-character tract GEOID

        Returns:
            DataFrame with block data for the tract

        Example:
            >>> geo = BlockGeography()
            >>> tract_blocks = geo.get_blocks_in_tract("06037207302")
            >>> print(f"Tract has {len(tract_blocks)} blocks")
        """
        return self.data[self.data["tract_geoid"] == tract_geoid].copy()

    def get_blocks_in_cd(self, cd_id: str) -> pd.DataFrame:
        """
        Get all blocks in a congressional district.

        Args:
            cd_id: Congressional district ID (e.g., "CA-37")

        Returns:
            DataFrame with block data for the congressional district

        Example:
            >>> geo = BlockGeography()
            >>> cd_blocks = geo.get_blocks_in_cd("CA-37")
            >>> print(f"District has {len(cd_blocks):,} blocks")
        """
        return self.data[self.data["cd_id"] == cd_id].copy()

    def get_blocks_in_sldu(self, sldu_id: str) -> pd.DataFrame:
        """
        Get all blocks in a State Senate (upper chamber) district.

        Args:
            sldu_id: SLDU ID (e.g., "CA-SLDU-22")

        Returns:
            DataFrame with block data for the State Senate district

        Example:
            >>> geo = BlockGeography()
            >>> sldu_blocks = geo.get_blocks_in_sldu("CA-SLDU-22")
            >>> print(f"District has {len(sldu_blocks):,} blocks")
        """
        if "sldu_id" not in self.data.columns:
            return pd.DataFrame()
        return self.data[self.data["sldu_id"] == sldu_id].copy()

    def get_blocks_in_sldl(self, sldl_id: str) -> pd.DataFrame:
        """
        Get all blocks in a State House (lower chamber) district.

        Args:
            sldl_id: SLDL ID (e.g., "CA-SLDL-46")

        Returns:
            DataFrame with block data for the State House district

        Example:
            >>> geo = BlockGeography()
            >>> sldl_blocks = geo.get_blocks_in_sldl("CA-SLDL-46")
            >>> print(f"District has {len(sldl_blocks):,} blocks")
        """
        if "sldl_id" not in self.data.columns:
            return pd.DataFrame()
        return self.data[self.data["sldl_id"] == sldl_id].copy()

    @property
    def states(self) -> List[str]:
        """List of all state FIPS codes in the data."""
        return sorted(self.data["state_fips"].unique())

    @property
    def n_blocks(self) -> int:
        """Total number of blocks in the data."""
        return len(self.data)

    def __repr__(self) -> str:
        if self._data is None:
            return "BlockGeography(not loaded)"
        return f"BlockGeography({self.n_blocks:,} blocks, {len(self.states)} states)"
