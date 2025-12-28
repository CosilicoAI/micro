"""
Tests for District Microplex Builder.

District microplex generates synthetic tax units at geographic granularity
(states, counties, congressional districts) and calibrates to local targets.
"""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, Tuple

from microplex.district import (
    DistrictMicroplex,
    build_targets_from_database,
    synthesize_district_records,
    STATE_FIPS,
)


class TestStateFIPS:
    """Test state FIPS code mapping."""

    def test_state_fips_has_all_states(self):
        """Should have all 50 states plus DC."""
        assert len(STATE_FIPS) == 51

    def test_state_fips_includes_major_states(self):
        """Should include major states with correct codes."""
        assert STATE_FIPS[6] == "CA"
        assert STATE_FIPS[36] == "NY"
        assert STATE_FIPS[48] == "TX"
        assert STATE_FIPS[12] == "FL"

    def test_state_fips_includes_dc(self):
        """Should include District of Columbia."""
        assert STATE_FIPS[11] == "DC"


class TestBuildTargetsFromDatabase:
    """Test building calibration targets from microplex targets database."""

    def test_returns_marginal_and_continuous_dicts(self):
        """Should return two dictionaries."""
        marginal, continuous = build_targets_from_database(year=2021)
        assert isinstance(marginal, dict)
        assert isinstance(continuous, dict)

    def test_marginal_targets_has_state_fips(self):
        """Should have state_fips in marginal targets."""
        marginal, _ = build_targets_from_database(year=2021)
        assert "state_fips" in marginal

    def test_marginal_targets_have_multiple_states(self):
        """Should have targets for multiple states."""
        marginal, _ = build_targets_from_database(year=2021)
        state_targets = marginal.get("state_fips", {})
        # At minimum should have some states
        assert len(state_targets) >= 10

    def test_state_targets_are_positive_counts(self):
        """State population targets should be positive."""
        marginal, _ = build_targets_from_database(year=2021)
        state_targets = marginal.get("state_fips", {})
        for state_fips, count in state_targets.items():
            assert count > 0, f"State {state_fips} has non-positive count"

    def test_california_has_large_population(self):
        """California should have a substantial population target."""
        marginal, _ = build_targets_from_database(year=2021)
        state_targets = marginal.get("state_fips", {})
        if 6 in state_targets:  # CA FIPS
            ca_count = state_targets[6]
            # CA should be larger than median (it's the most populous state)
            median_count = np.median(list(state_targets.values()))
            assert ca_count >= median_count, "CA should be above median population"


class TestSynthesizeDistrictRecords:
    """Test synthetic record generation for districts."""

    @pytest.fixture
    def seed_data(self) -> pd.DataFrame:
        """Create minimal seed data for testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "state_fips": np.random.choice([6, 36, 48], n),
            "wage_income": np.random.lognormal(10, 1, n),
            "self_employment_income": np.random.exponential(5000, n),
            "interest_income": np.random.exponential(500, n),
            "dividend_income": np.random.exponential(300, n),
            "rental_income": np.random.exponential(200, n),
            "social_security_income": np.random.exponential(2000, n),
            "head_age": np.random.randint(18, 85, n),
            "filing_status": np.random.choice(["single", "married", "hoh"], n),
            "num_dependents": np.random.randint(0, 4, n),
            "num_ctc_children": np.random.randint(0, 3, n),
            "num_eitc_children": np.random.randint(0, 3, n),
            "is_joint": np.random.choice([True, False], n),
            "weight": np.random.uniform(100, 1000, n),
        })

    def test_returns_dataframe(self, seed_data):
        """Should return a DataFrame."""
        result = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=50,
        )
        assert isinstance(result, pd.DataFrame)

    def test_returns_requested_number_of_records(self, seed_data):
        """Should return exactly n_records records."""
        n = 75
        result = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=n,
        )
        assert len(result) == n

    def test_includes_district_id_column(self, seed_data):
        """Should include district_id in output."""
        result = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=50,
        )
        assert "district_id" in result.columns
        assert (result["district_id"] == "06").all()

    def test_includes_state_fips(self, seed_data):
        """Should include state_fips based on district."""
        result = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=50,
        )
        assert "state_fips" in result.columns
        assert (result["state_fips"] == 6).all()

    def test_includes_income_columns(self, seed_data):
        """Should include income columns."""
        result = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=50,
        )
        income_cols = ["wage_income", "self_employment_income"]
        for col in income_cols:
            assert col in result.columns

    def test_includes_weight_column(self, seed_data):
        """Should initialize weight column."""
        result = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=50,
        )
        assert "weight" in result.columns

    def test_income_values_are_non_negative(self, seed_data):
        """Income values should be non-negative."""
        result = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=100,
        )
        income_cols = ["wage_income", "self_employment_income", "interest_income"]
        for col in income_cols:
            if col in result.columns:
                assert (result[col] >= 0).all(), f"{col} has negative values"

    def test_reproducible_with_seed(self, seed_data):
        """Should be reproducible with same seed."""
        result1 = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=50,
            seed=123,
        )
        result2 = synthesize_district_records(
            seed_data=seed_data,
            district_id="06",
            n_records=50,
            seed=123,
        )
        pd.testing.assert_frame_equal(result1, result2)


class TestDistrictMicroplex:
    """Test the main DistrictMicroplex class."""

    @pytest.fixture
    def seed_data(self) -> pd.DataFrame:
        """Create seed data for testing."""
        np.random.seed(42)
        n = 200
        states = [6, 36, 48, 12, 17]  # CA, NY, TX, FL, IL
        return pd.DataFrame({
            "state_fips": np.random.choice(states, n),
            "wage_income": np.random.lognormal(10, 1, n),
            "self_employment_income": np.random.exponential(5000, n),
            "interest_income": np.random.exponential(500, n),
            "dividend_income": np.random.exponential(300, n),
            "rental_income": np.random.exponential(200, n),
            "social_security_income": np.random.exponential(2000, n),
            "head_age": np.random.randint(18, 85, n),
            "filing_status": np.random.choice(["single", "married", "hoh"], n),
            "num_dependents": np.random.randint(0, 4, n),
            "num_ctc_children": np.random.randint(0, 3, n),
            "num_eitc_children": np.random.randint(0, 3, n),
            "is_joint": np.random.choice([True, False], n),
            "total_income": np.random.lognormal(10, 1, n),
            "weight": np.random.uniform(100, 1000, n),
        })

    def test_init_with_defaults(self):
        """Should initialize with defaults."""
        dm = DistrictMicroplex()
        assert dm.n_per_district == 1000
        assert dm.target_sparsity == 0.9

    def test_init_with_custom_params(self):
        """Should accept custom parameters."""
        dm = DistrictMicroplex(n_per_district=500, target_sparsity=0.8)
        assert dm.n_per_district == 500
        assert dm.target_sparsity == 0.8

    def test_fit_stores_maf(self, seed_data):
        """Fit should train and store normalizing flow."""
        dm = DistrictMicroplex(n_per_district=50)
        dm.fit(seed_data, epochs=5)
        assert dm._maf is not None

    def test_fit_stores_seed_data(self, seed_data):
        """Fit should store reference to seed data."""
        dm = DistrictMicroplex(n_per_district=50)
        dm.fit(seed_data, epochs=5)
        assert dm._seed_data is not None

    def test_generate_returns_dataframe(self, seed_data):
        """Generate should return DataFrame."""
        dm = DistrictMicroplex(n_per_district=50)
        dm.fit(seed_data, epochs=5)
        result = dm.generate(districts=["06", "36"])
        assert isinstance(result, pd.DataFrame)

    def test_generate_multiple_districts(self, seed_data):
        """Should generate for multiple districts."""
        dm = DistrictMicroplex(n_per_district=50)
        dm.fit(seed_data, epochs=5)
        districts = ["06", "36", "48"]
        result = dm.generate(districts=districts)
        # Should have records for each district
        assert len(result) == 50 * len(districts)

    def test_calibrate_adjusts_weights(self, seed_data):
        """Calibrate should adjust weights."""
        dm = DistrictMicroplex(n_per_district=50)
        dm.fit(seed_data, epochs=5)
        synthetic = dm.generate(districts=["06"])

        # Simple calibration targets
        marginal = {"state_fips": {6: 10000}}
        continuous = {"wage_income": 1_000_000_000}

        calibrated = dm.calibrate(synthetic, marginal, continuous)

        # Weights should be adjusted
        assert "weight" in calibrated.columns
        assert not (calibrated["weight"] == 1.0).all()

    def test_build_combines_all_steps(self, seed_data):
        """Build should combine fit, generate, and calibrate."""
        dm = DistrictMicroplex(n_per_district=50)

        # Simple targets
        marginal = {"state_fips": {6: 1000, 36: 800}}
        continuous = {}

        result = dm.build(
            seed_data=seed_data,
            districts=["06", "36"],
            marginal_targets=marginal,
            continuous_targets=continuous,
            epochs=5,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "weight" in result.columns


class TestDistrictMicroplexIntegration:
    """Integration tests with real targets database."""

    @pytest.fixture
    def seed_data(self) -> pd.DataFrame:
        """Create realistic seed data."""
        np.random.seed(42)
        n = 500
        # All 51 states
        states = list(STATE_FIPS.keys())
        state_probs = np.ones(len(states)) / len(states)

        return pd.DataFrame({
            "state_fips": np.random.choice(states, n, p=state_probs),
            "wage_income": np.random.lognormal(10, 1, n),
            "self_employment_income": np.random.exponential(5000, n),
            "interest_income": np.random.exponential(500, n),
            "dividend_income": np.random.exponential(300, n),
            "rental_income": np.random.exponential(200, n),
            "social_security_income": np.random.exponential(2000, n),
            "head_age": np.random.randint(18, 85, n),
            "filing_status": np.random.choice(["single", "married", "hoh"], n),
            "num_dependents": np.random.randint(0, 4, n),
            "num_ctc_children": np.random.randint(0, 3, n),
            "num_eitc_children": np.random.randint(0, 3, n),
            "is_joint": np.random.choice([True, False], n),
            "total_income": np.random.lognormal(10, 1, n),
            "weight": np.random.uniform(100, 1000, n),
        })

    @pytest.mark.integration
    def test_build_with_database_targets(self, seed_data):
        """Should build with targets from microplex database."""
        dm = DistrictMicroplex(n_per_district=20, target_sparsity=0.8)

        # Get targets from database
        marginal, continuous = build_targets_from_database(year=2021)

        # Use subset of districts for speed
        districts = ["06", "36", "48"]

        result = dm.build(
            seed_data=seed_data,
            districts=districts,
            marginal_targets=marginal,
            continuous_targets=continuous,
            epochs=5,
        )

        assert len(result) > 0
        assert "weight" in result.columns
        # Should have some sparsity
        non_zero = (result["weight"] > 1e-9).sum()
        assert non_zero < len(result)  # Some weights zeroed


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_seed_data_raises(self):
        """Should raise error for empty seed data."""
        dm = DistrictMicroplex()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            dm.fit(empty_df)

    def test_missing_required_columns_raises(self):
        """Should raise error if required columns missing."""
        dm = DistrictMicroplex()
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError):
            dm.fit(df)

    def test_invalid_district_id_format(self):
        """Should handle non-standard district IDs."""
        dm = DistrictMicroplex(n_per_district=10)
        seed = pd.DataFrame({
            "state_fips": [6, 6, 6],
            "wage_income": [50000, 60000, 70000],
            "head_age": [35, 40, 45],
            "weight": [100, 100, 100],
        })
        # Should not crash on unusual district ID
        result = synthesize_district_records(seed, "999", 5)
        assert len(result) == 5

    def test_single_record_seed_data(self):
        """Should handle seed data with single record."""
        dm = DistrictMicroplex(n_per_district=10)
        seed = pd.DataFrame({
            "state_fips": [6],
            "wage_income": [50000],
            "self_employment_income": [0],
            "interest_income": [100],
            "dividend_income": [0],
            "rental_income": [0],
            "social_security_income": [0],
            "head_age": [35],
            "weight": [100],
        })
        result = synthesize_district_records(seed, "06", 5)
        assert len(result) == 5
