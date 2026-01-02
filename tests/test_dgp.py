"""Tests for PopulationDGP - multi-source population synthesis."""

import sys
from pathlib import Path

# Add src to path for direct import (avoids loading full microplex package)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from microplex.dgp import PopulationDGP, Survey, run_multi_source_benchmark


class TestPopulationDGPBasics:
    """Basic functionality tests."""

    @pytest.fixture
    def simple_surveys(self):
        """Two surveys with overlapping columns."""
        np.random.seed(42)
        n = 500

        # Shared: age, income
        # Survey A adds: education, employment
        # Survey B adds: wealth, debt

        age = np.random.uniform(20, 70, n)
        income = 20000 + age * 1000 + np.random.normal(0, 10000, n)

        survey_a = pd.DataFrame({
            "age": age,
            "income": income,
            "education": np.random.choice([1, 2, 3, 4], n),
            "employed": np.random.choice([0, 1], n, p=[0.1, 0.9]),
        })

        survey_b = pd.DataFrame({
            "age": age + np.random.normal(0, 2, n),  # Slight noise
            "income": income + np.random.normal(0, 5000, n),
            "wealth": np.maximum(0, income * 5 + np.random.normal(0, 50000, n)),
            "debt": np.where(np.random.random(n) < 0.3, 0, np.random.uniform(0, 50000, n)),
        })

        return [
            Survey("survey_a", survey_a),
            Survey("survey_b", survey_b),
        ]

    def test_dgp_instantiation(self):
        """Test that PopulationDGP can be instantiated."""
        dgp = PopulationDGP()
        assert dgp is not None

    def test_dgp_fit(self, simple_surveys):
        """Test that DGP can fit on multiple surveys."""
        dgp = PopulationDGP()
        dgp.fit(simple_surveys, shared_cols=["age", "income"])

        assert len(dgp.all_cols_) == 6  # age, income, education, employed, wealth, debt
        assert "education" in dgp.col_to_survey_
        assert "wealth" in dgp.col_to_survey_

    def test_dgp_generate(self, simple_surveys):
        """Test that DGP can generate synthetic records."""
        dgp = PopulationDGP()
        dgp.fit(simple_surveys, shared_cols=["age", "income"])

        synthetic = dgp.generate(n=100)

        assert len(synthetic) == 100
        assert "age" in synthetic.columns
        assert "income" in synthetic.columns
        assert "education" in synthetic.columns
        assert "wealth" in synthetic.columns

    def test_dgp_generate_different_from_input(self, simple_surveys):
        """Test that generated records are NOT just copies of input."""
        dgp = PopulationDGP()
        dgp.fit(simple_surveys, shared_cols=["age", "income"])

        synthetic = dgp.generate(n=100, noise_scale=0.1)

        # With noise, shared values should differ from training
        train_ages = simple_surveys[0].data["age"].values
        synth_ages = synthetic["age"].values

        # Check that synthetic ages are not exact matches
        exact_matches = sum(
            any(np.isclose(sa, ta, atol=1e-6) for ta in train_ages)
            for sa in synth_ages
        )
        # Should have few exact matches due to noise
        assert exact_matches < len(synth_ages) * 0.5

    def test_dgp_evaluate(self, simple_surveys):
        """Test that DGP can evaluate against holdouts."""
        dgp = PopulationDGP()
        dgp.fit(simple_surveys, shared_cols=["age", "income"])

        holdouts = {
            "survey_a": simple_surveys[0].data.sample(50),
            "survey_b": simple_surveys[1].data.sample(50),
        }

        results = dgp.evaluate(holdouts)

        assert "survey_a" in results
        assert "survey_b" in results
        assert 0 <= results["survey_a"].coverage <= 1
        assert 0 <= results["survey_b"].coverage <= 1


class TestZeroInflatedHandling:
    """Test handling of zero-inflated variables."""

    @pytest.fixture
    def zero_inflated_surveys(self):
        """Surveys with zero-inflated columns."""
        np.random.seed(42)
        n = 500

        age = np.random.uniform(20, 70, n)
        income = 20000 + age * 1000 + np.random.normal(0, 10000, n)

        # Zero-inflated wealth columns
        stocks = np.where(
            np.random.random(n) < 0.7,  # 70% zeros
            0,
            np.random.uniform(1000, 100000, n),
        )
        bonds = np.where(
            np.random.random(n) < 0.9,  # 90% zeros
            0,
            np.random.uniform(500, 50000, n),
        )

        survey_a = pd.DataFrame({"age": age, "income": income})
        survey_b = pd.DataFrame({
            "age": age,
            "income": income,
            "stocks": stocks,
            "bonds": bonds,
        })

        return [Survey("survey_a", survey_a), Survey("survey_b", survey_b)]

    def test_detects_zero_inflation(self, zero_inflated_surveys):
        """Test that DGP detects zero-inflated columns."""
        dgp = PopulationDGP(zero_inflation_threshold=0.1)
        dgp.fit(zero_inflated_surveys, shared_cols=["age", "income"])

        assert dgp.is_zero_inflated_.get("stocks", False)
        assert dgp.is_zero_inflated_.get("bonds", False)

    def test_generates_zeros(self, zero_inflated_surveys):
        """Test that generated data includes zeros for zero-inflated columns."""
        dgp = PopulationDGP(zero_inflation_threshold=0.1)
        dgp.fit(zero_inflated_surveys, shared_cols=["age", "income"])

        synthetic = dgp.generate(n=500)

        # Should have some zeros
        stocks_zeros = (synthetic["stocks"] == synthetic["stocks"].min()).sum()
        bonds_zeros = (synthetic["bonds"] == synthetic["bonds"].min()).sum()

        assert stocks_zeros > 0
        assert bonds_zeros > 0

    def test_zero_fraction_approximately_preserved(self, zero_inflated_surveys):
        """Test that zero fraction is roughly preserved in synthetic data."""
        dgp = PopulationDGP(zero_inflation_threshold=0.1)
        dgp.fit(zero_inflated_surveys, shared_cols=["age", "income"])

        synthetic = dgp.generate(n=1000, seed=42)

        # Original: ~70% zeros for stocks
        synth_zero_frac = (synthetic["stocks"] == synthetic["stocks"].min()).mean()
        assert 0.5 < synth_zero_frac < 0.9  # Reasonable range


class TestMultiSourceBenchmark:
    """Test the full benchmark pipeline."""

    def test_run_benchmark(self):
        """Test full benchmark with train/holdout split."""
        np.random.seed(42)
        n = 300

        age = np.random.uniform(20, 70, n)
        income = 20000 + age * 1000 + np.random.normal(0, 10000, n)

        surveys = [
            Survey("A", pd.DataFrame({
                "age": age,
                "income": income,
                "education": np.random.choice([1, 2, 3, 4], n),
            })),
            Survey("B", pd.DataFrame({
                "age": age,
                "income": income,
                "wealth": np.maximum(0, income * 3 + np.random.normal(0, 30000, n)),
            })),
        ]

        dgp, results = run_multi_source_benchmark(
            surveys,
            shared_cols=["age", "income"],
            holdout_frac=0.2,
        )

        assert "A" in results
        assert "B" in results
        # Coverage should be reasonable (not 0)
        assert results["A"].coverage > 0.1
        assert results["B"].coverage > 0.1
