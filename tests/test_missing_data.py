"""Tests for masked loss with missing data (multi-survey fusion).

Verifies that microplex can train on data with NaN values (missing observations)
using masked loss, which is essential for multi-survey fusion.
"""

import numpy as np
import pandas as pd
import pytest

from microplex import Synthesizer


class TestMaskedLossBasics:
    """Test basic masked loss functionality."""

    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with NaN values simulating multi-survey stacking."""
        np.random.seed(42)
        n = 1000

        # Context variables (always observed)
        age = np.random.randint(18, 80, n)
        education = np.random.randint(0, 5, n)

        # Target variables with missing values (simulating different surveys)
        # Survey 1: observes wage, uc (first 500 rows)
        # Survey 2: observes wage, self_emp, investment (last 500 rows)
        wage = np.where(age > 25, age * 1000 + np.random.normal(0, 5000, n), 0)
        self_emp = np.where(education > 2, np.random.exponential(10000, n), 0)
        uc = np.where(age < 65, np.random.exponential(500, n) * (age < 30), 0)
        investment = np.where(age > 40, np.random.exponential(5000, n), 0)

        data = pd.DataFrame({
            "age": age,
            "education": education,
            "wage": wage,
            "self_emp": self_emp,
            "uc": uc,
            "investment": investment,
        })

        # Simulate missing data pattern
        # First 500: self_emp and investment are missing
        # Last 500: uc is missing
        data.loc[:499, "self_emp"] = np.nan
        data.loc[:499, "investment"] = np.nan
        data.loc[500:, "uc"] = np.nan

        return data

    def test_fit_with_nan_values(self, sample_data_with_missing):
        """Synthesizer should fit without error when data contains NaN."""
        synth = Synthesizer(
            target_vars=["wage", "self_emp", "uc", "investment"],
            condition_vars=["age", "education"],
            n_layers=2,
            hidden_dim=32,
        )

        # Should not raise
        synth.fit(sample_data_with_missing, epochs=10, verbose=False)
        assert synth.is_fitted_

    def test_generate_produces_no_nan(self, sample_data_with_missing):
        """Generated data should have no NaN values."""
        synth = Synthesizer(
            target_vars=["wage", "self_emp", "uc", "investment"],
            condition_vars=["age", "education"],
            n_layers=2,
            hidden_dim=32,
        )
        synth.fit(sample_data_with_missing, epochs=20, verbose=False)

        conditions = sample_data_with_missing[["age", "education"]].iloc[:100]
        generated = synth.generate(conditions, seed=42)

        # No NaN in output
        assert not generated.isna().any().any()

    def test_observation_frequencies_computed(self, sample_data_with_missing):
        """Observation frequencies should be computed correctly from original data."""
        synth = Synthesizer(
            target_vars=["wage", "self_emp", "uc", "investment"],
            condition_vars=["age", "education"],
            n_layers=2,
            hidden_dim=32,
        )
        synth.fit(sample_data_with_missing, epochs=10, verbose=False)

        # Check observation weights were computed
        assert synth._obs_weights is not None

        # wage is always observed (freq=1.0, weight=1.0)
        # self_emp and investment are 50% observed (freq=0.5, weight=2.0)
        # uc is 50% observed (freq=0.5, weight=2.0)
        expected_wage_weight = 1.0
        expected_missing_weight = 2.0  # 1/0.5

        assert abs(synth._obs_weights[0].item() - expected_wage_weight) < 0.1
        assert abs(synth._obs_weights[1].item() - expected_missing_weight) < 0.1


class TestMaskedLossVsNaiveFill:
    """Compare masked loss to naive NaN→0 filling."""

    @pytest.fixture
    def data_with_structured_missing(self):
        """Create data where NaN→0 would corrupt the distribution."""
        np.random.seed(42)
        n = 2000

        age = np.random.randint(18, 80, n)

        # income1 is always positive when observed
        income1 = np.abs(np.random.normal(50000, 20000, n))

        # income2 has realistic zero-inflation pattern
        # About 70% positive, 30% zero when observed
        income2_base = np.abs(np.random.normal(30000, 10000, n))
        income2 = np.where(np.random.random(n) > 0.3, income2_base, 0)

        data = pd.DataFrame({
            "age": age,
            "income1": income1,
            "income2": income2,
        })

        # Make income2 missing for half the records
        data.loc[:999, "income2"] = np.nan

        return data

    def test_masked_loss_learns_from_observed(self, data_with_structured_missing):
        """Masked loss should learn income2 distribution from observed data only."""
        synth = Synthesizer(
            target_vars=["income1", "income2"],
            condition_vars=["age"],
            n_layers=4,
            hidden_dim=64,
            zero_inflated=True,
        )
        synth.fit(data_with_structured_missing, epochs=100, verbose=False)

        conditions = data_with_structured_missing[["age"]].iloc[:200]
        generated = synth.generate(conditions, seed=42)

        # Generated income2 should have some positive values
        # (The observed data has ~70% positive, but model may differ)
        # Key test: should not be all zeros (which would happen with naive NaN→0)
        has_variation = generated["income2"].std() > 0
        assert has_variation, "income2 should have variation, not all zeros"


class TestCompleteDataBackwardCompatibility:
    """Ensure masked loss doesn't break complete data case."""

    @pytest.fixture
    def complete_data(self):
        """Data with no missing values."""
        np.random.seed(42)
        n = 500

        age = np.random.randint(18, 80, n)
        income = np.where(age > 25, age * 1000 + np.random.normal(0, 5000, n), 0)

        return pd.DataFrame({"age": age, "income": income})

    def test_fit_without_nan(self, complete_data):
        """Fitting complete data should work as before."""
        synth = Synthesizer(
            target_vars=["income"],
            condition_vars=["age"],
            n_layers=2,
            hidden_dim=32,
        )
        synth.fit(complete_data, epochs=20, verbose=False)
        assert synth.is_fitted_
        assert synth._obs_weights is None  # No weighting for complete data

    def test_generate_matches_distribution(self, complete_data):
        """Generated data should match training distribution."""
        synth = Synthesizer(
            target_vars=["income"],
            condition_vars=["age"],
            n_layers=4,
            hidden_dim=64,
        )
        synth.fit(complete_data, epochs=50, verbose=False)

        conditions = complete_data[["age"]]
        generated = synth.generate(conditions, seed=42)

        # Mean should be within 50% of training
        train_mean = complete_data["income"].mean()
        gen_mean = generated["income"].mean()
        ratio = gen_mean / train_mean if train_mean > 0 else 1
        assert 0.5 < ratio < 2.0, f"Mean ratio {ratio:.2f} outside [0.5, 2.0]"
