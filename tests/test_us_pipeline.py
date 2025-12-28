"""Tests for US Microplex pipeline."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipelines"))


class TestUSPipeline:
    """Test the US Microplex pipeline."""

    @pytest.fixture
    def mock_cps_data(self):
        """Create mock CPS-like data for testing."""
        np.random.seed(42)
        n = 1000

        return pd.DataFrame({
            "age": np.random.randint(18, 80, n),
            "sex": np.random.randint(0, 2, n),
            "race": np.random.randint(1, 6, n),
            "marital_status": np.random.randint(1, 7, n),
            "state_fips": np.random.randint(1, 51, n),
            "employment_status": np.random.randint(0, 3, n),
            "wage_salary_income": np.where(
                np.random.random(n) > 0.3,
                np.abs(np.random.normal(50000, 30000, n)),
                0
            ),
            "self_employment_income": np.where(
                np.random.random(n) > 0.9,
                np.abs(np.random.normal(20000, 15000, n)),
                0
            ),
            "interest_income": np.where(
                np.random.random(n) > 0.5,
                np.abs(np.random.normal(2000, 3000, n)),
                0
            ),
            "dividend_income": np.where(
                np.random.random(n) > 0.8,
                np.abs(np.random.normal(3000, 5000, n)),
                0
            ),
            "social_security_income": np.where(
                np.random.random(n) > 0.8,
                np.abs(np.random.normal(15000, 5000, n)),
                0
            ),
            "unemployment_compensation": np.where(
                np.random.random(n) > 0.95,
                np.abs(np.random.normal(5000, 3000, n)),
                0
            ),
            "weight": np.random.uniform(1000, 5000, n),
        })

    def test_pipeline_runs(self, mock_cps_data):
        """Pipeline should run without error on mock data."""
        from microplex import Synthesizer

        context_vars = ["age", "sex", "race", "marital_status", "state_fips", "employment_status"]
        target_vars = ["wage_salary_income", "self_employment_income", "interest_income"]

        synth = Synthesizer(
            target_vars=target_vars,
            condition_vars=context_vars,
            n_layers=2,
            hidden_dim=32,
        )
        synth.fit(mock_cps_data, weight_col="weight", epochs=10, verbose=False)

        result = synth.sample(100, seed=42)

        assert len(result) == 100
        assert all(v in result.columns for v in context_vars + target_vars)

    def test_generates_non_negative_income(self, mock_cps_data):
        """All generated incomes should be non-negative."""
        from microplex import Synthesizer

        target_vars = ["wage_salary_income", "self_employment_income"]

        synth = Synthesizer(
            target_vars=target_vars,
            condition_vars=["age", "employment_status"],
            n_layers=2,
            hidden_dim=32,
        )
        synth.fit(mock_cps_data, epochs=20, verbose=False)

        result = synth.sample(500, seed=42)

        for var in target_vars:
            assert (result[var] >= 0).all(), f"{var} has negative values"

    def test_preserves_zero_inflation(self, mock_cps_data):
        """Generated data should have similar zero rates to training."""
        from microplex import Synthesizer

        target_vars = ["wage_salary_income"]

        synth = Synthesizer(
            target_vars=target_vars,
            condition_vars=["age", "employment_status"],
            n_layers=4,
            hidden_dim=64,
        )
        synth.fit(mock_cps_data, epochs=50, verbose=False)

        result = synth.sample(1000, seed=42)

        train_zero_rate = (mock_cps_data["wage_salary_income"] == 0).mean()
        gen_zero_rate = (result["wage_salary_income"] == 0).mean()

        # Allow 20 percentage point difference
        assert abs(train_zero_rate - gen_zero_rate) < 0.2, \
            f"Zero rates differ too much: train={train_zero_rate:.2f}, gen={gen_zero_rate:.2f}"
