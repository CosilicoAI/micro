"""Tests for hierarchical household synthesis."""

import numpy as np
import pandas as pd
import pytest

from microplex.hierarchical import (
    HierarchicalSynthesizer,
    HouseholdSchema,
    prepare_cps_for_hierarchical,
)


def create_test_household_data(n_households: int = 100, seed: int = 42) -> tuple:
    """Create synthetic household and person data for testing."""
    np.random.seed(seed)

    # Generate households
    hh_data = pd.DataFrame({
        'household_id': range(n_households),
        'n_persons': np.random.choice([1, 2, 3, 4, 5], n_households, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
        'state_fips': np.random.choice([6, 36, 48], n_households),  # CA, NY, TX
        'tenure': np.random.choice([1, 2], n_households),  # Own, Rent
    })
    hh_data['n_adults'] = np.clip(hh_data['n_persons'] - np.random.randint(0, 3, n_households), 1, hh_data['n_persons'])
    hh_data['n_children'] = hh_data['n_persons'] - hh_data['n_adults']

    # Generate persons for each household
    person_records = []
    person_id = 0

    for _, hh_row in hh_data.iterrows():
        hh_id = hh_row['household_id']
        n_persons = hh_row['n_persons']
        n_adults = hh_row['n_adults']

        for p_num in range(n_persons):
            is_adult = p_num < n_adults

            if is_adult:
                age = np.random.randint(25, 70)
                income = np.random.lognormal(10.5, 0.8)
            else:
                age = np.random.randint(0, 18)
                income = 0

            person_records.append({
                'person_id': person_id,
                'household_id': hh_id,
                'age': age,
                'sex': np.random.choice([0, 1]),
                'income': income,
                'employment_status': 1 if is_adult and np.random.random() > 0.3 else 0,
                'education': np.random.randint(1, 5) if is_adult else 0,
                'relationship_to_head': 0 if p_num == 0 else (1 if p_num == 1 and is_adult else 2),
            })
            person_id += 1

    person_data = pd.DataFrame(person_records)

    return hh_data, person_data


class TestHouseholdSchema:
    """Tests for HouseholdSchema."""

    def test_default_schema(self):
        """Test default schema has expected fields."""
        schema = HouseholdSchema()

        assert 'n_persons' in schema.hh_vars
        assert 'n_adults' in schema.hh_vars
        assert 'age' in schema.person_vars
        assert 'income' in schema.person_vars
        assert 'hh_income' in schema.derived_vars

    def test_custom_schema(self):
        """Test custom schema configuration."""
        schema = HouseholdSchema(
            hh_vars=['n_persons', 'state'],
            person_vars=['age', 'income'],
            derived_vars={'total_income': 'sum:income'},
        )

        assert schema.hh_vars == ['n_persons', 'state']
        assert schema.person_vars == ['age', 'income']
        assert 'total_income' in schema.derived_vars


class TestHierarchicalSynthesizer:
    """Tests for HierarchicalSynthesizer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return create_test_household_data(n_households=50)

    @pytest.fixture
    def simple_schema(self):
        """Create simplified schema for faster tests."""
        return HouseholdSchema(
            hh_vars=['n_persons', 'n_adults', 'n_children'],
            person_vars=['age', 'income'],
            person_condition_vars=['n_persons', 'n_adults', 'person_number', 'is_first_adult'],
            derived_vars={'hh_income': 'sum:income'},
        )

    def test_init(self):
        """Test synthesizer initialization."""
        synth = HierarchicalSynthesizer()
        assert synth.schema is not None
        assert not synth._is_fitted

    def test_init_with_custom_schema(self, simple_schema):
        """Test initialization with custom schema."""
        synth = HierarchicalSynthesizer(schema=simple_schema)
        assert synth.schema == simple_schema

    def test_fit_validates_data(self, sample_data, simple_schema):
        """Test that fit validates required columns."""
        hh_data, person_data = sample_data

        # Remove required column
        bad_hh = hh_data.drop(columns=['n_persons'])

        synth = HierarchicalSynthesizer(schema=simple_schema)
        with pytest.raises(ValueError, match="missing columns"):
            synth.fit(bad_hh, person_data, epochs=1)

    def test_fit_runs(self, sample_data, simple_schema):
        """Test that fit completes without error."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        assert synth._is_fitted
        assert synth.hh_synthesizer is not None
        assert synth.person_synthesizer is not None

    def test_generate_requires_fit(self, simple_schema):
        """Test that generate requires fit first."""
        synth = HierarchicalSynthesizer(schema=simple_schema)

        with pytest.raises(ValueError, match="Must call fit"):
            synth.generate(n_households=10)

    def test_generate_returns_correct_structure(self, sample_data, simple_schema):
        """Test that generate returns expected DataFrames."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        synthetic_hh, synthetic_persons = synth.generate(n_households=20, verbose=False)

        # Check household DataFrame
        assert len(synthetic_hh) == 20
        assert 'household_id' in synthetic_hh.columns
        assert 'n_persons' in synthetic_hh.columns

        # Check person DataFrame
        assert len(synthetic_persons) > 0
        assert 'household_id' in synthetic_persons.columns
        assert 'person_id' in synthetic_persons.columns
        assert 'age' in synthetic_persons.columns
        assert 'income' in synthetic_persons.columns

        # Check every person belongs to a valid household
        assert set(synthetic_persons['household_id']).issubset(set(synthetic_hh['household_id']))

    def test_generate_with_units(self, sample_data, simple_schema):
        """Test generate with tax/SPM unit construction."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        result = synth.generate(n_households=10, return_units=True, verbose=False)

        assert len(result) == 4
        synthetic_hh, synthetic_persons, tax_units, spm_units = result

        # Check tax units
        assert len(tax_units) > 0
        assert 'tax_unit_id' in tax_units.columns
        assert 'household_id' in tax_units.columns

        # Check SPM units
        assert len(spm_units) > 0
        assert 'spm_unit_id' in spm_units.columns
        assert 'household_id' in spm_units.columns

    def test_derived_aggregates(self, sample_data, simple_schema):
        """Test that derived aggregates are computed correctly."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        synthetic_hh, synthetic_persons = synth.generate(n_households=20, verbose=False)

        # Check hh_income is derived
        assert 'hh_income' in synthetic_hh.columns

        # Verify it's the sum of person incomes
        for hh_id in synthetic_hh['household_id'].head(5):
            hh_persons = synthetic_persons[synthetic_persons['household_id'] == hh_id]
            expected_income = hh_persons['income'].sum()
            actual_income = synthetic_hh[synthetic_hh['household_id'] == hh_id]['hh_income'].iloc[0]
            np.testing.assert_almost_equal(actual_income, expected_income, decimal=2)

    def test_person_count_matches_n_persons(self, sample_data, simple_schema):
        """Test that number of persons matches n_persons in HH data."""
        hh_data, person_data = sample_data

        synth = HierarchicalSynthesizer(schema=simple_schema)
        synth.fit(hh_data, person_data, epochs=2, verbose=False)

        synthetic_hh, synthetic_persons = synth.generate(n_households=20, verbose=False)

        # Count persons per household
        person_counts = synthetic_persons.groupby('household_id').size()

        for hh_id in synthetic_hh['household_id']:
            expected = synthetic_hh[synthetic_hh['household_id'] == hh_id]['n_persons'].iloc[0]
            actual = person_counts.get(hh_id, 0)
            assert actual == expected, f"HH {hh_id}: expected {expected} persons, got {actual}"


class TestPrepareCpsForHierarchical:
    """Tests for CPS data preparation utility."""

    def test_basic_preparation(self):
        """Test basic CPS preparation."""
        # Create mock CPS person data
        cps_data = pd.DataFrame({
            'household_id': [1, 1, 1, 2, 2, 3],
            'age': [45, 42, 12, 67, 65, 35],
            'state_fips': [6, 6, 6, 36, 36, 48],
            'tenure': [1, 1, 1, 2, 2, 1],
            'hh_weight': [1000, 1000, 1000, 800, 800, 1200],
        })

        hh_data, person_data = prepare_cps_for_hierarchical(cps_data)

        # Check household data
        assert len(hh_data) == 3
        assert hh_data[hh_data['household_id'] == 1]['n_persons'].iloc[0] == 3
        assert hh_data[hh_data['household_id'] == 1]['n_adults'].iloc[0] == 2
        assert hh_data[hh_data['household_id'] == 1]['n_children'].iloc[0] == 1

        # Check person data unchanged
        assert len(person_data) == 6
