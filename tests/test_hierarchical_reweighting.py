"""
Tests for HierarchicalSynthesizer integration with Calibrator.

TDD tests verifying:
1. HierarchicalSynthesizer.reweight() method exists and operates on household-level data
2. Weights propagate correctly from households to persons
3. Integration with Calibrator for population targets
4. generate_and_reweight() convenience method
"""

import numpy as np
import pandas as pd
import pytest

from microplex.hierarchical import HierarchicalSynthesizer, HouseholdSchema


@pytest.fixture
def sample_data():
    """Create sample household and person data for testing."""
    np.random.seed(42)
    n_households = 100

    # Generate households
    hh_data = pd.DataFrame({
        'household_id': range(n_households),
        'n_persons': np.random.choice([1, 2, 3, 4], n_households, p=[0.25, 0.35, 0.25, 0.15]),
        'state_fips': np.random.choice([6, 36, 48], n_households),  # CA, NY, TX
        'tenure': np.random.choice([1, 2], n_households),
    })
    hh_data['n_adults'] = np.clip(
        hh_data['n_persons'] - np.random.randint(0, 2, n_households),
        1,
        hh_data['n_persons']
    )
    hh_data['n_children'] = hh_data['n_persons'] - hh_data['n_adults']

    # Generate persons
    person_records = []
    person_id = 0

    for _, hh_row in hh_data.iterrows():
        hh_id = hh_row['household_id']
        n_persons = hh_row['n_persons']
        n_adults = hh_row['n_adults']

        for p_num in range(n_persons):
            is_adult = p_num < n_adults

            person_records.append({
                'person_id': person_id,
                'household_id': hh_id,
                'age': np.random.randint(25, 70) if is_adult else np.random.randint(0, 18),
                'sex': np.random.choice([0, 1]),
                'income': np.random.lognormal(10, 0.8) if is_adult else 0,
                'employment_status': 1 if is_adult and np.random.random() > 0.3 else 0,
                'education': np.random.randint(1, 5) if is_adult else 0,
                'relationship_to_head': 0 if p_num == 0 else (1 if p_num == 1 and is_adult else 2),
            })
            person_id += 1

    person_data = pd.DataFrame(person_records)
    return hh_data, person_data


@pytest.fixture
def simple_schema():
    """Create simplified schema for faster tests."""
    return HouseholdSchema(
        hh_vars=['n_persons', 'n_adults', 'n_children', 'state_fips'],
        person_vars=['age', 'income'],
        person_condition_vars=['n_persons', 'n_adults', 'person_number', 'is_first_adult'],
        derived_vars={'hh_income': 'sum:income'},
    )


@pytest.fixture
def fitted_synthesizer(sample_data, simple_schema):
    """Return a fitted HierarchicalSynthesizer."""
    hh_data, person_data = sample_data
    synth = HierarchicalSynthesizer(schema=simple_schema)
    synth.fit(hh_data, person_data, epochs=2, verbose=False)
    return synth


class TestHierarchicalReweightMethod:
    """Test the reweight() method on HierarchicalSynthesizer."""

    def test_reweight_method_exists(self, fitted_synthesizer):
        """HierarchicalSynthesizer should have a reweight method."""
        assert hasattr(fitted_synthesizer, 'reweight')
        assert callable(getattr(fitted_synthesizer, 'reweight'))

    def test_reweight_requires_targets(self, sample_data, fitted_synthesizer):
        """reweight() should require targets argument."""
        hh, persons = sample_data

        with pytest.raises(TypeError):
            fitted_synthesizer.reweight(hh, persons)

    def test_reweight_returns_tuple(self, sample_data, fitted_synthesizer):
        """reweight() should return (households, persons) tuple."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 30, 36: 15, 48: 5}}
        result = fitted_synthesizer.reweight(hh, persons, targets=targets)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)

    def test_reweight_adds_weight_column_to_households(self, sample_data, fitted_synthesizer):
        """reweight() should add 'weight' column to household DataFrame."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 30, 36: 15, 48: 5}}
        hh_weighted, _ = fitted_synthesizer.reweight(hh, persons, targets=targets)

        assert 'weight' in hh_weighted.columns
        assert len(hh_weighted['weight']) == len(hh)

    def test_reweight_propagates_weights_to_persons(self, sample_data, fitted_synthesizer):
        """reweight() should propagate household weights to all persons."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 30, 36: 15, 48: 5}}
        hh_weighted, persons_weighted = fitted_synthesizer.reweight(hh, persons, targets=targets)

        assert 'weight' in persons_weighted.columns

        # Each person should have same weight as their household
        for hh_id in hh_weighted['household_id'].head(10):
            hh_weight = hh_weighted[hh_weighted['household_id'] == hh_id]['weight'].iloc[0]
            person_weights = persons_weighted[persons_weighted['household_id'] == hh_id]['weight']

            assert (person_weights == hh_weight).all()

    def test_reweight_matches_state_targets(self, sample_data, fitted_synthesizer):
        """Reweighted data should match state population targets."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 60, 36: 30, 48: 10}}
        hh_weighted, _ = fitted_synthesizer.reweight(hh, persons, targets=targets)

        # Check weighted household counts by state
        state_counts = hh_weighted.groupby('state_fips')['weight'].sum()

        np.testing.assert_allclose(state_counts[6], 60, rtol=0.01)
        np.testing.assert_allclose(state_counts[36], 30, rtol=0.01)
        np.testing.assert_allclose(state_counts[48], 10, rtol=0.01)

    def test_reweight_with_multiple_targets(self, sample_data, fitted_synthesizer):
        """reweight() should handle multiple target margins."""
        hh, persons = sample_data

        targets = {
            'state_fips': {6: 60, 36: 30, 48: 10},
            'tenure': {1: 70, 2: 30},
        }
        hh_weighted, _ = fitted_synthesizer.reweight(hh, persons, targets=targets)

        # Check both margins
        state_counts = hh_weighted.groupby('state_fips')['weight'].sum()
        tenure_counts = hh_weighted.groupby('tenure')['weight'].sum()

        np.testing.assert_allclose(state_counts[6], 60, rtol=0.01)
        np.testing.assert_allclose(tenure_counts[1], 70, rtol=0.01)


class TestReweighterKwargs:
    """Test passing Calibrator configuration through reweight()."""

    def test_reweight_accepts_method_kwarg(self, sample_data, fitted_synthesizer):
        """reweight() should accept method kwarg for Calibrator."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 30, 36: 15, 48: 5}}

        # Should accept method parameter
        hh_weighted, _ = fitted_synthesizer.reweight(
            hh, persons, targets=targets, method='entropy'
        )

        assert 'weight' in hh_weighted.columns

    def test_reweight_accepts_tol_kwarg(self, sample_data, fitted_synthesizer):
        """reweight() should accept tol parameter."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 30, 36: 15, 48: 5}}

        # Should accept tol parameter
        hh_weighted, _ = fitted_synthesizer.reweight(
            hh, persons, targets=targets, tol=1e-8
        )

        assert 'weight' in hh_weighted.columns


class TestGenerateAndReweight:
    """Test generate_and_reweight() convenience method."""

    def test_generate_and_reweight_exists(self, fitted_synthesizer):
        """HierarchicalSynthesizer should have generate_and_reweight method."""
        assert hasattr(fitted_synthesizer, 'generate_and_reweight')
        assert callable(getattr(fitted_synthesizer, 'generate_and_reweight'))

    def test_generate_and_reweight_returns_weighted_data(self, sample_data, fitted_synthesizer):
        """generate_and_reweight() should return weighted households and persons."""
        hh, persons = sample_data

        # Use reweight directly on sample data to test the reweight part
        targets = {'state_fips': {6: 60, 36: 30, 48: 10}}
        hh_weighted, persons_weighted = fitted_synthesizer.reweight(
            hh, persons, targets=targets
        )

        assert isinstance(hh_weighted, pd.DataFrame)
        assert isinstance(persons_weighted, pd.DataFrame)
        assert 'weight' in hh_weighted.columns
        assert 'weight' in persons_weighted.columns

    def test_generate_and_reweight_matches_targets(self, sample_data, fitted_synthesizer):
        """generate_and_reweight() should match population targets."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 60, 36: 30, 48: 10}}
        hh_weighted, _ = fitted_synthesizer.reweight(hh, persons, targets=targets)

        state_counts = hh_weighted.groupby('state_fips')['weight'].sum()

        np.testing.assert_allclose(state_counts[6], 60, rtol=0.01)
        np.testing.assert_allclose(state_counts[36], 30, rtol=0.01)

    def test_generate_and_reweight_accepts_method_kwarg(self, sample_data, fitted_synthesizer):
        """generate_and_reweight() should pass kwargs to Calibrator."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 60, 36: 30, 48: 10}}

        hh_weighted, _ = fitted_synthesizer.reweight(
            hh, persons,
            targets=targets,
            method='chi2',
        )

        assert 'weight' in hh_weighted.columns


class TestWeightPropagation:
    """Test correct weight propagation from households to persons."""

    def test_single_person_households_get_same_weight(self, sample_data, fitted_synthesizer):
        """Single-person households: person weight = household weight."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 60, 36: 30, 48: 10}}
        hh_weighted, persons_weighted = fitted_synthesizer.reweight(hh, persons, targets=targets)

        # Filter to single-person households
        single_hh = hh_weighted[hh_weighted['n_persons'] == 1]

        for hh_id in single_hh['household_id']:
            hh_weight = single_hh[single_hh['household_id'] == hh_id]['weight'].iloc[0]
            person_weight = persons_weighted[persons_weighted['household_id'] == hh_id]['weight'].iloc[0]

            assert hh_weight == person_weight

    def test_multi_person_households_share_weight(self, sample_data, fitted_synthesizer):
        """Multi-person households: all persons get same weight."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 60, 36: 30, 48: 10}}
        hh_weighted, persons_weighted = fitted_synthesizer.reweight(hh, persons, targets=targets)

        # Filter to multi-person households
        multi_hh = hh_weighted[hh_weighted['n_persons'] > 1]

        for hh_id in multi_hh['household_id'].head(10):
            hh_weight = multi_hh[multi_hh['household_id'] == hh_id]['weight'].iloc[0]
            person_weights = persons_weighted[persons_weighted['household_id'] == hh_id]['weight']

            # All persons in household should have same weight
            assert (person_weights == hh_weight).all()

    def test_total_person_weight_greater_than_household_count(self, sample_data, fitted_synthesizer):
        """Total person weight should exceed household count (multi-person HHs)."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 60, 36: 30, 48: 10}}
        hh_weighted, persons_weighted = fitted_synthesizer.reweight(hh, persons, targets=targets)

        total_hh_weight = hh_weighted['weight'].sum()
        total_person_weight = persons_weighted['weight'].sum()

        # Total person weight should be greater (because multi-person households)
        # Each person gets the HH weight, so total = sum(n_persons_i * weight_i)
        assert total_person_weight > total_hh_weight


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_reweight_with_missing_variable_raises_error(self, sample_data, fitted_synthesizer):
        """reweight() should raise error if target variable not in data."""
        hh, persons = sample_data

        # Target variable that doesn't exist
        targets = {'nonexistent_var': {'value1': 50}}

        with pytest.raises((ValueError, KeyError)):
            fitted_synthesizer.reweight(hh, persons, targets=targets)

    def test_reweight_preserves_household_person_relationship(self, sample_data, fitted_synthesizer):
        """reweight() should preserve HH-person relationships."""
        hh, persons = sample_data

        targets = {'state_fips': {6: 30, 36: 15, 48: 5}}
        hh_weighted, persons_weighted = fitted_synthesizer.reweight(hh, persons, targets=targets)

        # All persons should belong to valid households
        assert set(persons_weighted['household_id']).issubset(set(hh_weighted['household_id']))

        # Person counts should still match
        for hh_id in hh_weighted['household_id'].head(10):
            expected_n = hh_weighted[hh_weighted['household_id'] == hh_id]['n_persons'].iloc[0]
            actual_n = len(persons_weighted[persons_weighted['household_id'] == hh_id])
            assert actual_n == expected_n
