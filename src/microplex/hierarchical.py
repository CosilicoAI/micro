"""
Hierarchical synthesis for household microdata.

Two-pass approach:
1. Synthesize household skeleton (composition, location, tenure)
2. Synthesize person attributes conditioned on household context

Then derive aggregates (HH income = sum of person incomes) and
construct tax units / SPM units algorithmically.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .synthesizer import Synthesizer


@dataclass
class HouseholdSchema:
    """Schema defining household and person variables."""

    # Household-level variables (Pass 1)
    hh_vars: List[str] = field(default_factory=lambda: [
        'n_persons', 'n_adults', 'n_children', 'state_fips', 'tenure'
    ])

    # Person-level variables to synthesize (Pass 2)
    person_vars: List[str] = field(default_factory=lambda: [
        'age', 'sex', 'income', 'employment_status', 'education',
        'relationship_to_head'
    ])

    # Person-level conditioning variables (from HH + position)
    person_condition_vars: List[str] = field(default_factory=lambda: [
        'n_persons', 'n_adults', 'n_children', 'state_fips', 'tenure',
        'person_number', 'is_first_adult', 'is_child_slot'
    ])

    # Variables to derive by aggregation (not modeled)
    derived_vars: Dict[str, str] = field(default_factory=lambda: {
        'hh_income': 'sum:income',
        'hh_benefits': 'sum:benefits',
        'n_workers': 'count:employment_status==1',
    })

    # ID columns
    hh_id_col: str = 'household_id'
    person_id_col: str = 'person_id'


class HierarchicalSynthesizer:
    """
    Two-pass hierarchical synthesizer for household microdata.

    Pass 1: Learn P(household_features) from data
    Pass 2: Learn P(person_attributes | household_features) from data

    Then:
    - Generate households
    - Generate persons per household
    - Derive aggregates (HH income = sum of person incomes)
    - Construct tax units / SPM units algorithmically

    Example:
        >>> synth = HierarchicalSynthesizer()
        >>> synth.fit(hh_data, person_data)
        >>> synthetic_hh, synthetic_persons = synth.generate(n_households=10000)
    """

    def __init__(
        self,
        schema: Optional[HouseholdSchema] = None,
        hh_flow_kwargs: Optional[Dict] = None,
        person_flow_kwargs: Optional[Dict] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize hierarchical synthesizer.

        Args:
            schema: HouseholdSchema defining variables at each level
            hh_flow_kwargs: Kwargs passed to household-level Synthesizer
            person_flow_kwargs: Kwargs passed to person-level Synthesizer
            random_state: Random seed for reproducibility
        """
        self.schema = schema or HouseholdSchema()
        self.hh_flow_kwargs = hh_flow_kwargs or {}
        self.person_flow_kwargs = person_flow_kwargs or {}
        self.random_state = random_state

        self.hh_synthesizer: Optional[Synthesizer] = None
        self.person_synthesizer: Optional[Synthesizer] = None

        self._hh_data: Optional[pd.DataFrame] = None
        self._person_data: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
        hh_weight_col: Optional[str] = None,
        person_weight_col: Optional[str] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> 'HierarchicalSynthesizer':
        """
        Fit the two-pass hierarchical model.

        Args:
            hh_data: Household-level data (one row per household)
            person_data: Person-level data (one row per person, with HH ID)
            hh_weight_col: Weight column for households
            person_weight_col: Weight column for persons
            epochs: Training epochs for each flow
            verbose: Print progress

        Returns:
            self
        """
        self._hh_data = hh_data.copy()
        self._person_data = person_data.copy()

        # Validate schema
        self._validate_data()

        # Prepare person data with position features
        person_with_position = self._add_position_features(person_data, hh_data)

        # Pass 1: Fit household-level synthesizer
        if verbose:
            print("=" * 60)
            print("PASS 1: Fitting household-level model")
            print("=" * 60)
            print(f"  Variables: {self.schema.hh_vars}")
            print(f"  N households: {len(hh_data):,}")

        self.hh_synthesizer = Synthesizer(
            target_vars=self.schema.hh_vars,
            condition_vars=[],  # Unconditional for now
            **self.hh_flow_kwargs
        )
        self.hh_synthesizer.fit(
            hh_data,
            weight_col=hh_weight_col,
            epochs=epochs,
        )

        # Pass 2: Fit person-level synthesizer
        if verbose:
            print("\n" + "=" * 60)
            print("PASS 2: Fitting person-level model")
            print("=" * 60)
            print(f"  Target vars: {self.schema.person_vars}")
            print(f"  Condition vars: {self.schema.person_condition_vars}")
            print(f"  N persons: {len(person_data):,}")

        # Filter to available condition vars
        available_condition_vars = [
            v for v in self.schema.person_condition_vars
            if v in person_with_position.columns
        ]

        self.person_synthesizer = Synthesizer(
            target_vars=self.schema.person_vars,
            condition_vars=available_condition_vars,
            **self.person_flow_kwargs
        )
        self.person_synthesizer.fit(
            person_with_position,
            weight_col=person_weight_col,
            epochs=epochs,
        )

        self._is_fitted = True

        if verbose:
            print("\n" + "=" * 60)
            print("HIERARCHICAL MODEL FITTED")
            print("=" * 60)

        return self

    def generate(
        self,
        n_households: int,
        return_units: bool = False,
        verbose: bool = True,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
               Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Generate synthetic households and persons.

        Args:
            n_households: Number of households to generate
            return_units: If True, also return tax_units and spm_units DataFrames
            verbose: Print progress

        Returns:
            (households, persons) DataFrames, or
            (households, persons, tax_units, spm_units) if return_units=True
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before generate()")

        # Pass 1: Generate households
        if verbose:
            print(f"Generating {n_households:,} synthetic households...")

        # Generate household features
        # For unconditional generation, we need dummy conditioning data
        dummy_context = pd.DataFrame({'_dummy': np.zeros(n_households)})
        synthetic_hh = self.hh_synthesizer.generate(dummy_context)

        # Add household IDs
        synthetic_hh[self.schema.hh_id_col] = np.arange(n_households)

        # Ensure integer counts
        for col in ['n_persons', 'n_adults', 'n_children']:
            if col in synthetic_hh.columns:
                synthetic_hh[col] = np.clip(
                    np.round(synthetic_hh[col]).astype(int), 1, 20
                )

        # Pass 2: Generate persons for each household
        if verbose:
            print("Generating persons for each household...")

        person_records = []
        person_id = 0

        for hh_idx, hh_row in synthetic_hh.iterrows():
            n_persons = int(hh_row.get('n_persons', 1))
            n_adults = int(hh_row.get('n_adults', 1))

            # Create conditioning context for each person in this HH
            for p_num in range(n_persons):
                context = {
                    self.schema.hh_id_col: hh_row[self.schema.hh_id_col],
                    self.schema.person_id_col: person_id,
                    'person_number': p_num,
                    'is_first_adult': p_num == 0,
                    'is_child_slot': p_num >= n_adults,
                }
                # Add HH-level features to context
                for var in self.schema.hh_vars:
                    if var in hh_row.index:
                        context[var] = hh_row[var]

                person_records.append(context)
                person_id += 1

        # Convert to DataFrame
        person_context = pd.DataFrame(person_records)

        if verbose:
            print(f"  Total persons: {len(person_context):,}")
            print(f"  Avg HH size: {len(person_context) / n_households:.2f}")

        # Generate person attributes
        synthetic_persons = self.person_synthesizer.generate(person_context)

        # Add IDs and context back
        synthetic_persons[self.schema.hh_id_col] = person_context[self.schema.hh_id_col].values
        synthetic_persons[self.schema.person_id_col] = person_context[self.schema.person_id_col].values

        # Derive aggregates
        if verbose:
            print("Deriving household aggregates...")
        synthetic_hh = self._derive_aggregates(synthetic_hh, synthetic_persons)

        if return_units:
            if verbose:
                print("Constructing tax units and SPM units...")
            tax_units = self._construct_tax_units(synthetic_hh, synthetic_persons)
            spm_units = self._construct_spm_units(synthetic_hh, synthetic_persons)
            return synthetic_hh, synthetic_persons, tax_units, spm_units

        return synthetic_hh, synthetic_persons

    def _validate_data(self) -> None:
        """Validate that data has required columns."""
        hh_missing = set(self.schema.hh_vars) - set(self._hh_data.columns)
        if hh_missing:
            raise ValueError(f"Household data missing columns: {hh_missing}")

        person_missing = set(self.schema.person_vars) - set(self._person_data.columns)
        if person_missing:
            raise ValueError(f"Person data missing columns: {person_missing}")

        if self.schema.hh_id_col not in self._person_data.columns:
            raise ValueError(
                f"Person data must have household ID column: {self.schema.hh_id_col}"
            )

    def _add_position_features(
        self,
        person_data: pd.DataFrame,
        hh_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add position-within-household features to person data."""
        df = person_data.copy()

        # Person number within household
        df['person_number'] = df.groupby(self.schema.hh_id_col).cumcount()

        # Merge HH features
        hh_features = hh_data[[self.schema.hh_id_col] + [
            v for v in self.schema.hh_vars if v in hh_data.columns
        ]].copy()

        if self.schema.hh_id_col in hh_features.columns:
            df = df.merge(hh_features, on=self.schema.hh_id_col, how='left')

        # Compute position features
        n_adults = df.get('n_adults', 1)
        df['is_first_adult'] = df['person_number'] == 0
        df['is_child_slot'] = df['person_number'] >= n_adults

        return df

    def _derive_aggregates(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Derive household aggregates from person data."""
        hh = hh_data.copy()

        for derived_name, formula in self.schema.derived_vars.items():
            if ':' in formula:
                agg_type, var_expr = formula.split(':', 1)

                if agg_type == 'sum':
                    if var_expr in person_data.columns:
                        agg = person_data.groupby(self.schema.hh_id_col)[var_expr].sum()
                        hh[derived_name] = hh[self.schema.hh_id_col].map(agg).fillna(0)

                elif agg_type == 'count':
                    if '==' in var_expr:
                        var, val = var_expr.split('==')
                        mask = person_data[var.strip()] == int(val)
                        counts = person_data[mask].groupby(self.schema.hh_id_col).size()
                        hh[derived_name] = hh[self.schema.hh_id_col].map(counts).fillna(0)

        return hh

    def _construct_tax_units(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct tax units from persons.

        Simple heuristic for now:
        - Each married couple is a tax unit
        - Each single adult is a tax unit
        - Children assigned to parent tax units

        TODO: Optimize for minimum tax liability
        """
        tax_units = []
        tu_id = 0

        for hh_id in hh_data[self.schema.hh_id_col].unique():
            hh_persons = person_data[
                person_data[self.schema.hh_id_col] == hh_id
            ].copy()

            # Simple: first person is head, spouse if exists, rest are dependents
            # This is a placeholder - real logic would be more sophisticated
            n_persons = len(hh_persons)

            if n_persons == 0:
                continue

            # For now, one tax unit per household (simplified)
            tax_units.append({
                'tax_unit_id': tu_id,
                self.schema.hh_id_col: hh_id,
                'n_members': n_persons,
                'filing_status': 'married_joint' if n_persons >= 2 else 'single',
            })
            tu_id += 1

        return pd.DataFrame(tax_units)

    def _construct_spm_units(
        self,
        hh_data: pd.DataFrame,
        person_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct SPM (Supplemental Poverty Measure) units from persons.

        SPM unit rules (simplified):
        - All related individuals in household
        - Includes cohabiting partners and their relatives
        - Foster children included with foster families

        For now: SPM unit = household (simplified)
        """
        spm_units = []

        for idx, hh_row in hh_data.iterrows():
            hh_id = hh_row[self.schema.hh_id_col]
            hh_persons = person_data[
                person_data[self.schema.hh_id_col] == hh_id
            ]

            spm_units.append({
                'spm_unit_id': idx,
                self.schema.hh_id_col: hh_id,
                'n_members': len(hh_persons),
            })

        return pd.DataFrame(spm_units)


def prepare_cps_for_hierarchical(
    cps_person_data: pd.DataFrame,
    hh_id_col: str = 'household_id',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare CPS data for hierarchical synthesis.

    Takes person-level CPS data and creates:
    1. Household-level summary (one row per HH)
    2. Person-level data with position features

    Args:
        cps_person_data: CPS person-level data
        hh_id_col: Household ID column

    Returns:
        (hh_data, person_data) tuple
    """
    df = cps_person_data.copy()

    # Create household-level summary
    hh_agg = df.groupby(hh_id_col).agg({
        'age': ['count', lambda x: (x >= 18).sum(), lambda x: (x < 18).sum()],
    })
    hh_agg.columns = ['n_persons', 'n_adults', 'n_children']
    hh_agg = hh_agg.reset_index()

    # Add other HH-level vars (take first value per HH)
    hh_level_vars = ['state_fips', 'tenure', 'hh_weight']
    for var in hh_level_vars:
        if var in df.columns:
            first_vals = df.groupby(hh_id_col)[var].first()
            hh_agg[var] = hh_agg[hh_id_col].map(first_vals)

    return hh_agg, df
