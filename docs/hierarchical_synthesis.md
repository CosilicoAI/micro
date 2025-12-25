# Hierarchical Microdata Synthesis

## The Problem

Real microdata has hierarchical structure:

```
Household
├── Tax Unit 1
│   ├── Person 1 (Head)
│   ├── Person 2 (Spouse)
│   └── Person 3 (Dependent)
└── Tax Unit 2
    └── Person 4 (Adult child)
```

Variables exist at each level:
- **Household**: Total income, dwelling type, region
- **Tax Unit**: Tax liability, EITC eligibility, filing status
- **Person**: Age, earnings, education, employment

Relationships matter:
- Spouse ages are correlated
- Children's ages depend on parent ages
- All members must share the same household income
- Tax unit aggregates must sum to household totals

The question: Should microplex flatten this to single records or directly model the hierarchy?

## Approach 1: Flattening

### The Idea

Represent each person as a row with household/tax unit variables copied:

```
person_id | age | household_income | tax_unit_income | region
----------|-----|------------------|-----------------|--------
1         | 42  | 120000          | 100000          | CA
2         | 40  | 120000          | 100000          | CA
3         | 8   | 120000          | 100000          | CA
4         | 22  | 120000          | 20000           | CA
```

### Implementation

Current microplex already supports this:

```python
from microplex import Synthesizer

# Flatten CPS to person-level
flattened = []
for household in cps_households:
    for person in household.members:
        flattened.append({
            # Person-level
            "age": person.age,
            "earnings": person.earnings,
            "education": person.education,

            # Copied from household
            "household_income": household.income,
            "household_size": len(household.members),
            "region": household.region,

            # Copied from tax unit
            "tax_unit_income": person.tax_unit.income,
            "filing_status": person.tax_unit.filing_status,
        })

# Synthesize
synth = Synthesizer(
    target_vars=["earnings", "household_income", "tax_unit_income"],
    condition_vars=["age", "education", "region"],
)
synth.fit(pd.DataFrame(flattened))
```

### Pros

1. **Simple**: Works with existing microplex API
2. **No structural constraints**: Just learn conditional distributions
3. **Proven**: This is how most survey microdata is released (PUMS, CPS)
4. **Fast**: Single model, no hierarchical sampling

### Cons

1. **Loses within-household correlations**: Can't ensure spouse ages are realistic
2. **No consistency enforcement**: Can generate different household_income for same household
3. **Inefficient**: Copies household-level variables N times (storage + memory)
4. **Can't enforce aggregation**: Tax unit incomes might not sum to household income

## Approach 2: Hierarchical Synthesis

### 2a. Top-Down Sequential Sampling

Generate in order: household → tax units → persons

```python
# Pseudocode
class HierarchicalSynthesizer:
    def __init__(self):
        # Three separate models
        self.household_model = Synthesizer(
            target_vars=["household_income", "n_tax_units"],
            condition_vars=["region", "year"],
        )
        self.tax_unit_model = Synthesizer(
            target_vars=["tax_unit_income", "filing_status", "n_persons"],
            condition_vars=["household_income", "region"],
        )
        self.person_model = Synthesizer(
            target_vars=["age", "earnings", "education"],
            condition_vars=["tax_unit_income", "filing_status"],
        )

    def generate(self, n_households):
        # 1. Sample households
        households = self.household_model.generate(n_households)

        # 2. For each household, sample tax units
        tax_units = []
        for hh in households:
            n_tu = int(hh["n_tax_units"])
            tu = self.tax_unit_model.generate(
                conditions=pd.DataFrame([hh] * n_tu)
            )
            tax_units.append(tu)

        # 3. For each tax unit, sample persons
        persons = []
        for tu in tax_units:
            n_persons = int(tu["n_persons"])
            p = self.person_model.generate(
                conditions=pd.DataFrame([tu] * n_persons)
            )
            persons.append(p)

        return households, tax_units, persons
```

**Pros:**
- Natural modeling of causality (household determines tax units, etc.)
- Can enforce aggregation constraints at each level
- Memory efficient (no duplication)

**Cons:**
- Complex: Three models to train
- Error propagation: Mistakes at household level affect all downstream
- Training data requirements: Need nested structure in training data

### 2b. Copula-Based Hierarchical Synthesis

Use copulas to model cross-level dependencies.

From [Copula-Based Transferable Models](https://arxiv.org/abs/2302.09193):
- Separate marginal distributions from dependency structure
- Model household structure using vine copulas
- Can transfer learned dependencies to new populations

```python
# Conceptual (would need implementation)
class CopulaHierarchicalSynthesizer:
    def __init__(self):
        # Learn marginals at each level
        self.household_marginals = fit_marginals(household_vars)
        self.person_marginals = fit_marginals(person_vars)

        # Learn dependency structure
        self.within_household_copula = VineCopula()
        self.cross_level_copula = VineCopula()

    def generate(self, n):
        # Sample from copula, then transform to marginals
        u = self.copula.sample(n)
        x = self.marginals.inverse_cdf(u)
        return x
```

**Pros:**
- Preserves all correlations (within and across levels)
- Mathematically rigorous
- Transferable to new populations

**Cons:**
- High-dimensional copulas are complex
- Hard to enforce hard constraints (e.g., child age &lt; parent age)
- Training requires significant sample size

### 2c. Graph Neural Networks for Household Structure

From [University of Oxford student project](https://www.cs.ox.ac.uk/teaching/studentprojects/923.html):
- Represent households as graphs
- Nodes = persons
- Edges = relationships (spouse, parent-child)
- Use GNN to predict realistic household compositions

```python
# Conceptual
class GNNHouseholdSynthesizer:
    def __init__(self):
        self.person_generator = Synthesizer(target_vars=["age", "earnings"])
        self.structure_predictor = GraphNeuralNetwork()

    def generate(self, n_households):
        # 1. Generate persons independently
        persons = self.person_generator.generate(n_total_persons)

        # 2. Use GNN to cluster into realistic households
        household_graphs = self.structure_predictor.predict(persons)

        return household_graphs
```

**Pros:**
- Learns complex household structure patterns
- Can enforce relationship constraints
- State-of-the-art for spatial microsimulation

**Cons:**
- Bleeding edge (not production-ready)
- Requires graph-structured training data
- Computationally expensive

## Approach 3: Hybrid (Recommended for microplex v1)

### The Pragmatic Middle Ground

**For synthesis:** Use flattened representation
**For consistency:** Add post-processing to fix violations

```python
class HybridSynthesizer:
    def __init__(self):
        self.base = Synthesizer(
            target_vars=["age", "earnings", "household_income"],
            condition_vars=["education", "region"],
        )

    def generate(self, conditions, household_sizes):
        # 1. Generate persons with flattened approach
        synthetic = self.base.generate(conditions)

        # 2. Post-process to enforce consistency
        synthetic = self._enforce_household_consistency(
            synthetic, household_sizes
        )

        return synthetic

    def _enforce_household_consistency(self, data, household_sizes):
        # Group by household
        households = []
        start = 0
        for size in household_sizes:
            hh = data.iloc[start:start+size].copy()

            # All members get same household income
            hh["household_income"] = hh["household_income"].mean()

            # Ensure spouse ages are reasonable
            if size >= 2:
                ages = hh["age"].values[:2]
                if abs(ages[0] - ages[1]) > 20:
                    # Adjust to be within 20 years
                    hh.loc[1, "age"] = ages[0] + np.random.randint(-10, 10)

            households.append(hh)
            start += size

        return pd.concat(households)
```

### Why This Works

1. **Leverages existing microplex**: No architectural changes needed
2. **Handles 90% of constraints**: Post-processing fixes obvious violations
3. **Incremental path**: Can add more sophisticated approaches later
4. **Practical**: Microsimulation models mostly treat households atomically anyway

### Limitations

- Post-processing can't enforce complex joint distributions
- May slightly distort learned correlations
- Not guaranteed to match all training data properties

## How PolicyEngine Handles This

PolicyEngine uses a different approach entirely - it doesn't synthesize microdata at all. Instead:

1. **Source microdata** (CPS/ACS) already has hierarchical structure
2. **Entity definitions** in [PolicyEngine Core](https://github.com/PolicyEngine):
   ```python
   class Person(Entity):
       key = "person"

   class TaxUnit(GroupEntity):
       key = "tax_unit"
       roles = [Head, Spouse, Dependent]

   class Household(GroupEntity):
       key = "household"
   ```

3. **Projectors** handle cross-level calculations:
   - `entity_to_person_projector`: Broadcast household income to all members
   - `first_person_to_entity_projector`: Use head's age for tax unit age

4. **Reweighting** operates at household level:
   ```python
   # Reweight to match state populations
   reweight(microdata, targets={"state": {...}})
   ```

**Key insight:** For tax-benefit microsimulation, you rarely need to synthesize new household structures. You just reweight existing ones to match population margins.

## Recommendations

### For microplex v1: Hybrid Flattening + Post-Processing

```python
# Example workflow
from microplex import Synthesizer, enforce_hierarchy

# 1. Train on flattened data
synth = Synthesizer(
    target_vars=["age", "earnings", "household_income"],
    condition_vars=["education", "region"],
)
synth.fit(flattened_cps)

# 2. Generate
synthetic = synth.generate(new_demographics)

# 3. Enforce consistency (new utility function)
synthetic = enforce_hierarchy(
    synthetic,
    household_id="household_id",
    shared_vars=["household_income", "region"],
    relationship_constraints={
        "spouse_age_diff": {"max": 20},
        "child_age": {"max_parent_diff": 18},
    }
)
```

**Implementation effort:** Low (1-2 weeks)
- Modify `Synthesizer.generate()` to accept `household_id`
- Add `enforce_hierarchy()` utility function
- Document hierarchical use cases

### For microplex v2: Top-Down Hierarchical

When you need true joint modeling:

```python
from microplex import HierarchicalSynthesizer

synth = HierarchicalSynthesizer(
    levels={
        "household": {
            "target_vars": ["income", "size"],
            "condition_vars": ["region"],
        },
        "person": {
            "target_vars": ["age", "earnings"],
            "condition_vars": ["household_income", "household_size"],
            "parent_level": "household",
        },
    }
)
```

**Implementation effort:** Medium (4-6 weeks)
- New `HierarchicalSynthesizer` class
- Training pipeline for nested models
- Aggregation constraint enforcement

### For microplex v3: Copula/GNN Approaches

If you need:
- Perfect correlation preservation
- Transferability across populations
- State-of-the-art quality

See research papers:
- [Copula-Based Transferable Models](https://arxiv.org/abs/2302.09193)
- [GNN for Household Synthesis](https://www.cs.ox.ac.uk/teaching/studentprojects/923.html)
- [Hierarchical Mixture Models](https://www.sciencedirect.com/science/article/abs/pii/S0191261517308615)

**Implementation effort:** High (3-6 months)

## Practical Considerations

### Storage Format

For hierarchical data, use denormalized tables:

```
persons.parquet:
  person_id | household_id | tax_unit_id | age | earnings

households.parquet:
  household_id | income | size | region

tax_units.parquet:
  tax_unit_id | household_id | income | filing_status
```

Then join when needed:

```python
# For synthesis, flatten
training_data = (
    persons
    .merge(households, on="household_id")
    .merge(tax_units, on="tax_unit_id")
)

# For microsimulation, keep normalized
sim.load_persons(persons)
sim.load_households(households)
sim.calculate("tax_liability")
```

### Reweighting Level

Always reweight at the household level:

```python
from microplex import Reweighter

# CORRECT: Household-level reweighting
reweighter = Reweighter()
weighted = reweighter.fit_transform(
    households,  # One row per household
    targets={"state": {...}},
)

# WRONG: Person-level reweighting (violates hierarchy)
weighted = reweighter.fit_transform(
    persons,  # Persons in same household could get different weights
    targets={"state": {...}},
)
```

### Memory Efficiency

For large populations (100M+ persons):

1. **Don't duplicate household vars in memory**:
   ```python
   # Bad: 100M rows × 50 household vars = huge
   flat = persons.merge(households)

   # Good: Store separately, join on demand
   person_features = synth.generate(persons)
   full_data = person_features.merge(
       households[["household_id", "income"]],
       on="household_id"
   )
   ```

2. **Generate in batches**:
   ```python
   for batch in batches(demographics, size=1_000_000):
       synthetic_batch = synth.generate(batch)
       synthetic_batch.to_parquet(f"output_{i}.parquet")
   ```

## Code Sketch: Hybrid Approach

```python
# microplex/hierarchy.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def enforce_hierarchy(
    data: pd.DataFrame,
    household_id: str,
    shared_vars: List[str],
    relationship_constraints: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Enforce hierarchical consistency in flattened microdata.

    Ensures:
    - Household-level variables are identical for all members
    - Relationship constraints are satisfied (spouse ages, child ages, etc.)

    Args:
        data: Flattened microdata with person records
        household_id: Column identifying household membership
        shared_vars: Variables that must be identical within household
        relationship_constraints: Rules for relationships (optional)

    Returns:
        Corrected microdata

    Example:
        >>> synthetic = synth.generate(demographics)
        >>> consistent = enforce_hierarchy(
        ...     synthetic,
        ...     household_id="household_id",
        ...     shared_vars=["household_income", "region"],
        ...     relationship_constraints={
        ...         "spouse_age_diff": {"max": 20},
        ...     }
        ... )
    """
    result = []

    for hh_id, group in data.groupby(household_id):
        hh = group.copy()

        # Enforce shared variables (use mean/mode)
        for var in shared_vars:
            if var in hh.columns:
                if hh[var].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    # Numeric: use mean
                    hh[var] = hh[var].mean()
                else:
                    # Categorical: use mode
                    hh[var] = hh[var].mode()[0]

        # Apply relationship constraints
        if relationship_constraints:
            hh = _apply_constraints(hh, relationship_constraints)

        result.append(hh)

    return pd.concat(result, ignore_index=True)


def _apply_constraints(
    household: pd.DataFrame,
    constraints: Dict,
) -> pd.DataFrame:
    """Apply relationship constraints within household."""

    # Spouse age difference
    if "spouse_age_diff" in constraints and len(household) >= 2:
        max_diff = constraints["spouse_age_diff"]["max"]
        ages = household["age"].values[:2]

        if abs(ages[0] - ages[1]) > max_diff:
            # Adjust second person's age
            household.loc[household.index[1], "age"] = (
                ages[0] + np.random.randint(-max_diff//2, max_diff//2)
            )

    # Child age constraints
    if "child_age" in constraints and len(household) > 2:
        max_parent_diff = constraints["child_age"]["max_parent_diff"]
        parent_ages = household["age"].values[:2]
        max_parent_age = max(parent_ages)

        for i in range(2, len(household)):
            child_age = household.iloc[i]["age"]
            if child_age > max_parent_age - max_parent_diff:
                # Adjust child age to be reasonable
                household.loc[household.index[i], "age"] = max(
                    0,
                    max_parent_age - max_parent_diff - np.random.randint(0, 5)
                )

    return household


# Add to Synthesizer class
from microplex.synthesizer import Synthesizer

def generate_with_hierarchy(
    self,
    conditions: pd.DataFrame,
    household_id: str,
    shared_vars: List[str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic data with hierarchical consistency enforced.

    Convenience method that combines generate() + enforce_hierarchy().
    """
    synthetic = self.generate(conditions, seed=seed)
    return enforce_hierarchy(synthetic, household_id, shared_vars)

# Monkey-patch for backwards compatibility
Synthesizer.generate_with_hierarchy = generate_with_hierarchy
```

## Future Roadmap

### v1.0 (Current): Flattening
- ✅ Works with existing API
- ✅ Documented hierarchical patterns
- ✅ `enforce_hierarchy()` utility

### v1.5: Hybrid Post-Processing
- [ ] `generate_with_hierarchy()` method
- [ ] Automatic constraint detection
- [ ] Validation metrics for hierarchical quality

### v2.0: Native Hierarchical Synthesis
- [ ] `HierarchicalSynthesizer` class
- [ ] Top-down sampling
- [ ] Aggregation constraint enforcement

### v3.0: Advanced Methods
- [ ] Copula-based synthesis
- [ ] GNN household structure prediction
- [ ] Transferable models across populations

## References

### Hierarchical Synthesis Methods
- [Copula-Based Transferable Models for Synthetic Population Generation](https://arxiv.org/abs/2302.09193) (2024)
- [A Hierarchical Mixture Modeling Framework for Population Synthesis](https://www.sciencedirect.com/science/article/abs/pii/S0191261517308615)
- [Synthetic Population Generation by Combining Hierarchical Approach with Reweighting](https://www.researchgate.net/publication/295244935_Synthetic_Population_Generation_by_Combining_a_Hierarchical_Simulation-Based_Approach_with_Reweighting_by_Generalized_Raking)

### Graph Neural Networks
- [University of Oxford: Synthetic Population Generation Using GNNs](https://www.cs.ox.ac.uk/teaching/studentprojects/923.html)
- [A National Synthetic Populations Dataset for the United States](https://www.nature.com/articles/s41597-025-04380-7) (2025)

### Census/Government Approaches
- [SIPP Synthetic Beta Data Product](https://www.census.gov/programs-surveys/sipp/guidance/sipp-synthetic-beta-data-product.html) - Census Bureau approach
- [2024 SIPP Users' Guide](https://www2.census.gov/programs-surveys/sipp/tech-documentation/methodology/2024_SIPP_Users_Guide.pdf) - Hierarchical data structure
- [A Synthetic Population for Agent-Based Modelling in Canada](https://www.nature.com/articles/s41597-023-02030-4)

### Microsimulation Models
- [PolicyEngine US](https://github.com/PolicyEngine/policyengine-us) - Open-source tax-benefit microsimulation
- [Tax Microsimulation at The Budget Lab](https://budgetlab.yale.edu/research/tax-microsimulation-budget-lab) - Building tax units from CPS

### Review Articles
- [Generation of Synthetic Populations in Social Simulations: A Review](https://www.jasss.org/25/2/6.html) (2022)
- [Advances in Population Synthesis](https://link.springer.com/article/10.1007/s11116-011-9367-4) - Fitting household and person margins
