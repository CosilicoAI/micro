# PolicyEngine Household/Person/Tax Unit Structure

## Overview

This document summarizes PolicyEngine's approach to handling household, person, and tax unit structures in their microdata enhancement and microsimulation system.

## Entity Hierarchy

PolicyEngine defines five main group entities that nest within each other, plus the individual person entity:

### Entity Definitions

From [`policyengine_us/entities.py`](https://github.com/PolicyEngine/policyengine-us/blob/master/policyengine_us/entities.py):

1. **Person** (Individual entity)
   - Key: `"person"`, Plural: `"people"`
   - The atomic unit - all other entities are groups of persons
   - Marked with `is_person=True`

2. **Household** (Top-level group)
   - Key: `"household"`, Plural: `"households"`
   - Contains all members living together
   - Top of the nesting hierarchy

3. **SPMUnit** (Supplemental Poverty Measure Unit)
   - Key: `"spm_unit"`, Plural: `"spm_units"`
   - Nested within households
   - Used for poverty calculations following Census SPM methodology

4. **Family**
   - Key: `"family"`, Plural: `"families"`
   - Nested within SPM units and households
   - Represents Census family definition

5. **TaxUnit**
   - Key: `"tax_unit"`, Plural: `"tax_units"`
   - Nested within SPM units, families, and households
   - Represents IRS tax filing units

6. **MaritalUnit**
   - Key: `"marital_unit"`, Plural: `"marital_units"`
   - An unmarried person OR a married co-habiting couple
   - Nested within households

### Nesting Hierarchy

```
Household
├── SPMUnit(s)
│   ├── Family(ies)
│   │   └── TaxUnit(s)
│   └── TaxUnit(s)
└── MaritalUnit(s)
```

## Data Storage Format

### HDF5 Arrays (Primary Storage)

PolicyEngine stores microdata in **HDF5 format** (`.h5` files) using arrays. From the [PolicyEngine Core Data documentation](https://policyengine.github.io/policyengine-core/python_api/data.html):

**Three storage formats:**

1. **`Dataset.ARRAYS`** - Collection of arrays (most common)
2. **`Dataset.TIME_PERIOD_ARRAYS`** - Arrays with one per time period
3. **`Dataset.TABLES`** - Collection of DataFrames

**Loading example:**
```python
from policyengine_us import Microsimulation

# Load dataset for a year
dataset = Dataset.load(2022)

# Access like a dictionary
person_weights = dataset["person_weight"]
employment_income = dataset["employment_income"]
```

### Long Format Structure

The CPS microdata follows a **long format** (also called person-level format):
- **One row per person**
- Household-level variables are **repeated** for all members of the same household
- Person-level variables vary by individual

From [Census CPS guidance](https://www.census.gov/programs-surveys/cps/data/datasets.html):
> "Each record is a person, with all characteristics numerically coded. Persons are organized into households, making it possible to study characteristics of people in the context of their families."

### Key ID and Relationship Arrays

Based on PolicyEngine's [population.py](https://github.com/PolicyEngine/policyengine-core/blob/master/policyengine_core/populations/population.py), the system maintains:

1. **Member IDs** - List of unique identifiers for each entity instance
2. **Members Position** - Each person's position within their group entity
3. **Members Entity ID** - Maps individuals to their parent group entity
4. **Roles** - Specific positions within groups (e.g., "head", "spouse", "child")

**Example arrays in dataset:**
- `person_id` - Unique person identifier (0, 1, 2, ...)
- `household_id` - Household identifier (repeated for all household members)
- `tax_unit_id` - Tax unit identifier
- `person_household_role` - Role within household
- `person_tax_unit_role` - Role within tax unit (e.g., filer, spouse, dependent)

## Population Architecture

From [`policyengine_core/populations/population.py`](https://github.com/PolicyEngine/policyengine-core):

### Population Object

Each entity type has a `Population` object that:
- Maintains an `ids` list of member identifiers
- Stores `Holder` objects for each variable (using NumPy arrays)
- Tracks member count
- Provides methods for array operations matching population size

### Member-Group Relationships

Relationships are managed through:
- **Role assignments** - Each person has roles in their group entities
- **Position tracking** - Numeric position within group
- **Entity ID mapping** - Links to parent group
- **Projection methods** - Transfer values between related members (e.g., `value_from_partner()`)

### Array Operations

All data is stored as NumPy arrays:
```python
# Create arrays matching population size
empty = population.empty_array()
filled = population.filled_array(value=0)

# Get array index from ID
index = population.get_index(person_id)

# Rank members within entities
ranks = population.get_rank(variable_name, condition)
```

## Microdata Enhancement Process

From [PolicyEngine US Data documentation](https://policyengine.github.io/policyengine-us-data/):

### 1. Source Data

**CPS (Current Population Survey):**
- Provides demographic detail
- Geographic granularity
- Household structure and relationships

**IRS Public Use File (PUF):**
- Detailed tax reporting data
- Administrative source

### 2. Imputation Phase

**Method:** Quantile regression forests
- Imputes 67 tax variables from PUF onto CPS records
- **Maintains household composition and member relationships**
- Preserves distributional characteristics
- Each person in CPS gets imputed tax variables

### 3. Reweighting Phase

**Method:** Dropout-regularized gradient descent
- Calibrates to **2,813 administrative targets**:
  - IRS Statistics of Income
  - Census population projections
  - CBO benefit estimates
  - Treasury expenditure data
  - Joint Committee on Taxation estimates
  - Healthcare spending patterns
  - Other benefit program costs

### 4. Output Format

**Maintains CPS structure:**
- Long format (one row per person)
- Household-level variables repeated
- Person-level variables unique
- Imputed tax variables added
- Calibrated weights

**Key insight:** The household/person structure from CPS is **preserved** throughout imputation and reweighting.

## Simulation API

From [PolicyEngine Core simulation documentation](https://policyengine.github.io/policyengine-core/usage/simulation.html):

### Defining Situations

```python
from policyengine_us import Simulation

situation = {
    "people": {
        "person_1": {
            "age": {2023: 30},
            "employment_income": {2023: 50_000},
        },
        "person_2": {
            "age": {2023: 28},
            "employment_income": {2023: 45_000},
        },
        "child_1": {
            "age": {2023: 5},
        },
    },
    "households": {
        "household_1": {
            "members": ["person_1", "person_2", "child_1"],
            "state_code": {2023: "CA"},
        },
    },
    "tax_units": {
        "tax_unit_1": {
            "members": ["person_1", "person_2", "child_1"],
            "filing_status": {2023: "JOINT"},
        },
    },
    "spm_units": {
        "spm_unit_1": {
            "members": ["person_1", "person_2", "child_1"],
        },
    },
}

sim = Simulation(situation=situation)
income_tax = sim.calculate("income_tax", 2023)
```

### Key Principles

1. **Nested dictionaries** - Organized by entity plural, then ID, then variables
2. **Time periods** - Variables can have values for specific periods
3. **Explicit membership** - Must declare which persons belong to which groups
4. **Role relationships** - Framework handles data flow based on declared relationships

## Weights

From [CPS weight documentation](https://cps.ipums.org/cps/sample_weights.shtml):

### Person Weights
- `person_weight` (monthly CPS) or `ASECWT` (CPS ASEC)
- Use for person-level analyses
- Scales to represent full population

### Household Weights
- `household_weight` or `ASECWTH` (CPS ASEC)
- Use for household-level analyses
- Different from person weights

### Family Weights
- March Supplement family weight
- For family-level ASEC analyses

**Critical:** PolicyEngine maintains these weights through the imputation and reweighting process, with final calibrated weights matching administrative targets.

## Unit of Analysis

From examining the architecture:

1. **Storage unit:** Person (long format, one row per person)
2. **Imputation unit:** Person (tax variables imputed to each person)
3. **Reweighting unit:** Person (weights calibrated per person)
4. **Calculation unit:** Depends on variable
   - Person-level: employment_income, age, etc.
   - Household-level: aggregated via relationships
   - Tax unit-level: aggregated via tax_unit membership

## Household-Level vs Person-Level Variables

### How They're Handled

**Person-level variables:**
- Stored once per person
- Examples: `age`, `employment_income`, `is_disabled`

**Household-level variables:**
- In CPS: Repeated for all household members
- In PolicyEngine dataset: Also repeated (maintains CPS structure)
- Examples: `state_code`, `county_fips`, `household_vehicles`

**Aggregated variables:**
- Calculated on-demand by summing/aggregating person-level values
- Examples: `household_income = sum of employment_income across household members`

### Calculation Flow

When PolicyEngine calculates a household-level variable:
1. Framework identifies household membership via `members` arrays
2. Gathers person-level inputs for each member
3. Applies aggregation formula
4. Returns result (can be broadcast back to person-level if needed)

## Key Differences from Wide Format

**Wide format** (not used by PolicyEngine):
- One row per household
- Columns like `person_1_age`, `person_2_age`, etc.
- Fixed maximum household size
- Difficult to iterate over persons

**Long format** (PolicyEngine's approach):
- One row per person
- Single `age` column
- Flexible household sizes
- Easy person-level operations
- Household operations via groupby/aggregation

## Within-Household Relationships

From [IPUMS CPS documentation](https://cps.ipums.org/cps/about.shtml):

### Pointer Variables
IPUMS provides "pointer" variables indicating location of:
- Mother
- Father
- Spouse

### PolicyEngine Roles
In PolicyEngine's system:
- **Role types:** "head", "spouse", "child", "dependent", etc.
- **Role checks:** `population.has_role(role_name)`
- **Projections:** `value_from_partner()` transfers values between spouses

### Example Role Usage

```python
# In a tax calculation
filing_status = tax_unit("filing_status", period)
is_joint = filing_status == "JOINT"

# Project spouse income to filer
spouse_income = person.value_from_partner("employment_income", period, role="spouse")

# Calculate combined income for joint filers
combined_income = where(
    is_joint,
    person("employment_income", period) + spouse_income,
    person("employment_income", period)
)
```

## Summary: Answers to Key Questions

### 1. Wide vs Long Format?

**Long format** - One row per person with household IDs linking members together.

### 2. How are within-household relationships captured?

- **Pointer arrays** - Link persons to household/tax unit/SPM unit IDs
- **Role assignments** - Each person has roles (head, spouse, child, etc.)
- **Position tracking** - Numeric position within group entities
- **IPUMS pointers** - Mother, father, spouse location pointers in source data

### 3. How does CPS/ACS data come in and get transformed?

- **Input:** CPS ASEC in long format (one row per person)
- **Enhancement:** Impute 67 tax variables using quantile regression forests
- **Preservation:** Maintain household composition and relationships throughout
- **Reweighting:** Calibrate to 2,813 administrative targets
- **Output:** Same structure as input (long format) with added variables and updated weights

### 4. What's the unit of analysis for imputation/enhancement?

**Person** - All imputation and reweighting operates at the person level, while maintaining household relationships.

### 5. How are household-level variables handled vs person-level?

- **Person-level:** Stored once per person (e.g., age, income)
- **Household-level:** Repeated for all household members (e.g., state_code)
- **Aggregated:** Calculated on-demand by framework using membership arrays
- **Access:** Both types accessible via same array interface

## References

- [PolicyEngine US Repository](https://github.com/PolicyEngine/policyengine-us)
- [PolicyEngine US Data Repository](https://github.com/PolicyEngine/policyengine-us-data)
- [PolicyEngine US Data Documentation](https://policyengine.github.io/policyengine-us-data/)
- [PolicyEngine Core Documentation](https://policyengine.github.io/policyengine-core/intro.html)
- [PolicyEngine Core Simulation Guide](https://policyengine.github.io/policyengine-core/usage/simulation.html)
- [PolicyEngine Core Data API](https://policyengine.github.io/policyengine-core/python_api/data.html)
- [IPUMS CPS Documentation](https://cps.ipums.org/cps/about.shtml)
- [Census CPS ASEC Guidance](https://www.census.gov/programs-surveys/cps/data/datasets.html)
- [PolicyEngine entities.py](https://github.com/PolicyEngine/policyengine-us/blob/master/policyengine_us/entities.py)
- [PolicyEngine Core population.py](https://github.com/PolicyEngine/policyengine-core/blob/master/policyengine_core/populations/population.py)
