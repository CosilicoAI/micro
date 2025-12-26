# CPS ASEC Microdata

This directory contains processed CPS ASEC (Current Population Survey Annual Social and Economic Supplement) microdata for use with the microplex HierarchicalSynthesizer.

## Data Files

| File | Description | Records |
|------|-------------|---------|
| `cps_asec_households.parquet` | Household-level data | ~60,000 |
| `cps_asec_persons.parquet` | Person-level data | ~150,000 |

## Data Source

The data comes from the U.S. Census Bureau's Current Population Survey Annual Social and Economic Supplement (CPS ASEC), also known as the March Supplement.

**Official Sources:**
- Census Bureau CPS ASEC page: https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html
- 2024 ASEC data: https://www.census.gov/data/datasets/2024/demo/cps/cps-asec-2024.html
- FTP directory: https://www2.census.gov/programs-surveys/cps/datasets/2024/march/

**Alternative Sources:**
- IPUMS CPS: https://cps.ipums.org/ (requires free account, provides cleaner extracts)

## Downloading the Data

Run the download script to fetch and process CPS ASEC data:

```bash
# Download and process real CPS ASEC 2024 data
python scripts/download_cps_asec.py

# Use a specific year
python scripts/download_cps_asec.py --year 2023

# Generate sample data for testing (no download required)
python scripts/download_cps_asec.py --sample
```

## Variable Definitions

### Household Variables

| Variable | Description | Values |
|----------|-------------|--------|
| `household_id` | Unique household identifier | Integer |
| `n_persons` | Number of persons in household | 1-20 |
| `n_adults` | Number of adults (age >= 18) | 1-20 |
| `n_children` | Number of children (age < 18) | 0-19 |
| `state_fips` | State FIPS code | 1-56 |
| `tenure` | Housing tenure | 1=Owned, 2=Rented, 3=Occupied without rent |
| `hh_weight` | Household survey weight | Float |

### Person Variables

| Variable | Description | Values |
|----------|-------------|--------|
| `person_id` | Unique person identifier | Integer |
| `household_id` | Household this person belongs to | Integer |
| `age` | Age in years | 0-120 |
| `sex` | Sex | 1=Male, 2=Female |
| `income` | Total personal income (annual, USD) | 0+ |
| `employment_status` | Employment status | 0=NILF, 1=Employed, 2=Unemployed |
| `education` | Educational attainment | 1=Less than HS, 2=HS, 3=Some college, 4=Bachelor+ |
| `relationship_to_head` | Relationship to household head | 1=Head, 2=Spouse, 3=Other relative, 4=Child |

## Processing Steps

1. **Download**: Raw CPS ASEC CSV files downloaded from Census Bureau FTP
2. **Parse**: Extract person, household, and family record files
3. **Map columns**: Rename raw CPS variables to standardized schema
4. **Aggregate**: Create household-level summaries from person data
5. **Clean**: Handle missing values, ensure proper types
6. **Save**: Export to parquet format for efficient storage/loading

## Usage with microplex

```python
from microplex import HierarchicalSynthesizer, load_cps_for_synthesis

# Load prepared data
households, persons = load_cps_for_synthesis()

# Train hierarchical synthesizer
synth = HierarchicalSynthesizer()
synth.fit(households, persons, epochs=100)

# Generate synthetic households
syn_hh, syn_persons = synth.generate(n_households=10000)
```

## Privacy and Confidentiality

In accordance with Title 13, U.S. Code, CPS ASEC public use microdata files do not contain personally identifiable information. Some data have been "perturbed" by the Census Bureau to protect respondent confidentiality.

## Citation

When using this data, please cite:

> U.S. Census Bureau, Current Population Survey, Annual Social and Economic Supplement (CPS ASEC), [YEAR].

## Data Dictionary

For complete variable definitions and coding details, see:
- CPS ASEC Data Dictionary: https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf
- IPUMS CPS Variable Documentation: https://cps.ipums.org/cps-action/variables/group
