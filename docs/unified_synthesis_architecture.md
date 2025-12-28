# Unified Population Synthesis Architecture

## The Core Problem

We want to reconstruct the full US population (330M people) from partial observations:

| Source | N | Entity | Variables Observed |
|--------|---|--------|-------------------|
| CPS | 100k persons | Person, HH | Demographics, employment, income (some) |
| ACS | 3.5M persons | Person, HH | Demographics, housing, geography |
| IRS PUF | 180k tax units | Tax Unit | All income sources, deductions, credits |
| SIPP | 100k persons | Person, HH | Program participation, detailed income |
| SCF | 6k families | Primary Economic Unit | Wealth, assets, detailed finances |
| PSID | 10k families | Family | Longitudinal (35+ years) |
| SSA Admin | 160M workers | Person | Lifetime earnings, benefits |

Each source has:
1. Different sample size
2. Different entity definition (person vs HH vs tax unit vs family)
3. Different variables observed
4. Different selection mechanism (CPS: oversamples unemployed; IRS: only filers)
5. Different time structure (cross-sectional vs panel)

## Current Approach (PolicyEngine)

```
ACS (demographics)
  ↓ QRF imputation
CPS (add employment, some income)
  ↓ QRF imputation
IRS PUF (add detailed income)
  ↓ Gradient descent calibration
Synthetic population (2,813 calibration targets)
```

Problems:
- Sequential chaining loses joint structure
- No explicit selection bias correction
- Can't generate at 1:1 scale (stuck at ~300k weighted records)
- No panel/longitudinal component

## Unified Latent Space Approach

### Key Insight

Every person in the US can be characterized by a **latent vector** `z ∈ R^d` that encodes:
- Demographics (age, gender, education, location)
- Economic characteristics (income, wealth, employment)
- Lifecycle trajectory (earnings path, marriage timing, mortality)

Different surveys are **projections** of this latent space:
- `CPS(z) → observed_cps_vars` (with CPS selection mechanism)
- `IRS(z) → observed_irs_vars` (with IRS selection: filers only)
- `PSID(z) → observed_panel_vars` (with PSID selection)

### Architecture

```
Layer 1: Latent Population Model
  - Learn P(z) from all survey data jointly
  - z is high-dimensional latent capturing all characteristics
  - VAE or flow-based model

Layer 2: Survey Observation Models
  - P(CPS_vars | z, selection_CPS)
  - P(IRS_vars | z, selection_IRS)
  - P(PSID_vars | z, selection_PSID)
  - Each handles its own selection mechanism

Layer 3: Calibration
  - Known marginals (from ACS, admin data)
  - Soft constraints during training or post-hoc reweighting

Layer 4: Panel Extension
  - z encodes full trajectory, not just current state
  - P(earnings_t | z) for any t ∈ [birth_year, 2024]
  - Demographic transitions as hazard functions of z
```

### Training Procedure

1. **Pre-training**: Train separate autoencoders for each survey's observed variables
2. **Alignment**: Learn cross-survey mappings using shared variables as anchors
3. **Joint training**: Train unified latent model with all surveys as projections
4. **Calibration**: Add known marginal constraints

### Generation

To generate 330M synthetic people:
1. Sample `z_1, ..., z_330M ~ P(z)`
2. Decode to full attribute space: `x_i = decode(z_i)`
3. For panel attributes: `trajectory_i = panel_decode(z_i)`
4. Optional: Calibration weights for exact marginals

## Entity Harmonization

The hardest part: different surveys use different entity definitions.

### Approach 1: Person-centric with Composition Models

1. Latent space is person-level: `z_person`
2. Household composition: `P(HH_structure | z_person)`
3. Tax unit formation: `P(TU_structure | HH_structure, filing_rules)`
4. Family definition: deterministic from HH + relationships

### Approach 2: Hierarchical Latent Space

```
z_household → { z_person_1, ..., z_person_n }
            → z_tax_unit_1, z_tax_unit_2
            → shared_HH_characteristics
```

Each survey observes projections at different levels of this hierarchy.

## Panel/Longitudinal Component

For Social Security, we need lifetime trajectories:

### Earnings Trajectories

Current approach (PSID-based QRF):
- Condition on: initial age, education, gender, initial earnings
- Predict: earnings at each age

Unified approach:
- Latent `z` encodes "earnings type" (trajectory shape)
- Different types: steady growth, peak-then-decline, volatile, etc.
- Learn from PSID but generate many more variants

### Demographic Transitions

Event times as functions of latent + current state:
- `P(marriage_age | z, current_age, never_married)`
- `P(divorce | z, marriage_duration)`
- `P(disability_onset | z, age, occupation)`
- `P(death_age | z, health_trajectory)`

## Advantages Over Current Approach

1. **Principled uncertainty**: Latent model captures full distribution
2. **Selection bias correction**: Explicit in observation models
3. **1:1 scale**: Sample z directly, no record copying
4. **Panel coherence**: Single z governs full trajectory
5. **Extensibility**: Add new surveys by adding observation models

## Open Questions

1. **Dimensionality of z**: How big does latent need to be for 100+ variables?
2. **Identifiability**: Can we learn z with so much missing data?
3. **Computation**: Training on 330M records?
4. **Validation**: How do we know synthetic trajectories are realistic?
5. **Entity crosswalks**: Is hierarchical latent tractable?

## Next Steps

1. **Minimal viable experiment**:
   - True population: 100k people, 20 variables, simple trajectories
   - 4 simulated surveys with different selection/coverage
   - Compare unified vs sequential on reconstruction quality

2. **Scale test**:
   - 1M people, 50 variables
   - Test whether approach works at scale

3. **Panel prototype**:
   - Add simple earnings trajectory (10 years)
   - Test longitudinal coherence

4. **Real data**:
   - Train on actual CPS/IRS/SIPP
   - Compare to PolicyEngine baseline
