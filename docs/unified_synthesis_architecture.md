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

For Social Security, we need lifetime trajectories. Our target is parity with DYNASIM.

### DYNASIM Comparison

**DYNASIM** (Urban Institute) is the gold standard for Social Security microsimulation:
- Uses SIPP as base population (restricted versions)
- Year-by-year simulation with transition probabilities
- Statistical matching to administrative earnings
- 50+ year projection horizon

**Our approach (microplex)** differs in key ways:

| Aspect | DYNASIM | Microplex |
|--------|---------|-----------|
| Base data | SIPP (restricted) | CPS + PSID (public) |
| Generation | Sequential year-by-year | All-at-once |
| Transitions | Hazard models | Learned from joint distribution |
| Earnings | Parametric projection | Conditional flow |
| Uncertainty | Single run | Multiple draws |

**Parity requirements:**
1. ✅ Earnings trajectories (18-70) - TrajectoryModel
2. 🔲 Marriage/divorce transitions - Need hazard models
3. 🔲 Fertility simulation - Need birth hazards
4. 🔲 Disability onset - Need disability hazards
5. 🔲 Mortality - Use SSA life tables
6. 🔲 Forward projection beyond training data

### Earnings Trajectories

**Key architectural decision**: Generate full trajectory ALL AT ONCE (not sequentially).

Why all-at-once vs sequential:
- Sequential (AR): P(earnings_t | earnings_{t-1}, ...) compounds errors over time
- All-at-once: P(earnings_18:70 | demographics) preserves full correlations

The model learns latent "trajectory types" that capture lifecycle shapes:
- **Type A**: Steady growth (professional career)
- **Type B**: Peak-then-decline (manual labor, physical jobs)
- **Type C**: Volatile (self-employment, gig economy)
- **Type D**: Interrupted (disability, caregiving, unemployment spells)

Implementation:
- Use microplex's ConditionalMAF with 35-50 output dimensions (one per year)
- Condition on: education, gender, birth cohort, initial earnings
- Training data: PSID (50+ year histories, ideal) or SIPP (4-year panels, public)

### Demographic Transitions

Event times as functions of latent + current state:
- `P(marriage_age | z, current_age, never_married)`
- `P(divorce | z, marriage_duration)`
- `P(disability_onset | z, age, occupation)`
- `P(death_age | z, health_trajectory)`

### Hierarchical + Temporal Synthesis

The full synthesis pipeline is:
1. **Households** (HierarchicalSynthesizer): size, geography, housing costs
2. **Persons within households** (HierarchicalSynthesizer): demographics, initial income
3. **Trajectories for each person** (TrajectoryModel): 35-year earnings histories

This produces a complete longitudinal synthetic population suitable for
Social Security and lifetime tax modeling.

## Advantages Over Current Approach

1. **Principled uncertainty**: Latent model captures full distribution
2. **Selection bias correction**: Explicit in observation models
3. **1:1 scale**: Sample z directly, no record copying
4. **Panel coherence**: Single z governs full trajectory
5. **Extensibility**: Add new surveys by adding observation models

## Experimental Findings

### The Coverage Problem

For 1:1 population synthesis (generating 330M records from surveys totaling ~4M), the key metric is **coverage**: how well the synthetic population fills the attribute space.

```
Coverage = average distance from each holdout person to nearest synthetic record
```

Lower is better. Unlike MMD (distribution matching), coverage measures whether we can generate the full diversity of the population.

### Curse of Dimensionality

With 50 dimensions (typical for tax/benefit modeling), we discovered:

| Sample Ratio | Coverage Distance |
|--------------|-------------------|
| 1:1          | 2.35              |
| 5:1          | 0.05              |
| 10:1         | 0.00              |

This explains why PolicyEngine's ~300k weighted records cannot adequately cover a 330M population in 50D space. We need either:
1. **More samples** (1:1 synthesis at full scale)
2. **Better generalization** (flow-based models that extrapolate)

### Method Comparison on Coverage

Testing at realistic sparse coverage (1% training data):

| Method           | Coverage | vs Oracle |
|------------------|----------|-----------|
| Oracle           | 0.17     | 1.0x      |
| **Microplex**    | **0.42** | **2.4x**  |
| Resample         | 0.48     | 2.8x      |
| NND.hotdeck      | 0.50     | 2.9x      |
| CT-GAN           | 0.57     | 3.3x      |
| QRF+ZI           | 0.62     | 3.6x      |
| TVAE             | 1.68     | 9.6x      |
| Gaussian Copula  | 4.07     | 23.3x     |

**Key findings:**
1. **Microplex wins on coverage** - Flow-based models extrapolate better
2. **QRF is WORSE than resampling** - Quantile regression regresses toward the mean, producing more concentrated outputs than training data
3. **MMD tells opposite story** - CT-GAN wins on MMD but loses on coverage. For 1:1 synthesis, coverage matters more.

### Multi-Survey Fusion Challenge

Tested naive approach: stack all surveys, fill NaN with 0, train single model.

| Approach              | Coverage |
|-----------------------|----------|
| Single survey         | 0.58     |
| Multi-survey (NaN→0)  | 0.82     |

**Naive stacking is WORSE.** NaN→0 corrupts the joint distribution. Proper approaches:
1. Masked loss (only compute loss on observed values)
2. Sequential fusion (current PE approach) - preserves structure better
3. Imputation during training with uncertainty

### Real CPS Validation

Tested on actual CPS 2024 data (142,125 records, 40 columns) at 1% training:

| Method    | Coverage |
|-----------|----------|
| Microplex | 0.54     |
| Resample  | 0.59     |

**~8% improvement** on real data confirms simulation findings.

## Product: The Microplex

The output of this synthesis process is called **"the Microplex"** - a 1:1 synthetic population:

- **Initial target**: 330M US persons with full tax/benefit attributes
- **Future scope**: Everyone on Earth (~8B records)
- **Uncertainty**: Multiple draws for uncertainty quantification

Unlike weighted microdata (300k records × weights = 330M), the Microplex is:
- **Explicit**: Every person is a concrete record
- **Diverse**: Covers the full attribute space
- **Scalable**: Can generate arbitrary N

## Open Questions

1. **Dimensionality of z**: How big does latent need to be for 100+ variables?
2. **Identifiability**: Can we learn z with so much missing data?
3. **Computation**: Training on 330M records?
4. **Validation**: How do we know synthetic trajectories are realistic?
5. **Entity crosswalks**: Is hierarchical latent tractable?
6. **Multi-survey training**: How to handle missing data properly (masked loss, impute during training)?

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
