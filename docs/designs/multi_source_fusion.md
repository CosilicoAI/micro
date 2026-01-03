# Multi-Source Fusion Architecture

## Overview

Microplex synthesizes microdata by combining multiple survey sources, each with different strengths:

| Source | Type | Strengths | Limitations |
|--------|------|-----------|-------------|
| **CPS ASEC** | Cross-sectional | Broad income components, large sample | No panel dynamics |
| **SIPP** | Short panel (4yr) | Job-level income, monthly granularity | Limited history |
| **PSID** | Long panel (50yr) | Lifecycle trajectories, transitions | Smaller sample |
| **SCF** | Cross-sectional | Wealth, detailed assets | Income less detailed |

The goal: produce synthetic microdata with **maximum coverage** across all source distributions while learning **joint patterns** that no single source captures.

## The Fusion Strategy

### Core Insight

Each source S has:
- `shared_vars`: Variables present in all sources (age, total_income)
- `source_vars`: Variables unique to S (SIPP: job1_income; CPS: dividend_income)

We want synthetic data with **all variables** that:
1. Matches each source's marginal distributions
2. Captures joint relationships from the richest source
3. Has realistic dynamics from the longest panel

### The Stacked Imputation Approach

```
For each source S in {CPS, SIPP, PSID, ...}:
    1. Take S as "base" (preserves S's observed patterns)
    2. Impute missing vars onto S from other sources:
       - Train imputer: P(missing_vars | shared_vars) using richest source
       - Apply to S: fills in variables S doesn't have
    3. Result: S_complete with all variables

Stack all S_complete datasets → training data for unified model

Train unified synthesizer on stacked data → synthetic microdata
```

### Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MULTI-SOURCE FUSION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                         │
│  │   CPS   │    │  SIPP   │    │  PSID   │   Raw Sources           │
│  │ (cross) │    │ (4yr)   │    │ (50yr)  │                         │
│  └────┬────┘    └────┬────┘    └────┬────┘                         │
│       │              │              │                               │
│       ▼              ▼              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              IMPUTATION LAYER                            │       │
│  │                                                          │       │
│  │  For each source S:                                      │       │
│  │    - Use PSID/SIPP (richest) to train imputer           │       │
│  │    - Impute missing vars onto S                          │       │
│  │    - S now has all variables                             │       │
│  └─────────────────────────────────────────────────────────┘       │
│       │              │              │                               │
│       ▼              ▼              ▼                               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                         │
│  │CPS_full │    │SIPP_full│    │PSID_full│   Complete Sources      │
│  │ + job1  │    │ + div   │    │ + div   │   (all vars)            │
│  │ + job2  │    │ + int   │    │ + int   │                         │
│  └────┬────┘    └────┬────┘    └────┬────┘                         │
│       │              │              │                               │
│       └──────────────┼──────────────┘                               │
│                      ▼                                              │
│            ┌─────────────────┐                                      │
│            │  STACKED DATA   │  All sources combined               │
│            └────────┬────────┘                                      │
│                     │                                               │
│                     ▼                                               │
│            ┌─────────────────┐                                      │
│            │ UNIFIED MODEL   │  Learns joint distribution          │
│            │ (ZI-QDNN/MAF)   │  P(all_vars)                        │
│            └────────┬────────┘                                      │
│                     │                                               │
│                     ▼                                               │
│            ┌─────────────────┐                                      │
│            │   SYNTHETIC     │  Has all vars + dynamics            │
│            │   MICRODATA     │                                      │
│            └─────────────────┘                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Coverage Evaluation

We evaluate synthetic data against **each source's holdout** separately:

```python
coverage = fusion.evaluate_coverage(
    synthetic,
    holdouts={
        'cps': cps_holdout,      # How well do we cover CPS patterns?
        'sipp': sipp_holdout,    # How well do we cover SIPP patterns?
        'psid': psid_holdout,    # How well do we cover PSID patterns?
    }
)
```

A successful fusion improves coverage across all sources vs single-source baselines.

## Adding Panel Evolution (PSID Integration)

PSID's unique value: 50 years of **actual state transitions**.

Instead of hardcoded transition rates, we train `PanelEvolutionModel`:

```python
from microplex.models import PanelEvolutionModel
from microplex.data_sources import load_psid_panel

# Load PSID trajectories
psid = load_psid_panel(data_dir="./psid_data")

# Train evolution model on real transitions
evolution = PanelEvolutionModel(
    state_vars=["is_married", "income", "is_disabled"],
    condition_vars=["age", "is_male", "education"],
    lags=[1, 2, 3],  # Look back 3 periods
    history_features={"is_married": ["duration", "ever"]},
)
evolution.fit(psid.persons, epochs=100)

# Use in synthesis pipeline
initial_state = cross_sectional_synthesizer.generate(n=10000)  # From CPS
trajectories = evolution.simulate_trajectory(initial_state, n_steps=20)
```

### The Unified Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                    COMPLETE SYNTHESIS PIPELINE                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. CROSS-SECTIONAL SYNTHESIS (CPS + SIPP + PSID fusion)          │
│     ─────────────────────────────────────────────────────          │
│     Produces: initial population with all variables                │
│                                                                    │
│  2. PANEL EVOLUTION (trained on PSID trajectories)                 │
│     ─────────────────────────────────────────────────────          │
│     state[t+1] ~ state[t] + state[t-1] + ... + covariates         │
│     - Non-Markov (looks back multiple periods)                     │
│     - History features (marriage duration, ever divorced)          │
│     - Joint evolution (income + marriage correlated)               │
│                                                                    │
│  3. OUTPUT                                                         │
│     ─────                                                          │
│     Synthetic panel data with:                                     │
│     - All income components (from CPS+SIPP fusion)                 │
│     - Job-level detail (from SIPP)                                 │
│     - Realistic lifecycle dynamics (from PSID)                     │
│     - Correlated transitions (learned, not hardcoded)              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Variable Coverage Matrix

| Variable | CPS | SIPP | PSID | Impute From |
|----------|-----|------|------|-------------|
| age | ✓ | ✓ | ✓ | - |
| is_male | ✓ | ✓ | ✓ | - |
| total_income | ✓ | ✓ | ✓ | - |
| wage_income | ✓ | ✓ | ✓ | - |
| job1_income | - | ✓ | - | SIPP |
| job2_income | - | ✓ | - | SIPP |
| dividend_income | ✓ | - | ✓ | CPS/PSID |
| interest_income | ✓ | - | ✓ | CPS/PSID |
| is_married | ✓ | ✓ | ✓ | - |
| marriage_duration | - | - | ✓ | PSID |
| ever_divorced | - | - | ✓ | PSID |
| total_wealth | - | - | ✓ | PSID |

## Implementation Status

### Done ✓
- [x] `MultiSourceFusion` class for stacked imputation
- [x] CPS data source (`load_cps_asec`)
- [x] PSID data source (`load_psid_panel`)
- [x] Transition rate extraction (`extract_transition_rates`)
- [x] `PanelEvolutionModel` (unified autoregressive model)
- [x] Coverage evaluation per source

### TODO
- [ ] Add PSID to fusion experiments
- [ ] Train evolution model on real PSID data
- [ ] Validate against PSID holdout trajectories
- [ ] Add SCF for wealth variables
- [ ] Document variable harmonization

## API Usage

```python
from microplex.fusion import MultiSourceFusion
from microplex.data_sources import load_cps_asec, load_psid_panel
from microplex.models import PanelEvolutionModel

# Load sources
cps = load_cps_asec(year=2023)
sipp = load_sipp_panel(data_dir="./sipp")
psid = load_psid_panel(data_dir="./psid")

# Define variable sets
shared_vars = ["age", "total_income", "is_married"]
all_vars = shared_vars + ["job1_income", "dividend_income", "marriage_duration"]

# Create fusion pipeline
fusion = MultiSourceFusion(
    shared_vars=shared_vars,
    all_vars=all_vars,
)

# Add sources
fusion.add_source("cps", cps.persons, source_vars=["age", "total_income", "dividend_income"])
fusion.add_source("sipp", sipp, source_vars=["age", "total_income", "job1_income"])
fusion.add_source("psid", psid.persons, source_vars=all_vars)

# Fit
fusion.fit(epochs=100)

# Generate with evolution
initial = fusion.generate(n_per_source=10000)

evolution = PanelEvolutionModel(
    state_vars=["is_married", "total_income"],
    condition_vars=["age", "is_male"],
    lags=[1, 2],
)
evolution.fit(psid.persons)

trajectories = evolution.simulate_trajectory(initial, n_steps=20)

# Evaluate coverage on each source
coverage = fusion.evaluate_coverage(
    trajectories,
    holdouts={"cps": cps_holdout, "sipp": sipp_holdout, "psid": psid_holdout}
)
```

## References

- `src/microplex/fusion/multi_source_fusion.py` - Fusion pipeline
- `src/microplex/data_sources/psid.py` - PSID loader
- `src/microplex/models/panel_evolution.py` - Autoregressive evolution
- `experiments/real_fusion_experiment.py` - SIPP+CPS experiment
