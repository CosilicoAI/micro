# PolicyEngine Parity Status

Tracking microplex calibration target coverage vs PolicyEngine-US.

## Summary

| Metric | Microplex | PolicyEngine | Notes |
|--------|-----------|--------------|-------|
| **Total Targets** | 7,291 | 1,661 | Microplex has more geographic detail |
| **Geographic** | 7,254 | 52 | CDs, SLDs vs just states |
| **Income/Benefits** | 22 | 30 | Similar coverage |
| **Health/Tax** | 15 | 1,579 | PE has state-level Medicaid/CHIP |

## ✅ Working Calibration (2024-12-29)

Successfully calibrating CPS to 65 targets:
- **51 state populations**: 0% error
- **14 income/benefit targets**: 0-4% error on most

```
Target                          Computed     Target     Error
─────────────────────────────────────────────────────────────
State populations (51)          331.4M       331.4M     0.00%
rental_income                   $46.0B       $46.0B     0.00%
self_employment_income          $418.9B      $436.4B    4.01%
unemployment_compensation       $200.3B      $208.0B    3.72%
taxable_pension_income          $827.6B      $827.6B    0.00%
alimony_income                  $8.5B        $8.5B      0.00%
snap                            $103.1B      $103.1B    0.00%
ssi                             $78.5B       $78.5B     0.00%
eitc                            $72.7B       $72.7B     0.00%
```

### Known Gaps (CPS Data Limitations)

| Target | Error | Reason |
|--------|-------|--------|
| capital_gains | 96% | CPS has limited capital gains data |
| social_security | 35% | Underreported in CPS |
| employment_income | 21% | Underreported in CPS |
| dividend_income | 73% | Underreported in CPS |

These require income imputation (like PE's enhanced CPS) to fix.

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MICROPLEX PE PARITY FRAMEWORK                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TargetRegistry (7,291 targets)                                    │
│  ───────────────────────────────                                   │
│  • State populations (51)     • SLDU populations (1,950)           │
│  • CD populations (440)       • SLDL populations (4,813)           │
│  • IRS SOI income (14)        • Benefit programs (8)               │
│  • Health insurance (7)       • Tax/demographics (8)               │
│                                                                     │
│  CalibrationHarness                                                │
│  ─────────────────────                                             │
│  • Select target subsets by category/level/group                   │
│  • Run IPF calibration experiments                                 │
│  • Track data availability vs accuracy                             │
│  • Compare results across experiments                              │
│                                                                     │
│  Usage:                                                            │
│  ───────                                                           │
│  from microplex import CalibrationHarness, run_pe_parity_suite     │
│  harness = CalibrationHarness()                                    │
│  result = harness.run_experiment(df, "my_test",                    │
│      groups=["state_population", "irs_soi_income"],                │
│      only_available=True)                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Target Categories

### ✅ Microplex Has (PE Doesn't)

| Category | Count | Description |
|----------|-------|-------------|
| Geographic (CD) | 436 | Population by Congressional District |
| Geographic (SLDU) | ~1,950 | Population by State Legislative District (Upper) |
| Age Distribution | 918 | Population by age band by state |
| AGI Distribution | 393 | Tax returns by filing status × AGI bracket |

### ✅ Both Have

| Category | Microplex | PE | Notes |
|----------|-----------|----|----|
| SNAP participation | ✓ | ✓ | National aggregate |
| SSI participation | ✓ | ✓ | National aggregate |
| Total population | ✓ | ✓ | 372M |
| State population | ✓ | ✓ | Via age distribution |

### ❌ PE Has (Microplex Needs)

| Category | Count | Notes | Priority |
|----------|-------|-------|----------|
| **IRS SOI Income Totals** | 17 | Employment, dividends, capital gains, etc. | HIGH |
| **CBO Aggregates** | 6 | SNAP $, SSI $, SS $, payroll/income tax | HIGH |
| **EITC Outlays** | 1 | $72.7B | HIGH |
| **Medicaid by State** | 713 | Enrollment + spending by eligibility | MEDIUM |
| **CHIP by State** | 765 | Enrollment + spending | MEDIUM |
| **ACA by State** | 102 | Marketplace enrollment + spending | MEDIUM |
| **Census Pop by State** | 52 | Simple state totals | LOW (have via age) |

## Priority Targets to Add

### Phase 1: Income Distribution (17 targets)

These require income columns in CPS microdata:

| Target | Value | CPS Column | Status |
|--------|-------|-----------|--------|
| employment_income | $9,022B | WSAL_VAL | 🟡 Need column |
| self_employment_income | $436B | SEMP_VAL | 🟡 Need column |
| taxable_pension_income | $828B | RET_VAL | 🟡 Need column |
| social_security | $774B | SS_VAL | 🟡 Need column |
| tax_exempt_pension_income | $580B | RET_VAL | 🟡 Need column |
| long_term_capital_gains | $1,137B | - | 🔴 Imputation needed |
| partnership_s_corp_income | $976B | - | 🔴 Imputation needed |
| qualified_dividend_income | $260B | DIV_VAL | 🟡 Need column |
| unemployment_compensation | $208B | UC_VAL | 🟡 Need column |
| taxable_interest_income | $127B | INT_VAL | 🟡 Need column |

### Phase 2: Benefit Program Spending (6 targets)

| Target | Value | Notes |
|--------|-------|-------|
| snap | $103B | SNAP total benefits |
| social_security | $2,624B | Old-age, survivors, disability |
| ssi | $78B | Supplemental Security Income |
| eitc | $73B | Earned Income Tax Credit |
| unemployment_compensation | $59B | UI benefits |

### Phase 3: State-Level Benefits (1,631 targets)

| Category | States | Targets |
|----------|--------|---------|
| Medicaid enrollment | 51 | 255 |
| Medicaid spending | 51 | 254 |
| CHIP enrollment | 51 | 153 |
| CHIP spending | 51 | 612 |
| ACA enrollment | 51 | 51 |
| ACA spending | 51 | 51 |
| Census population | 51 | 51 |

## Implementation Plan

### Step 1: Expand CPS Microdata
Load full CPS ASEC with income detail columns.

### Step 2: Add Income Targets
Create calibration targets for IRS SOI income totals.

### Step 3: Add Benefit Spending Targets
Add CBO/Treasury aggregate spending targets.

### Step 4: State-Level Benefits
Add Medicaid, CHIP, ACA enrollment by state.

### Step 5: Capital Gains Imputation
Model capital gains distribution (not directly in CPS).

## Current Calibration

With current targets, microplex calibrates to:
- **436 CD populations** (0% error with 500K sample)
- **~1,950 SLDU populations** (0% error with 500K sample)
- **918 age distributions** (8% error with 100K sample)

Missing PolicyEngine parity:
- Income totals by source (17)
- Benefit spending aggregates (6)
- State-level benefit enrollment (1,631)
