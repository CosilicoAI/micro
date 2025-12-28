# Target Expansion Plan for Microplex Calibration

**Goal**: Expand from ~50 usable targets to ~2,500+ (PolicyEngine parity)

## Current State

| Category | Current | PE Has | Gap |
|----------|---------|--------|-----|
| IRS AGI brackets (national) | 16 | 16 | ✅ |
| IRS AGI × filing status | 0 | 64 | 64 |
| IRS AGI × state | 0 | 800+ | 800+ |
| EITC by children | 0 | 4 | 4 |
| CTC/ACTC | 0 | 2 | 2 |
| State population | 10 | 51 | 41 |
| Congressional districts | 0 | 436 | 436 |
| Age distribution | 0 | 18 groups × 51 | 918 |
| Medicaid enrollment | 0 | 51 | 51 |
| SNAP by state | 10 | 51 | 41 |
| SSI | 3 | ~50 | ~47 |
| **Total** | **~50** | **~2,800** | **~2,750** |

## Phase 1: Core Tax Targets (Priority: High)

### 1.1 Filing Status × AGI (Est. +64 targets)

**Source**: IRS SOI Table 1.2

**Variables needed in CPS**:
- `filing_status` (derived from marital_status)
  - Single
  - Married Filing Jointly
  - Married Filing Separately
  - Head of Household

**ETL Task**:
```python
# Add to etl_soi.py
def fetch_soi_filing_status():
    """Fetch IRS SOI Table 1.2 - by filing status and AGI"""
    # 4 filing statuses × 16 AGI brackets = 64 targets
```

### 1.2 State-Level Tax Returns (Est. +800 targets)

**Source**: IRS SOI Historic Table 2 (state data)

**Variables needed in CPS**:
- `state_fips` (already have)
- `agi` (already have)

**ETL Task**:
```python
# Expand etl_soi_state.py to all 50 states + DC
def fetch_soi_state_all():
    """Fetch returns and AGI by state × AGI bracket"""
    # 51 states × 16 AGI brackets = 816 targets
```

### 1.3 EITC by Number of Children (Est. +4 targets)

**Source**: IRS SOI (lines 59661-59664)

**Variables needed in CPS**:
- `n_qualifying_children` (need to derive from family relationships)

**ETL Task**:
```python
def fetch_eitc_by_children():
    """EITC claims by 0, 1, 2, 3+ children"""
    # National: 4 targets
    # By state: 4 × 51 = 204 targets (Phase 2)
```

## Phase 2: Demographics (Priority: Medium)

### 2.1 Age Distribution (Est. +918 targets)

**Source**: Census Table S0101

**Variables needed in CPS**:
- `age` (already have)

**ETL Task**:
```python
def fetch_census_age():
    """Population by 18 age groups × 51 states"""
    # Age groups: 0-4, 5-9, 10-14, ..., 80-84, 85+
```

### 2.2 All States Population/Households (Est. +82 targets)

**Source**: Census Population Estimates

**Currently have**: 10 states
**Need**: All 51

## Phase 3: Benefit Programs (Priority: Medium)

### 3.1 Medicaid Enrollment (Est. +51 targets)

**Source**: CMS/Medicaid.gov

**Variables needed in CPS**:
- `medicaid_enrolled` (exists as `public_assistance_income > 0` proxy)

### 3.2 SNAP Expansion (Est. +41 targets)

**Currently have**: 10 states
**Need**: All 51

### 3.3 SSI by State (Est. +47 targets)

**Source**: SSA

**Variables needed in CPS**:
- `ssi_income` (already have)

## Phase 4: Advanced (Priority: Lower)

### 4.1 Congressional Districts (Est. +436 targets)

**Source**: Census ACS 5-year

**Variables needed in CPS**:
- `congressional_district` (not in CPS, would need ACS)

### 4.2 Income by Source (Est. +100 targets)

**Source**: IRS SOI

**Variables**:
- Interest income
- Dividend income
- Capital gains
- Partnership/S-Corp income

### 4.3 Deductions (Est. +50 targets)

**Source**: IRS SOI

**Variables**:
- SALT deduction
- Mortgage interest
- Charitable contributions

## Implementation Order

```
Week 1: Filing status × AGI (64 targets)
        - Update CPS filing_status mapping
        - ETL for SOI Table 1.2
        - Add constraints to calibration

Week 2: State expansion (816 targets)
        - ETL for SOI state data
        - All 51 states × 16 AGI brackets

Week 3: Demographics (918 targets)
        - Census age distribution
        - Population by state

Week 4: Benefits (139 targets)
        - Medicaid
        - SNAP all states
        - SSI all states

Total: ~1,937 targets (70% of PE)
```

## CPS Variable Mapping Required

| Target Variable | CPS Variable | Status |
|-----------------|--------------|--------|
| `filing_status` | Derive from `marital_status` | Need mapping |
| `n_qualifying_children` | Derive from family relationships | Need logic |
| `congressional_district` | Not in CPS | Need ACS or imputation |
| `medicaid_enrolled` | `public_assistance_income > 0` | Proxy only |
| `eitc_amount` | `eitc_received` | Have it |
| `ctc_amount` | `ctc_received` | Have it |

## Notes

1. **Synthesis helps feasibility**: With 100M+ synthetic households, calibration has more degrees of freedom
2. **Expect ~5-10% residual error**: Different sources, different years, sampling variance
3. **GD for conflicting targets**: When infeasible, GD finds better compromise than IPF
4. **Prioritize tax targets**: These are most important for policy simulation accuracy
