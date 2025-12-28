# Cross-Sectional Comparison Report

_Generated: 2025-12-27_

## Data Summary

- **Records**: 142,125
- **Total Weight**: 337,689,642
- **Weight Column**: weight

## IRS SOI Comparison (2021 targets)

| Variable | IRS Target | CPS Weighted | Ratio | Status |
|----------|------------|--------------|-------|--------|
| total_returns | 150M | - | - | ⏳ |
| wage_salary_income | 9,431B | 12,072B | 1.28 | ⚠️ |
| self_employment_income | 397B | 532B | 1.34 | ⚠️ |
| interest_income | 117B | 817B | 6.98 | ❌ |
| dividend_income | 398B | 218B | 0.55 | ⚠️ |
| capital_gains | 1,648B | missing | - | ❓ |
| social_security_income | 413B | 1,226B | 2.97 | ❌ |
| rental_income | 168B | 239B | 1.42 | ⚠️ |

## Census Comparison (2023 targets)

| Variable | Census Target | CPS Weighted | Ratio | Status |
|----------|---------------|--------------|-------|--------|
| total_population | 334.9M | 337.7M | 1.01 | ✅ |
| population_under_18 | 72.8M | 73.0M | 1.00 | ✅ |
| population_18_64 | 205.0M | 203.2M | 0.99 | ✅ |
| population_65_plus | 57.1M | 61.5M | 1.08 | ⚠️ |

## SSA Comparison (2023 targets)

| Variable | SSA Target | CPS Weighted | Ratio | Status |
|----------|------------|--------------|-------|--------|
| ss_recipients | 66.7M | 59.2M | 0.89 | ✅ |
| ss_benefits_total | 1,352B | 1,226B | 0.91 | ✅ |

## Key Observations

### Issues to Address

1. **Interest Income (6.98x)**: CPS significantly overreports interest income vs IRS.
   - IRS reports taxable interest on returns
   - CPS asks about all interest received
   - Need: Imputation adjustment or IRS-based correction

2. **Social Security Income (2.97x)**: CPS overreports SS income.
   - IRS reports taxable portion only (~50-85% of benefits)
   - CPS reports total benefits
   - This is expected - not an error

3. **Dividend Income (0.55x)**: CPS underreports dividends.
   - High-income households with significant dividends undersampled
   - Need: Top-coding adjustment or PUF imputation

### What's Working

- Demographics closely match Census (within 1-8%)
- Wage income reasonably close (1.28x - some expected difference)
- SS recipients close to SSA (0.89x)
