"""
Target Loaders

Load calibration targets from various sources:
- IRS Statistics of Income (SOI) - income, deductions, credits
- Census ACS/CPS - demographics, household structure
- Admin data (SNAP, Medicaid, SSI, TANF, etc.)
- State-level targets from cosilico-data-sources

Target counts for parity:
- National: ~500 targets
- State-level: ~2500 targets (50 states × 50 variables)
- Total: ~3000 targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from microplex.targets.database import Target, TargetCategory
from microplex.targets.rac_mapping import RAC_VARIABLE_MAP


# State FIPS codes
STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


def load_soi_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """
    Load IRS Statistics of Income targets.

    Covers:
    - AGI distribution by bracket and filing status
    - Income sources (wages, dividends, capital gains, etc.)
    - Deductions (itemized, charitable, SALT)
    - Tax liability

    These are the most comprehensive and authoritative targets.
    """
    targets = []

    # IRS SOI 2021 National Targets (hardcoded for reliability)
    # Source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns
    AGI_BRACKETS = [
        ("under_1", -np.inf, 1),
        ("1_to_5k", 1, 5000),
        ("5k_to_10k", 5000, 10000),
        ("10k_to_15k", 10000, 15000),
        ("15k_to_20k", 15000, 20000),
        ("20k_to_25k", 20000, 25000),
        ("25k_to_30k", 25000, 30000),
        ("30k_to_40k", 30000, 40000),
        ("40k_to_50k", 40000, 50000),
        ("50k_to_75k", 50000, 75000),
        ("75k_to_100k", 75000, 100000),
        ("100k_to_200k", 100000, 200000),
        ("200k_to_500k", 200000, 500000),
        ("500k_to_1m", 500000, 1000000),
        ("1m_plus", 1000000, np.inf),
    ]

    # Returns by AGI bracket (IRS SOI Table 1.1, 2021)
    RETURNS_BY_AGI = {
        "under_1": 1_686_440,
        "1_to_5k": 5_183_390,
        "5k_to_10k": 7_929_860,
        "10k_to_15k": 9_883_050,
        "15k_to_20k": 9_113_990,
        "20k_to_25k": 8_186_640,
        "25k_to_30k": 7_407_890,
        "30k_to_40k": 13_194_450,
        "40k_to_50k": 10_930_780,
        "50k_to_75k": 19_494_660,
        "75k_to_100k": 15_137_070,
        "100k_to_200k": 22_849_380,
        "200k_to_500k": 7_167_290,
        "500k_to_1m": 1_106_040,
        "1m_plus": 664_340,
    }

    # AGI by bracket (IRS SOI Table 1.1, 2021)
    AGI_BY_BRACKET = {
        "under_1": -94_000_000_000,
        "1_to_5k": 15_000_000_000,
        "5k_to_10k": 59_000_000_000,
        "10k_to_15k": 123_000_000_000,
        "15k_to_20k": 160_000_000_000,
        "20k_to_25k": 184_000_000_000,
        "25k_to_30k": 204_000_000_000,
        "30k_to_40k": 461_000_000_000,
        "40k_to_50k": 492_000_000_000,
        "50k_to_75k": 1_210_000_000_000,
        "75k_to_100k": 1_316_000_000_000,
        "100k_to_200k": 3_187_000_000_000,
        "200k_to_500k": 2_161_000_000_000,
        "500k_to_1m": 762_000_000_000,
        "1m_plus": 4_466_000_000_000,
    }

    # Add return count targets
    for bracket, (name, lower, upper) in zip(RETURNS_BY_AGI.keys(), AGI_BRACKETS):
        targets.append(Target(
            name=f"returns_{bracket}",
            category=TargetCategory.AGI_DISTRIBUTION,
            value=RETURNS_BY_AGI[bracket],
            year=year,
            source="IRS SOI Table 1.1",
            source_url="https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns",
            geography="US",
            agi_lower=lower,
            agi_upper=upper,
            is_count=True,
            rac_variable="is_tax_filer",
            rac_statute="26/6012",
            microdata_column="is_tax_filer",
        ))

    # Add AGI amount targets
    for bracket, (name, lower, upper) in zip(AGI_BY_BRACKET.keys(), AGI_BRACKETS):
        targets.append(Target(
            name=f"agi_{bracket}",
            category=TargetCategory.AGI_DISTRIBUTION,
            value=AGI_BY_BRACKET[bracket],
            year=year,
            source="IRS SOI Table 1.1",
            source_url="https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns",
            geography="US",
            agi_lower=lower,
            agi_upper=upper,
            is_count=False,
            rac_variable="adjusted_gross_income",
            rac_statute="26/62",
            microdata_column="adjusted_gross_income",
        ))

    # Income source totals (IRS SOI Table 1.4, 2021)
    INCOME_SOURCES = {
        "wages_and_salaries": (9_909_000_000_000, "employment_income", "26/61/a/1"),
        "taxable_interest": (187_000_000_000, "taxable_interest_income", "26/61/a/4"),
        "tax_exempt_interest": (71_000_000_000, "tax_exempt_interest_income", "26/103"),
        "ordinary_dividends": (412_000_000_000, "dividend_income", "26/61/a/7"),
        "qualified_dividends": (297_000_000_000, "qualified_dividend_income", "26/1/h/11"),
        "business_income": (419_000_000_000, "self_employment_income", "26/1402"),
        "business_losses": (128_000_000_000, "business_net_losses", "26/1402"),
        "capital_gains": (2_531_000_000_000, "capital_gains", "26/1222"),
        "capital_losses": (23_000_000_000, "capital_gains_losses", "26/1211"),
        "ira_distributions": (312_000_000_000, "ira_distributions", "26/408"),
        "pensions_annuities": (749_000_000_000, "pension_income", "26/72"),
        "social_security": (625_000_000_000, "social_security_income", "26/86"),
        "partnership_s_corp_income": (1_113_000_000_000, "partnership_s_corp_income", "26/702"),
        "partnership_s_corp_losses": (241_000_000_000, "partnership_s_corp_losses", "26/702"),
        "rental_royalty_income": (156_000_000_000, "rental_income", "26/61/a/5"),
        "rental_royalty_losses": (89_000_000_000, "rental_losses", "26/469"),
        "estate_trust_income": (47_000_000_000, "estate_income", "26/641"),
        "estate_trust_losses": (12_000_000_000, "estate_losses", "26/641"),
        "unemployment": (129_000_000_000, "unemployment_compensation", "26/85"),
    }

    for name, (value, rac_var, statute) in INCOME_SOURCES.items():
        targets.append(Target(
            name=f"total_{name}",
            category=TargetCategory.INCOME_SOURCES,
            value=value,
            year=year,
            source="IRS SOI Table 1.4",
            source_url="https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns",
            geography="US",
            is_count=False,
            rac_variable=rac_var,
            rac_statute=statute,
            microdata_column=rac_var,
        ))

    # Deduction totals (IRS SOI Table 2.1, 2021)
    DEDUCTIONS = {
        "itemized_deductions": (1_847_000_000_000, "itemized_deductions", "26/63/d"),
        "charitable_contributions": (298_000_000_000, "charitable_deduction", "26/170"),
        "interest_paid": (192_000_000_000, "mortgage_interest_deduction", "26/163/h"),
        "taxes_paid": (423_000_000_000, "salt_deduction", "26/164"),
        "medical_expenses": (105_000_000_000, "medical_expense_deduction", "26/213"),
        "qbi_deduction": (203_000_000_000, "qbi_deduction", "26/199A"),
    }

    for name, (value, rac_var, statute) in DEDUCTIONS.items():
        targets.append(Target(
            name=f"total_{name}",
            category=TargetCategory.DEDUCTIONS,
            value=value,
            year=year,
            source="IRS SOI Table 2.1",
            source_url="https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns",
            geography="US",
            is_count=False,
            rac_variable=rac_var,
            rac_statute=statute,
            microdata_column=rac_var,
        ))

    return targets


def load_eitc_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """
    Load Earned Income Tax Credit targets.

    Source: IRS SOI EITC Statistics
    """
    targets = []

    # National EITC totals (IRS, 2021)
    targets.append(Target(
        name="eitc_claims",
        category=TargetCategory.EITC,
        value=31_000_000,
        year=year,
        source="IRS SOI EITC Statistics",
        source_url="https://www.irs.gov/statistics/soi-tax-stats-eitc-claims",
        geography="US",
        is_count=True,
        rac_variable="earned_income_credit",
        rac_statute="26/32",
        microdata_column="eitc",
    ))

    targets.append(Target(
        name="eitc_amount",
        category=TargetCategory.EITC,
        value=64_000_000_000,
        year=year,
        source="IRS SOI EITC Statistics",
        source_url="https://www.irs.gov/statistics/soi-tax-stats-eitc-claims",
        geography="US",
        is_count=False,
        rac_variable="earned_income_credit",
        rac_statute="26/32",
        microdata_column="eitc",
    ))

    # EITC by number of children (IRS, 2021)
    EITC_BY_CHILDREN = {
        "0_children": (6_500_000, 3_500_000_000),
        "1_child": (7_800_000, 17_000_000_000),
        "2_children": (8_200_000, 23_000_000_000),
        "3_plus_children": (8_500_000, 20_500_000_000),
    }

    for children, (count, amount) in EITC_BY_CHILDREN.items():
        targets.append(Target(
            name=f"eitc_claims_{children}",
            category=TargetCategory.EITC,
            value=count,
            year=year,
            source="IRS SOI EITC Statistics",
            geography="US",
            is_count=True,
            rac_variable="earned_income_credit",
            rac_statute="26/32",
            notes=f"EITC claims with {children.replace('_', ' ')}",
        ))

        targets.append(Target(
            name=f"eitc_amount_{children}",
            category=TargetCategory.EITC,
            value=amount,
            year=year,
            source="IRS SOI EITC Statistics",
            geography="US",
            is_count=False,
            rac_variable="earned_income_credit",
            rac_statute="26/32",
            notes=f"EITC amount with {children.replace('_', ' ')}",
        ))

    return targets


def load_ctc_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """Load Child Tax Credit targets."""
    targets = []

    # CTC totals (IRS, 2021 - expanded under ARPA)
    targets.append(Target(
        name="ctc_claims",
        category=TargetCategory.CTC,
        value=48_000_000,
        year=year,
        source="IRS SOI CTC Statistics",
        geography="US",
        is_count=True,
        rac_variable="child_tax_credit",
        rac_statute="26/24",
        microdata_column="ctc",
    ))

    targets.append(Target(
        name="ctc_amount",
        category=TargetCategory.CTC,
        value=122_000_000_000,
        year=year,
        source="IRS SOI CTC Statistics",
        geography="US",
        is_count=False,
        rac_variable="child_tax_credit",
        rac_statute="26/24",
        microdata_column="ctc",
    ))

    # Refundable portion (ACTC)
    targets.append(Target(
        name="actc_claims",
        category=TargetCategory.ACTC,
        value=25_000_000,
        year=year,
        source="IRS SOI CTC Statistics",
        geography="US",
        is_count=True,
        rac_variable="additional_child_tax_credit",
        rac_statute="26/24/h",
        microdata_column="refundable_ctc",
    ))

    targets.append(Target(
        name="actc_amount",
        category=TargetCategory.ACTC,
        value=41_000_000_000,
        year=year,
        source="IRS SOI CTC Statistics",
        geography="US",
        is_count=False,
        rac_variable="additional_child_tax_credit",
        rac_statute="26/24/h",
        microdata_column="refundable_ctc",
    ))

    return targets


def load_snap_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """
    Load SNAP (food stamps) targets.

    Source: USDA Food and Nutrition Service
    """
    targets = []

    # National SNAP totals (USDA FNS, FY2021)
    targets.append(Target(
        name="snap_households",
        category=TargetCategory.SNAP,
        value=21_600_000,
        year=year,
        source="USDA FNS SNAP Data",
        source_url="https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap",
        geography="US",
        is_count=True,
        rac_variable="snap_benefit",
        rac_statute="7/2017",
        microdata_column="snap_participation",
    ))

    targets.append(Target(
        name="snap_participants",
        category=TargetCategory.SNAP,
        value=41_500_000,
        year=year,
        source="USDA FNS SNAP Data",
        source_url="https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap",
        geography="US",
        is_count=True,
        rac_variable="snap_benefit",
        rac_statute="7/2017",
        microdata_column="snap_participation",
        notes="Individual participants",
    ))

    targets.append(Target(
        name="snap_benefits",
        category=TargetCategory.SNAP,
        value=113_000_000_000,
        year=year,
        source="USDA FNS SNAP Data",
        source_url="https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap",
        geography="US",
        is_count=False,
        rac_variable="snap_benefit",
        rac_statute="7/2017",
        microdata_column="snap",
    ))

    return targets


def load_medicaid_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """
    Load Medicaid enrollment targets.

    Source: CMS Medicaid & CHIP Enrollment Data
    """
    targets = []

    # National Medicaid totals (CMS, 2021)
    targets.append(Target(
        name="medicaid_enrollment",
        category=TargetCategory.MEDICAID,
        value=85_000_000,
        year=year,
        source="CMS Medicaid Enrollment",
        source_url="https://www.medicaid.gov/medicaid/program-information/medicaid-chip-enrollment-data",
        geography="US",
        is_count=True,
        rac_variable="medicaid_eligible",
        rac_statute="42/1396a",
        microdata_column="is_medicaid_eligible",
    ))

    targets.append(Target(
        name="chip_enrollment",
        category=TargetCategory.MEDICAID,
        value=7_000_000,
        year=year,
        source="CMS CHIP Enrollment",
        source_url="https://www.medicaid.gov/medicaid/program-information/medicaid-chip-enrollment-data",
        geography="US",
        is_count=True,
        rac_variable="medicaid_eligible",
        rac_statute="42/1397aa",
        notes="Children's Health Insurance Program",
    ))

    return targets


def load_ssi_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """Load Supplemental Security Income targets from SSA."""
    targets = []

    # SSI totals (SSA, 2021)
    targets.append(Target(
        name="ssi_recipients",
        category=TargetCategory.SSI,
        value=7_800_000,
        year=year,
        source="SSA SSI Statistics",
        source_url="https://www.ssa.gov/policy/docs/statcomps/ssi_asr/",
        geography="US",
        is_count=True,
        rac_variable="ssi_benefit",
        rac_statute="42/1382",
        microdata_column="ssi",
    ))

    targets.append(Target(
        name="ssi_payments",
        category=TargetCategory.SSI,
        value=59_000_000_000,
        year=year,
        source="SSA SSI Statistics",
        source_url="https://www.ssa.gov/policy/docs/statcomps/ssi_asr/",
        geography="US",
        is_count=False,
        rac_variable="ssi_benefit",
        rac_statute="42/1382",
        microdata_column="ssi",
    ))

    return targets


def load_tanf_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """Load TANF cash assistance targets from HHS."""
    targets = []

    # TANF totals (HHS ACF, 2021)
    targets.append(Target(
        name="tanf_families",
        category=TargetCategory.TANF,
        value=1_100_000,
        year=year,
        source="HHS ACF TANF Data",
        source_url="https://www.acf.hhs.gov/ofa/programs/tanf/data-reports",
        geography="US",
        is_count=True,
        rac_variable="tanf_benefit",
        rac_statute="42/601",
        microdata_column="tanf",
    ))

    targets.append(Target(
        name="tanf_recipients",
        category=TargetCategory.TANF,
        value=2_400_000,
        year=year,
        source="HHS ACF TANF Data",
        source_url="https://www.acf.hhs.gov/ofa/programs/tanf/data-reports",
        geography="US",
        is_count=True,
        rac_variable="tanf_benefit",
        rac_statute="42/601",
        microdata_column="tanf",
        notes="Individual recipients",
    ))

    targets.append(Target(
        name="tanf_expenditures",
        category=TargetCategory.TANF,
        value=16_000_000_000,
        year=year,
        source="HHS ACF TANF Data",
        source_url="https://www.acf.hhs.gov/ofa/programs/tanf/data-reports",
        geography="US",
        is_count=False,
        rac_variable="tanf_benefit",
        rac_statute="42/601",
        microdata_column="tanf",
        notes="Total federal and state TANF expenditures",
    ))

    return targets


def load_housing_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """Load housing assistance targets from HUD."""
    targets = []

    # Housing assistance totals (HUD, 2021)
    targets.append(Target(
        name="housing_voucher_households",
        category=TargetCategory.HOUSING,
        value=2_300_000,
        year=year,
        source="HUD Picture of Subsidized Households",
        source_url="https://www.huduser.gov/portal/datasets/assthsg.html",
        geography="US",
        is_count=True,
        rac_variable="housing_subsidy",
        rac_statute="42/1437f",
        microdata_column="housing_subsidy",
        notes="Section 8 Housing Choice Voucher households",
    ))

    targets.append(Target(
        name="public_housing_households",
        category=TargetCategory.HOUSING,
        value=920_000,
        year=year,
        source="HUD Picture of Subsidized Households",
        source_url="https://www.huduser.gov/portal/datasets/assthsg.html",
        geography="US",
        is_count=True,
        rac_variable="housing_subsidy",
        rac_statute="42/1437",
        microdata_column="housing_subsidy",
        notes="Public housing residents",
    ))

    targets.append(Target(
        name="housing_assistance_spending",
        category=TargetCategory.HOUSING,
        value=52_000_000_000,
        year=year,
        source="HUD Budget",
        source_url="https://www.hud.gov/budget",
        geography="US",
        is_count=False,
        rac_variable="housing_subsidy",
        rac_statute="42/1437f",
        microdata_column="housing_subsidy",
    ))

    return targets


def load_aca_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """Load ACA/marketplace targets from CMS."""
    targets = []

    # ACA marketplace totals (CMS, 2021)
    targets.append(Target(
        name="marketplace_enrollment",
        category=TargetCategory.OTHER_CREDITS,
        value=12_000_000,
        year=year,
        source="CMS Marketplace Enrollment",
        source_url="https://www.cms.gov/research-statistics-data-and-systems",
        geography="US",
        is_count=True,
        rac_variable="premium_tax_credit",
        rac_statute="26/36B",
        microdata_column="marketplace_enrollment",
    ))

    targets.append(Target(
        name="premium_tax_credit_recipients",
        category=TargetCategory.OTHER_CREDITS,
        value=9_000_000,
        year=year,
        source="CMS Marketplace Enrollment",
        source_url="https://www.cms.gov/research-statistics-data-and-systems",
        geography="US",
        is_count=True,
        rac_variable="premium_tax_credit",
        rac_statute="26/36B",
        microdata_column="premium_tax_credit",
        notes="Marketplace enrollees receiving PTC",
    ))

    targets.append(Target(
        name="premium_tax_credit_amount",
        category=TargetCategory.OTHER_CREDITS,
        value=57_000_000_000,
        year=year,
        source="CMS/Treasury",
        source_url="https://www.cms.gov/research-statistics-data-and-systems",
        geography="US",
        is_count=False,
        rac_variable="premium_tax_credit",
        rac_statute="26/36B",
        microdata_column="premium_tax_credit",
    ))

    return targets


def load_wic_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """Load WIC (Women, Infants, and Children) targets from USDA."""
    targets = []

    # WIC totals (USDA FNS, 2021)
    targets.append(Target(
        name="wic_participants",
        category=TargetCategory.SNAP,  # Grouped with nutrition
        value=6_200_000,
        year=year,
        source="USDA FNS WIC Data",
        source_url="https://www.fns.usda.gov/pd/wic-program",
        geography="US",
        is_count=True,
        rac_variable="wic",
        rac_statute="42/1786",
        microdata_column="wic",
    ))

    targets.append(Target(
        name="wic_expenditures",
        category=TargetCategory.SNAP,
        value=5_000_000_000,
        year=year,
        source="USDA FNS WIC Data",
        source_url="https://www.fns.usda.gov/pd/wic-program",
        geography="US",
        is_count=False,
        rac_variable="wic",
        rac_statute="42/1786",
        microdata_column="wic",
    ))

    return targets


def load_other_benefit_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """Load other benefit program targets (school lunch, LIHEAP, CCDF)."""
    targets = []

    # School lunch (USDA, 2021)
    targets.append(Target(
        name="free_school_lunch_participants",
        category=TargetCategory.SNAP,
        value=22_000_000,
        year=year,
        source="USDA FNS School Meals",
        source_url="https://www.fns.usda.gov/pd/child-nutrition-tables",
        geography="US",
        is_count=True,
        rac_variable="school_lunch",
        rac_statute="42/1758",
        microdata_column="school_lunch",
        notes="Free lunch participants",
    ))

    targets.append(Target(
        name="reduced_price_lunch_participants",
        category=TargetCategory.SNAP,
        value=2_500_000,
        year=year,
        source="USDA FNS School Meals",
        geography="US",
        is_count=True,
        rac_variable="school_lunch",
        rac_statute="42/1758",
        microdata_column="school_lunch",
        notes="Reduced-price lunch participants",
    ))

    # LIHEAP (HHS, 2021)
    targets.append(Target(
        name="liheap_households",
        category=TargetCategory.HOUSING,
        value=5_400_000,
        year=year,
        source="HHS LIHEAP Data",
        source_url="https://www.acf.hhs.gov/ocs/programs/liheap",
        geography="US",
        is_count=True,
        rac_variable="liheap",
        rac_statute="42/8621",
        microdata_column="liheap",
    ))

    targets.append(Target(
        name="liheap_expenditures",
        category=TargetCategory.HOUSING,
        value=5_000_000_000,
        year=year,
        source="HHS LIHEAP Data",
        geography="US",
        is_count=False,
        rac_variable="liheap",
        rac_statute="42/8621",
        microdata_column="liheap",
    ))

    # CCDF Child Care (HHS, 2021)
    targets.append(Target(
        name="ccdf_children",
        category=TargetCategory.OTHER_CREDITS,
        value=1_400_000,
        year=year,
        source="HHS ACF CCDF Data",
        source_url="https://www.acf.hhs.gov/occ/data",
        geography="US",
        is_count=True,
        rac_variable="ccdf",
        rac_statute="42/9858",
        microdata_column="ccdf",
        notes="Children receiving CCDF child care subsidies",
    ))

    targets.append(Target(
        name="ccdf_expenditures",
        category=TargetCategory.OTHER_CREDITS,
        value=11_000_000_000,
        year=year,
        source="HHS ACF CCDF Data",
        geography="US",
        is_count=False,
        rac_variable="ccdf",
        rac_statute="42/9858",
        microdata_column="ccdf",
    ))

    return targets


def load_demographics_targets(year: int = 2021, source_path: Optional[Path] = None) -> List[Target]:
    """
    Load demographic targets from Census.

    Source: Census Bureau ACS/CPS
    """
    targets = []

    # Population totals (Census, 2021)
    targets.append(Target(
        name="total_population",
        category=TargetCategory.POPULATION,
        value=332_000_000,
        year=year,
        source="Census Bureau",
        source_url="https://www.census.gov/data.html",
        geography="US",
        is_count=True,
    ))

    targets.append(Target(
        name="total_households",
        category=TargetCategory.HOUSEHOLD_STRUCTURE,
        value=131_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
    ))

    targets.append(Target(
        name="total_tax_units",
        category=TargetCategory.HOUSEHOLD_STRUCTURE,
        value=154_000_000,
        year=year,
        source="IRS SOI",
        geography="US",
        is_count=True,
        rac_variable="is_tax_filer",
        rac_statute="26/6012",
    ))

    # Detailed age distribution (5-year brackets)
    AGE_GROUPS_DETAILED = {
        "0_to_4": 19_000_000,
        "5_to_9": 20_000_000,
        "10_to_14": 21_000_000,
        "15_to_19": 21_000_000,
        "20_to_24": 21_000_000,
        "25_to_29": 23_000_000,
        "30_to_34": 23_000_000,
        "35_to_39": 22_000_000,
        "40_to_44": 20_000_000,
        "45_to_49": 20_000_000,
        "50_to_54": 20_000_000,
        "55_to_59": 21_000_000,
        "60_to_64": 21_000_000,
        "65_to_69": 17_000_000,
        "70_to_74": 15_000_000,
        "75_to_79": 10_000_000,
        "80_to_84": 6_000_000,
        "85_plus": 7_000_000,
    }

    for age_group, count in AGE_GROUPS_DETAILED.items():
        targets.append(Target(
            name=f"population_{age_group}",
            category=TargetCategory.AGE_DISTRIBUTION,
            value=count,
            year=year,
            source="Census Bureau ACS",
            geography="US",
            is_count=True,
            rac_variable="age",
        ))

    # Summary age groups
    AGE_GROUPS = {
        "under_18": 73_000_000,
        "18_to_64": 200_000_000,
        "65_plus": 55_000_000,
    }

    for age_group, count in AGE_GROUPS.items():
        targets.append(Target(
            name=f"population_{age_group}",
            category=TargetCategory.AGE_DISTRIBUTION,
            value=count,
            year=year,
            source="Census Bureau ACS",
            geography="US",
            is_count=True,
            rac_variable="age",
        ))

    # By sex
    targets.append(Target(
        name="population_male",
        category=TargetCategory.POPULATION,
        value=163_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
    ))
    targets.append(Target(
        name="population_female",
        category=TargetCategory.POPULATION,
        value=169_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
    ))

    # Employment
    targets.append(Target(
        name="employed_population",
        category=TargetCategory.EMPLOYMENT,
        value=158_000_000,
        year=year,
        source="BLS",
        geography="US",
        is_count=True,
        rac_variable="employment_income",
        rac_statute="26/61/a/1",
    ))

    targets.append(Target(
        name="labor_force",
        category=TargetCategory.EMPLOYMENT,
        value=161_000_000,
        year=year,
        source="BLS",
        geography="US",
        is_count=True,
    ))

    targets.append(Target(
        name="unemployed_population",
        category=TargetCategory.UNEMPLOYMENT,
        value=6_000_000,
        year=year,
        source="BLS",
        geography="US",
        is_count=True,
        rac_variable="unemployment_compensation",
        rac_statute="26/85",
    ))

    targets.append(Target(
        name="not_in_labor_force",
        category=TargetCategory.EMPLOYMENT,
        value=100_000_000,
        year=year,
        source="BLS",
        geography="US",
        is_count=True,
    ))

    return targets


def load_filing_status_targets(year: int = 2021) -> List[Target]:
    """
    Load filing status breakdown targets from IRS SOI.

    Returns by filing status × AGI bracket = 60 count targets + 60 amount targets.
    """
    targets = []

    # IRS SOI Table 1.2: Returns by filing status and AGI bracket (2021)
    # Filing status: Single, MFJ (Married Filing Jointly), MFS, HOH (Head of Household)
    FILING_STATUS_DATA = {
        # (Single count, MFJ count, MFS count, HOH count)
        "under_25k": (32_000_000, 8_000_000, 1_500_000, 8_000_000),
        "25k_to_50k": (18_000_000, 12_000_000, 1_200_000, 6_000_000),
        "50k_to_75k": (10_000_000, 13_000_000, 800_000, 3_500_000),
        "75k_to_100k": (6_500_000, 11_000_000, 600_000, 2_000_000),
        "100k_to_200k": (7_000_000, 18_000_000, 700_000, 2_000_000),
        "200k_to_500k": (2_000_000, 6_000_000, 300_000, 400_000),
        "500k_plus": (600_000, 1_400_000, 100_000, 100_000),
    }

    AGI_BOUNDS = {
        "under_25k": (-np.inf, 25000),
        "25k_to_50k": (25000, 50000),
        "50k_to_75k": (50000, 75000),
        "75k_to_100k": (75000, 100000),
        "100k_to_200k": (100000, 200000),
        "200k_to_500k": (200000, 500000),
        "500k_plus": (500000, np.inf),
    }

    filing_statuses = ["single", "mfj", "mfs", "hoh"]

    for bracket, counts in FILING_STATUS_DATA.items():
        lower, upper = AGI_BOUNDS[bracket]
        for i, status in enumerate(filing_statuses):
            targets.append(Target(
                name=f"returns_{status}_{bracket}",
                category=TargetCategory.AGI_DISTRIBUTION,
                value=counts[i],
                year=year,
                source="IRS SOI Table 1.2",
                source_url="https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns",
                geography="US",
                filing_status=status.upper(),
                agi_lower=lower,
                agi_upper=upper,
                is_count=True,
                rac_variable="filing_status",
                rac_statute="26/1",
            ))

    # Total by filing status (2021)
    TOTAL_BY_STATUS = {
        "single": (76_000_000, 3_800_000_000_000),
        "mfj": (69_500_000, 9_200_000_000_000),
        "mfs": (5_200_000, 280_000_000_000),
        "hoh": (22_000_000, 630_000_000_000),
    }

    for status, (count, agi) in TOTAL_BY_STATUS.items():
        targets.append(Target(
            name=f"returns_{status}_total",
            category=TargetCategory.AGI_DISTRIBUTION,
            value=count,
            year=year,
            source="IRS SOI Table 1.2",
            geography="US",
            filing_status=status.upper(),
            is_count=True,
            rac_variable="filing_status",
            rac_statute="26/1",
        ))
        targets.append(Target(
            name=f"agi_{status}_total",
            category=TargetCategory.AGI_DISTRIBUTION,
            value=agi,
            year=year,
            source="IRS SOI Table 1.2",
            geography="US",
            filing_status=status.upper(),
            is_count=False,
            rac_variable="adjusted_gross_income",
            rac_statute="26/62",
        ))

    return targets


def load_social_security_targets(year: int = 2021) -> List[Target]:
    """
    Load Social Security and Medicare targets from SSA.
    """
    targets = []

    # OASDI (Old Age, Survivors, Disability Insurance) - SSA 2021
    targets.append(Target(
        name="oasdi_beneficiaries",
        category=TargetCategory.POPULATION,
        value=65_000_000,
        year=year,
        source="SSA Annual Statistical Supplement",
        source_url="https://www.ssa.gov/policy/docs/statcomps/supplement/",
        geography="US",
        is_count=True,
        rac_variable="social_security_income",
        rac_statute="42/402",
    ))

    targets.append(Target(
        name="oasdi_retired_workers",
        category=TargetCategory.POPULATION,
        value=50_000_000,
        year=year,
        source="SSA Annual Statistical Supplement",
        geography="US",
        is_count=True,
        rac_variable="social_security_income",
        rac_statute="42/402",
        notes="Retired worker beneficiaries",
    ))

    targets.append(Target(
        name="oasdi_disabled_workers",
        category=TargetCategory.POPULATION,
        value=7_900_000,
        year=year,
        source="SSA Annual Statistical Supplement",
        geography="US",
        is_count=True,
        rac_variable="social_security_income",
        rac_statute="42/423",
        notes="Disabled worker beneficiaries",
    ))

    targets.append(Target(
        name="oasdi_survivors",
        category=TargetCategory.POPULATION,
        value=5_900_000,
        year=year,
        source="SSA Annual Statistical Supplement",
        geography="US",
        is_count=True,
        rac_variable="social_security_income",
        rac_statute="42/402",
        notes="Survivors beneficiaries",
    ))

    targets.append(Target(
        name="oasdi_total_benefits",
        category=TargetCategory.INCOME_SOURCES,
        value=1_133_000_000_000,
        year=year,
        source="SSA Annual Statistical Supplement",
        geography="US",
        is_count=False,
        rac_variable="social_security_income",
        rac_statute="42/402",
    ))

    # Average benefits
    targets.append(Target(
        name="oasdi_avg_retired_benefit",
        category=TargetCategory.INCOME_SOURCES,
        value=1_565 * 12,  # Monthly to annual
        year=year,
        source="SSA Annual Statistical Supplement",
        geography="US",
        is_count=False,
        rac_variable="social_security_income",
        notes="Average annual retired worker benefit",
    ))

    # Medicare enrollment (CMS, 2021)
    targets.append(Target(
        name="medicare_total_enrollment",
        category=TargetCategory.MEDICAID,
        value=63_000_000,
        year=year,
        source="CMS Medicare Enrollment",
        source_url="https://www.cms.gov/research-statistics-data-systems",
        geography="US",
        is_count=True,
        notes="Total Medicare enrollment (Part A + B)",
    ))

    targets.append(Target(
        name="medicare_part_d_enrollment",
        category=TargetCategory.MEDICAID,
        value=49_000_000,
        year=year,
        source="CMS Medicare Enrollment",
        geography="US",
        is_count=True,
        notes="Medicare Part D (prescription drug) enrollment",
    ))

    return targets


def load_race_ethnicity_targets(year: int = 2021) -> List[Target]:
    """
    Load race and ethnicity demographic targets from Census.
    """
    targets = []

    # Census 2021 ACS estimates
    RACE_ETHNICITY = {
        "white_alone": 204_000_000,
        "black_alone": 41_000_000,
        "asian_alone": 19_500_000,
        "hispanic_any_race": 62_000_000,
        "two_or_more_races": 10_000_000,
        "american_indian_alaska_native": 3_000_000,
        "native_hawaiian_pacific_islander": 700_000,
    }

    for race, count in RACE_ETHNICITY.items():
        targets.append(Target(
            name=f"population_{race}",
            category=TargetCategory.POPULATION,
            value=count,
            year=year,
            source="Census Bureau ACS",
            source_url="https://data.census.gov/",
            geography="US",
            is_count=True,
        ))

    # Households by race of householder
    HOUSEHOLDS_BY_RACE = {
        "white_alone": 98_000_000,
        "black_alone": 17_000_000,
        "asian_alone": 7_000_000,
        "hispanic_any_race": 19_000_000,
    }

    for race, count in HOUSEHOLDS_BY_RACE.items():
        targets.append(Target(
            name=f"households_{race}",
            category=TargetCategory.HOUSEHOLD_STRUCTURE,
            value=count,
            year=year,
            source="Census Bureau ACS",
            geography="US",
            is_count=True,
        ))

    return targets


def load_education_targets(year: int = 2021) -> List[Target]:
    """
    Load education attainment targets from Census.
    """
    targets = []

    # Educational attainment for population 25+ (Census ACS 2021)
    EDUCATION_LEVELS = {
        "less_than_hs": 24_000_000,
        "high_school_diploma": 56_000_000,
        "some_college": 44_000_000,
        "associates_degree": 22_000_000,
        "bachelors_degree": 48_000_000,
        "graduate_degree": 28_000_000,
    }

    for level, count in EDUCATION_LEVELS.items():
        targets.append(Target(
            name=f"education_{level}",
            category=TargetCategory.POPULATION,
            value=count,
            year=year,
            source="Census Bureau ACS",
            source_url="https://data.census.gov/",
            geography="US",
            is_count=True,
            notes="Population 25 years and over",
        ))

    return targets


def load_disability_targets(year: int = 2021) -> List[Target]:
    """
    Load disability status targets from Census and SSA.
    """
    targets = []

    # Census ACS disability estimates (2021)
    targets.append(Target(
        name="population_with_disability",
        category=TargetCategory.POPULATION,
        value=42_000_000,
        year=year,
        source="Census Bureau ACS",
        source_url="https://data.census.gov/",
        geography="US",
        is_count=True,
        notes="Civilian noninstitutionalized population with a disability",
    ))

    # By age group
    DISABILITY_BY_AGE = {
        "under_18": 3_000_000,
        "18_to_64": 20_000_000,
        "65_plus": 19_000_000,
    }

    for age_group, count in DISABILITY_BY_AGE.items():
        targets.append(Target(
            name=f"disability_{age_group}",
            category=TargetCategory.POPULATION,
            value=count,
            year=year,
            source="Census Bureau ACS",
            geography="US",
            is_count=True,
            rac_variable="age",
        ))

    # SSDI recipients (SSA 2021)
    targets.append(Target(
        name="ssdi_recipients",
        category=TargetCategory.SSI,
        value=8_500_000,
        year=year,
        source="SSA SSDI Statistics",
        source_url="https://www.ssa.gov/policy/docs/statcomps/di_asr/",
        geography="US",
        is_count=True,
        notes="Social Security Disability Insurance beneficiaries",
    ))

    targets.append(Target(
        name="ssdi_total_benefits",
        category=TargetCategory.SSI,
        value=150_000_000_000,
        year=year,
        source="SSA SSDI Statistics",
        geography="US",
        is_count=False,
    ))

    return targets


def load_household_composition_targets(year: int = 2021) -> List[Target]:
    """
    Load detailed household composition targets from Census.
    """
    targets = []

    # Household types (Census ACS 2021)
    HOUSEHOLD_TYPES = {
        "married_couple_families": 59_000_000,
        "male_householder_no_spouse": 6_000_000,
        "female_householder_no_spouse": 15_000_000,
        "nonfamily_living_alone": 37_000_000,
        "nonfamily_not_alone": 7_000_000,
    }

    for hh_type, count in HOUSEHOLD_TYPES.items():
        targets.append(Target(
            name=f"households_{hh_type}",
            category=TargetCategory.HOUSEHOLD_STRUCTURE,
            value=count,
            year=year,
            source="Census Bureau ACS",
            source_url="https://data.census.gov/",
            geography="US",
            is_count=True,
        ))

    # Households with children
    targets.append(Target(
        name="households_with_children_under_18",
        category=TargetCategory.HOUSEHOLD_STRUCTURE,
        value=35_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
    ))

    targets.append(Target(
        name="households_with_children_under_6",
        category=TargetCategory.HOUSEHOLD_STRUCTURE,
        value=16_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
    ))

    # By number of children
    CHILDREN_COUNTS = {
        "0_children": 92_000_000,
        "1_child": 16_000_000,
        "2_children": 12_000_000,
        "3_plus_children": 7_000_000,
    }

    for n_children, count in CHILDREN_COUNTS.items():
        targets.append(Target(
            name=f"households_{n_children}",
            category=TargetCategory.HOUSEHOLD_STRUCTURE,
            value=count,
            year=year,
            source="Census Bureau ACS",
            geography="US",
            is_count=True,
        ))

    # Household size
    HOUSEHOLD_SIZE = {
        "1_person": 37_000_000,
        "2_person": 44_000_000,
        "3_person": 19_000_000,
        "4_person": 16_000_000,
        "5_plus_person": 12_000_000,
    }

    for size, count in HOUSEHOLD_SIZE.items():
        targets.append(Target(
            name=f"households_{size}",
            category=TargetCategory.HOUSEHOLD_STRUCTURE,
            value=count,
            year=year,
            source="Census Bureau ACS",
            geography="US",
            is_count=True,
        ))

    return targets


def load_employment_industry_targets(year: int = 2021) -> List[Target]:
    """
    Load employment by industry targets from BLS.
    """
    targets = []

    # Employment by major industry (BLS 2021, in thousands)
    INDUSTRY_EMPLOYMENT = {
        "agriculture_forestry": 2_100_000,
        "mining_logging": 600_000,
        "construction": 7_500_000,
        "manufacturing": 12_500_000,
        "wholesale_trade": 5_800_000,
        "retail_trade": 15_000_000,
        "transportation_warehousing": 6_200_000,
        "utilities": 550_000,
        "information": 2_900_000,
        "financial_activities": 8_800_000,
        "professional_business_services": 21_500_000,
        "education_health_services": 24_000_000,
        "leisure_hospitality": 15_000_000,
        "other_services": 5_500_000,
        "government": 22_000_000,
    }

    for industry, employment in INDUSTRY_EMPLOYMENT.items():
        targets.append(Target(
            name=f"employment_{industry}",
            category=TargetCategory.EMPLOYMENT,
            value=employment,
            year=year,
            source="BLS Current Employment Statistics",
            source_url="https://www.bls.gov/ces/",
            geography="US",
            is_count=True,
            rac_variable="employment_income",
            rac_statute="26/61/a/1",
        ))

    # Wages by industry (billions)
    INDUSTRY_WAGES = {
        "manufacturing": 950_000_000_000,
        "professional_business_services": 1_800_000_000_000,
        "education_health_services": 1_600_000_000_000,
        "financial_activities": 700_000_000_000,
        "government": 1_400_000_000_000,
    }

    for industry, wages in INDUSTRY_WAGES.items():
        targets.append(Target(
            name=f"wages_{industry}",
            category=TargetCategory.INCOME_SOURCES,
            value=wages,
            year=year,
            source="BLS Quarterly Census of Employment and Wages",
            source_url="https://www.bls.gov/cew/",
            geography="US",
            is_count=False,
            rac_variable="employment_income",
            rac_statute="26/61/a/1",
        ))

    return targets


def load_wealth_targets(year: int = 2021) -> List[Target]:
    """
    Load wealth and asset targets from Federal Reserve SCF.
    """
    targets = []

    # Median and mean wealth by age (Fed SCF, 2021)
    WEALTH_BY_AGE = {
        "under_35": (14_000, 76_000, 17_000_000),
        "35_to_44": (91_000, 310_000, 23_000_000),
        "45_to_54": (168_000, 520_000, 22_000_000),
        "55_to_64": (250_000, 1_175_000, 21_000_000),
        "65_to_74": (266_000, 1_215_000, 17_000_000),
        "75_plus": (254_000, 977_000, 15_000_000),
    }

    for age_group, (median, mean, count) in WEALTH_BY_AGE.items():
        targets.append(Target(
            name=f"households_{age_group}",
            category=TargetCategory.HOUSEHOLD_STRUCTURE,
            value=count,
            year=year,
            source="Federal Reserve SCF",
            source_url="https://www.federalreserve.gov/econres/scfindex.htm",
            geography="US",
            is_count=True,
            rac_variable="age",
        ))

    # Homeownership
    targets.append(Target(
        name="owner_occupied_households",
        category=TargetCategory.HOUSEHOLD_STRUCTURE,
        value=83_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
    ))

    targets.append(Target(
        name="renter_households",
        category=TargetCategory.HOUSEHOLD_STRUCTURE,
        value=45_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
    ))

    # Retirement accounts (Fed SCF)
    targets.append(Target(
        name="households_with_retirement_accounts",
        category=TargetCategory.HOUSEHOLD_STRUCTURE,
        value=65_000_000,
        year=year,
        source="Federal Reserve SCF",
        geography="US",
        is_count=True,
    ))

    return targets


def load_poverty_targets(year: int = 2021) -> List[Target]:
    """
    Load poverty and income threshold targets from Census.
    """
    targets = []

    # Official poverty (Census 2021)
    targets.append(Target(
        name="population_in_poverty",
        category=TargetCategory.POPULATION,
        value=37_900_000,
        year=year,
        source="Census Bureau ACS",
        source_url="https://www.census.gov/topics/income-poverty/poverty.html",
        geography="US",
        is_count=True,
    ))

    # By age
    POVERTY_BY_AGE = {
        "children_in_poverty": 10_900_000,
        "adults_18_64_in_poverty": 22_000_000,
        "seniors_65_plus_in_poverty": 5_000_000,
    }

    for group, count in POVERTY_BY_AGE.items():
        targets.append(Target(
            name=group,
            category=TargetCategory.POPULATION,
            value=count,
            year=year,
            source="Census Bureau ACS",
            geography="US",
            is_count=True,
            rac_variable="age",
        ))

    # Deep poverty (<50% FPL)
    targets.append(Target(
        name="population_in_deep_poverty",
        category=TargetCategory.POPULATION,
        value=17_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
        notes="Below 50% of poverty line",
    ))

    # Near poverty (100-150% FPL)
    targets.append(Target(
        name="population_near_poverty",
        category=TargetCategory.POPULATION,
        value=30_000_000,
        year=year,
        source="Census Bureau ACS",
        geography="US",
        is_count=True,
        notes="100-150% of poverty line",
    ))

    return targets


def load_state_demographics_targets(
    year: int = 2021,
    data_path: Optional[Path] = None,
) -> List[Target]:
    """
    Load state-level demographic targets from cosilico-data-sources.

    Returns ~500 targets (51 states × 10 variables).
    """
    targets = []

    if data_path is None:
        data_path = Path.home() / "CosilicoAI" / "cosilico-data-sources" / "data" / "targets"

    parquet_path = data_path / "state_demographics.parquet"
    if not parquet_path.exists():
        return targets

    df = pd.read_parquet(parquet_path)
    df = df[df["year"] == year] if "year" in df.columns else df

    DEMO_VARS = [
        ("total_population", TargetCategory.POPULATION, True, None, None),
        ("population_under_18", TargetCategory.AGE_DISTRIBUTION, True, "age", None),
        ("population_18_64", TargetCategory.AGE_DISTRIBUTION, True, "age", None),
        ("population_65_plus", TargetCategory.AGE_DISTRIBUTION, True, "age", None),
        ("total_households", TargetCategory.HOUSEHOLD_STRUCTURE, True, None, None),
        ("married_households", TargetCategory.HOUSEHOLD_STRUCTURE, True, None, None),
    ]

    for _, row in df.iterrows():
        state_fips = str(row.get("state_fips", "")).zfill(2)
        state_code = row.get("state_code", STATE_FIPS.get(state_fips, ""))

        for var, category, is_count, rac_var, rac_statute in DEMO_VARS:
            if var in row and pd.notna(row[var]):
                targets.append(Target(
                    name=f"{var}_{state_code}",
                    category=category,
                    value=float(row[var]),
                    year=year,
                    source="Census Bureau ACS",
                    geography=state_code,
                    state_fips=state_fips,
                    is_count=is_count,
                    rac_variable=rac_var,
                    rac_statute=rac_statute,
                ))

    return targets


def load_state_income_targets(
    year: int = 2021,
    data_path: Optional[Path] = None,
) -> List[Target]:
    """
    Load state-level income distribution targets from cosilico-data-sources.

    Returns ~2000 targets (51 states × 40 AGI brackets × 1 variable).
    """
    targets = []

    if data_path is None:
        data_path = Path.home() / "CosilicoAI" / "cosilico-data-sources" / "data" / "targets"

    parquet_path = data_path / "state_income_distribution.parquet"
    if not parquet_path.exists():
        return targets

    df = pd.read_parquet(parquet_path)
    df = df[df["year"] == year] if "year" in df.columns else df

    for _, row in df.iterrows():
        state_fips = str(row.get("state_fips", "")).zfill(2)
        state_code = row.get("state_code", STATE_FIPS.get(state_fips, ""))
        bracket = row.get("agi_bracket", "unknown")
        lower = row.get("agi_bracket_min", -np.inf)
        upper = row.get("agi_bracket_max", np.inf)

        # Return count
        if "target_returns" in row and pd.notna(row["target_returns"]):
            targets.append(Target(
                name=f"returns_{bracket}_{state_code}",
                category=TargetCategory.AGI_DISTRIBUTION,
                value=float(row["target_returns"]),
                year=year,
                source="IRS SOI State Data",
                source_url="https://www.irs.gov/statistics/soi-tax-stats-historic-table-2",
                geography=state_code,
                state_fips=state_fips,
                agi_lower=float(lower) if pd.notna(lower) else -np.inf,
                agi_upper=float(upper) if pd.notna(upper) else np.inf,
                is_count=True,
                rac_variable="is_tax_filer",
                rac_statute="26/6012",
            ))

        # AGI amount
        if "target_agi" in row and pd.notna(row["target_agi"]):
            targets.append(Target(
                name=f"agi_{bracket}_{state_code}",
                category=TargetCategory.AGI_DISTRIBUTION,
                value=float(row["target_agi"]),
                year=year,
                source="IRS SOI State Data",
                geography=state_code,
                state_fips=state_fips,
                agi_lower=float(lower) if pd.notna(lower) else -np.inf,
                agi_upper=float(upper) if pd.notna(upper) else np.inf,
                is_count=False,
                rac_variable="adjusted_gross_income",
                rac_statute="26/62",
            ))

    return targets


def load_state_tax_credit_targets(
    year: int = 2021,
    data_path: Optional[Path] = None,
) -> List[Target]:
    """
    Load state-level tax credit targets from cosilico-data-sources.

    Returns ~200 targets (51 states × 4 credit variables).
    """
    targets = []

    if data_path is None:
        data_path = Path.home() / "CosilicoAI" / "cosilico-data-sources" / "data" / "targets"

    parquet_path = data_path / "state_tax_credits.parquet"
    if not parquet_path.exists():
        return targets

    df = pd.read_parquet(parquet_path)
    df = df[df["year"] == year] if "year" in df.columns else df

    CREDIT_VARS = [
        ("eitc_claims", TargetCategory.EITC, True, "earned_income_credit", "26/32"),
        ("eitc_amount", TargetCategory.EITC, False, "earned_income_credit", "26/32"),
        ("ctc_claims", TargetCategory.CTC, True, "child_tax_credit", "26/24"),
        ("ctc_amount", TargetCategory.CTC, False, "child_tax_credit", "26/24"),
    ]

    for _, row in df.iterrows():
        state_fips = str(row.get("state_fips", "")).zfill(2)
        state_code = row.get("state_code", STATE_FIPS.get(state_fips, ""))

        for var, category, is_count, rac_var, rac_statute in CREDIT_VARS:
            if var in row and pd.notna(row[var]):
                targets.append(Target(
                    name=f"{var}_{state_code}",
                    category=category,
                    value=float(row[var]),
                    year=year,
                    source="IRS SOI State EITC/CTC",
                    source_url="https://www.irs.gov/statistics/soi-tax-stats-eitc-claims-by-state",
                    geography=state_code,
                    state_fips=state_fips,
                    is_count=is_count,
                    rac_variable=rac_var,
                    rac_statute=rac_statute,
                ))

    return targets


def load_state_unemployment_targets(
    year: int = 2021,
    data_path: Optional[Path] = None,
) -> List[Target]:
    """
    Load state-level unemployment targets from cosilico-data-sources.

    Returns ~250 targets (51 states × 5 variables).
    """
    targets = []

    if data_path is None:
        data_path = Path.home() / "CosilicoAI" / "cosilico-data-sources" / "data" / "targets"

    parquet_path = data_path / "state_unemployment.parquet"
    if not parquet_path.exists():
        return targets

    df = pd.read_parquet(parquet_path)
    df = df[df["year"] == year] if "year" in df.columns else df

    UNEMP_VARS = [
        ("labor_force", TargetCategory.EMPLOYMENT, True, None, None),
        ("unemployed", TargetCategory.UNEMPLOYMENT, True, "unemployment_compensation", "26/85"),
        ("initial_claims", TargetCategory.UNEMPLOYMENT, True, "unemployment_compensation", "26/85"),
        ("continued_claims", TargetCategory.UNEMPLOYMENT, True, "unemployment_compensation", "26/85"),
    ]

    for _, row in df.iterrows():
        state_fips = str(row.get("state_fips", "")).zfill(2)
        state_code = row.get("state_code", STATE_FIPS.get(state_fips, ""))

        for var, category, is_count, rac_var, rac_statute in UNEMP_VARS:
            if var in row and pd.notna(row[var]):
                targets.append(Target(
                    name=f"{var}_{state_code}",
                    category=category,
                    value=float(row[var]),
                    year=year,
                    source="BLS Local Area Unemployment Statistics",
                    source_url="https://www.bls.gov/lau/",
                    geography=state_code,
                    state_fips=state_fips,
                    is_count=is_count,
                    rac_variable=rac_var,
                    rac_statute=rac_statute,
                ))

    return targets


def load_state_snap_targets(year: int = 2021) -> List[Target]:
    """
    Load state-level SNAP targets from USDA FNS.

    Returns ~150 targets (51 states × 3 SNAP variables).
    """
    targets = []

    # SNAP participation and benefits by state (USDA FNS FY2021)
    # Format: (households, participants, benefits in millions)
    STATE_SNAP = {
        "AL": (442_000, 775_000, 1_400),
        "AK": (44_000, 78_000, 190),
        "AZ": (425_000, 865_000, 1_600),
        "AR": (225_000, 395_000, 700),
        "CA": (2_950_000, 4_250_000, 9_500),
        "CO": (280_000, 490_000, 900),
        "CT": (225_000, 390_000, 750),
        "DE": (55_000, 95_000, 175),
        "DC": (70_000, 105_000, 220),
        "FL": (1_650_000, 3_100_000, 5_800),
        "GA": (830_000, 1_550_000, 2_850),
        "HI": (95_000, 170_000, 400),
        "ID": (95_000, 195_000, 330),
        "IL": (1_050_000, 1_850_000, 3_600),
        "IN": (450_000, 800_000, 1_450),
        "IA": (175_000, 330_000, 560),
        "KS": (135_000, 255_000, 430),
        "KY": (405_000, 700_000, 1_350),
        "LA": (475_000, 850_000, 1_700),
        "ME": (115_000, 185_000, 380),
        "MD": (380_000, 670_000, 1_280),
        "MA": (480_000, 820_000, 1_650),
        "MI": (750_000, 1_350_000, 2_600),
        "MN": (270_000, 490_000, 880),
        "MS": (305_000, 545_000, 1_000),
        "MO": (420_000, 755_000, 1_350),
        "MT": (55_000, 100_000, 175),
        "NE": (95_000, 175_000, 300),
        "NV": (235_000, 420_000, 800),
        "NH": (50_000, 80_000, 140),
        "NJ": (470_000, 810_000, 1_600),
        "NM": (235_000, 435_000, 850),
        "NY": (1_750_000, 2_900_000, 6_200),
        "NC": (760_000, 1_350_000, 2_500),
        "ND": (35_000, 60_000, 105),
        "OH": (850_000, 1_480_000, 2_800),
        "OK": (330_000, 590_000, 1_080),
        "OR": (370_000, 650_000, 1_300),
        "PA": (1_050_000, 1_750_000, 3_500),
        "RI": (85_000, 145_000, 300),
        "SC": (380_000, 680_000, 1_250),
        "SD": (50_000, 90_000, 155),
        "TN": (540_000, 940_000, 1_800),
        "TX": (2_150_000, 4_050_000, 7_400),
        "UT": (105_000, 210_000, 360),
        "VT": (40_000, 65_000, 130),
        "VA": (440_000, 775_000, 1_400),
        "WA": (525_000, 930_000, 1_900),
        "WV": (175_000, 285_000, 560),
        "WI": (350_000, 620_000, 1_150),
        "WY": (18_000, 30_000, 55),
    }

    for state_code, (households, participants, benefits_m) in STATE_SNAP.items():
        state_fips = {v: k for k, v in STATE_FIPS.items()}.get(state_code, "")

        targets.append(Target(
            name=f"snap_households_{state_code}",
            category=TargetCategory.SNAP,
            value=households,
            year=year,
            source="USDA FNS SNAP Data",
            source_url="https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap",
            geography=state_code,
            state_fips=state_fips,
            is_count=True,
            rac_variable="snap_benefit",
            rac_statute="7/2017",
        ))

        targets.append(Target(
            name=f"snap_participants_{state_code}",
            category=TargetCategory.SNAP,
            value=participants,
            year=year,
            source="USDA FNS SNAP Data",
            geography=state_code,
            state_fips=state_fips,
            is_count=True,
            rac_variable="snap_benefit",
            rac_statute="7/2017",
        ))

        targets.append(Target(
            name=f"snap_benefits_{state_code}",
            category=TargetCategory.SNAP,
            value=benefits_m * 1_000_000,
            year=year,
            source="USDA FNS SNAP Data",
            geography=state_code,
            state_fips=state_fips,
            is_count=False,
            rac_variable="snap_benefit",
            rac_statute="7/2017",
        ))

    return targets


def load_state_medicaid_targets(year: int = 2021) -> List[Target]:
    """
    Load state-level Medicaid enrollment targets from CMS.

    Returns ~100 targets (51 states × 2 Medicaid variables).
    """
    targets = []

    # Medicaid enrollment by state (CMS, 2021)
    STATE_MEDICAID = {
        "AL": 1_100_000, "AK": 240_000, "AZ": 2_400_000, "AR": 1_000_000,
        "CA": 14_000_000, "CO": 1_500_000, "CT": 1_000_000, "DE": 280_000,
        "DC": 300_000, "FL": 5_000_000, "GA": 2_200_000, "HI": 420_000,
        "ID": 450_000, "IL": 3_300_000, "IN": 1_800_000, "IA": 800_000,
        "KS": 450_000, "KY": 1_600_000, "LA": 1_900_000, "ME": 380_000,
        "MD": 1_500_000, "MA": 2_000_000, "MI": 2_800_000, "MN": 1_400_000,
        "MS": 780_000, "MO": 1_100_000, "MT": 300_000, "NE": 330_000,
        "NV": 850_000, "NH": 220_000, "NJ": 2_000_000, "NM": 900_000,
        "NY": 7_500_000, "NC": 2_500_000, "ND": 110_000, "OH": 3_300_000,
        "OK": 1_000_000, "OR": 1_400_000, "PA": 3_500_000, "RI": 340_000,
        "SC": 1_300_000, "SD": 135_000, "TN": 1_800_000, "TX": 5_500_000,
        "UT": 400_000, "VT": 200_000, "VA": 1_700_000, "WA": 2_200_000,
        "WV": 600_000, "WI": 1_400_000, "WY": 80_000,
    }

    for state_code, enrollment in STATE_MEDICAID.items():
        state_fips = {v: k for k, v in STATE_FIPS.items()}.get(state_code, "")

        targets.append(Target(
            name=f"medicaid_enrollment_{state_code}",
            category=TargetCategory.MEDICAID,
            value=enrollment,
            year=year,
            source="CMS Medicaid Enrollment",
            source_url="https://www.medicaid.gov/medicaid/program-information/medicaid-chip-enrollment-data",
            geography=state_code,
            state_fips=state_fips,
            is_count=True,
            rac_variable="medicaid_eligible",
            rac_statute="42/1396a",
        ))

    return targets


def load_state_ssi_targets(year: int = 2021) -> List[Target]:
    """
    Load state-level SSI recipients from SSA.

    Returns ~100 targets (51 states × 2 SSI variables).
    """
    targets = []

    # SSI recipients by state (SSA, 2021)
    STATE_SSI = {
        "AL": 175_000, "AK": 12_000, "AZ": 115_000, "AR": 85_000,
        "CA": 1_300_000, "CO": 75_000, "CT": 60_000, "DE": 18_000,
        "DC": 25_000, "FL": 500_000, "GA": 245_000, "HI": 28_000,
        "ID": 30_000, "IL": 250_000, "IN": 110_000, "IA": 50_000,
        "KS": 45_000, "KY": 175_000, "LA": 160_000, "ME": 35_000,
        "MD": 105_000, "MA": 175_000, "MI": 245_000, "MN": 85_000,
        "MS": 120_000, "MO": 125_000, "MT": 18_000, "NE": 28_000,
        "NV": 55_000, "NH": 18_000, "NJ": 165_000, "NM": 65_000,
        "NY": 700_000, "NC": 200_000, "ND": 10_000, "OH": 280_000,
        "OK": 90_000, "OR": 85_000, "PA": 350_000, "RI": 35_000,
        "SC": 105_000, "SD": 15_000, "TN": 165_000, "TX": 550_000,
        "UT": 30_000, "VT": 15_000, "VA": 150_000, "WA": 145_000,
        "WV": 75_000, "WI": 110_000, "WY": 8_000,
    }

    for state_code, recipients in STATE_SSI.items():
        state_fips = {v: k for k, v in STATE_FIPS.items()}.get(state_code, "")

        targets.append(Target(
            name=f"ssi_recipients_{state_code}",
            category=TargetCategory.SSI,
            value=recipients,
            year=year,
            source="SSA SSI Statistics",
            source_url="https://www.ssa.gov/policy/docs/statcomps/ssi_asr/",
            geography=state_code,
            state_fips=state_fips,
            is_count=True,
            rac_variable="ssi_benefit",
            rac_statute="42/1382",
        ))

    return targets


def load_state_tanf_targets(year: int = 2021) -> List[Target]:
    """
    Load state-level TANF caseload from HHS.

    Returns ~50 targets (51 states × 1 TANF variable).
    """
    targets = []

    # TANF families by state (HHS ACF, 2021)
    STATE_TANF = {
        "AL": 8_000, "AK": 3_500, "AZ": 16_000, "AR": 4_500,
        "CA": 340_000, "CO": 15_000, "CT": 16_000, "DE": 4_000,
        "DC": 5_000, "FL": 30_000, "GA": 15_000, "HI": 10_000,
        "ID": 2_000, "IL": 35_000, "IN": 10_000, "IA": 10_000,
        "KS": 6_000, "KY": 20_000, "LA": 6_000, "ME": 8_000,
        "MD": 25_000, "MA": 45_000, "MI": 20_000, "MN": 25_000,
        "MS": 4_000, "MO": 20_000, "MT": 3_000, "NE": 5_000,
        "NV": 10_000, "NH": 4_000, "NJ": 25_000, "NM": 15_000,
        "NY": 150_000, "NC": 18_000, "ND": 1_500, "OH": 40_000,
        "OK": 8_000, "OR": 25_000, "PA": 60_000, "RI": 8_000,
        "SC": 8_000, "SD": 2_500, "TN": 25_000, "TX": 22_000,
        "UT": 5_000, "VT": 4_000, "VA": 20_000, "WA": 45_000,
        "WV": 8_000, "WI": 20_000, "WY": 500,
    }

    for state_code, families in STATE_TANF.items():
        state_fips = {v: k for k, v in STATE_FIPS.items()}.get(state_code, "")

        targets.append(Target(
            name=f"tanf_families_{state_code}",
            category=TargetCategory.TANF,
            value=families,
            year=year,
            source="HHS ACF TANF Data",
            source_url="https://www.acf.hhs.gov/ofa/programs/tanf/data-reports",
            geography=state_code,
            state_fips=state_fips,
            is_count=True,
            rac_variable="tanf_benefit",
            rac_statute="42/601",
        ))

    return targets


def load_state_housing_targets(year: int = 2021) -> List[Target]:
    """
    Load state-level housing assistance from HUD.

    Returns ~100 targets (51 states × 2 housing variables).
    """
    targets = []

    # Housing vouchers by state (HUD, 2021)
    STATE_VOUCHERS = {
        "AL": 35_000, "AK": 5_000, "AZ": 45_000, "AR": 20_000,
        "CA": 350_000, "CO": 45_000, "CT": 45_000, "DE": 8_000,
        "DC": 12_000, "FL": 130_000, "GA": 75_000, "HI": 12_000,
        "ID": 10_000, "IL": 115_000, "IN": 50_000, "IA": 20_000,
        "KS": 20_000, "KY": 40_000, "LA": 55_000, "ME": 15_000,
        "MD": 55_000, "MA": 95_000, "MI": 75_000, "MN": 45_000,
        "MS": 25_000, "MO": 50_000, "MT": 8_000, "NE": 15_000,
        "NV": 25_000, "NH": 10_000, "NJ": 95_000, "NM": 18_000,
        "NY": 300_000, "NC": 65_000, "ND": 5_000, "OH": 105_000,
        "OK": 30_000, "OR": 40_000, "PA": 130_000, "RI": 15_000,
        "SC": 35_000, "SD": 6_000, "TN": 50_000, "TX": 175_000,
        "UT": 12_000, "VT": 8_000, "VA": 60_000, "WA": 65_000,
        "WV": 18_000, "WI": 40_000, "WY": 3_000,
    }

    # Public housing by state (HUD, 2021)
    STATE_PUBLIC_HOUSING = {
        "AL": 25_000, "AK": 3_000, "AZ": 10_000, "AR": 8_000,
        "CA": 55_000, "CO": 10_000, "CT": 18_000, "DE": 4_000,
        "DC": 8_000, "FL": 35_000, "GA": 35_000, "HI": 8_000,
        "ID": 2_000, "IL": 45_000, "IN": 15_000, "IA": 5_000,
        "KS": 5_000, "KY": 18_000, "LA": 22_000, "ME": 5_000,
        "MD": 18_000, "MA": 45_000, "MI": 25_000, "MN": 18_000,
        "MS": 15_000, "MO": 15_000, "MT": 2_000, "NE": 5_000,
        "NV": 5_000, "NH": 5_000, "NJ": 45_000, "NM": 5_000,
        "NY": 175_000, "NC": 25_000, "ND": 2_000, "OH": 40_000,
        "OK": 10_000, "OR": 8_000, "PA": 55_000, "RI": 8_000,
        "SC": 15_000, "SD": 2_000, "TN": 25_000, "TX": 55_000,
        "UT": 3_000, "VT": 3_000, "VA": 25_000, "WA": 15_000,
        "WV": 8_000, "WI": 12_000, "WY": 1_000,
    }

    for state_code, vouchers in STATE_VOUCHERS.items():
        state_fips = {v: k for k, v in STATE_FIPS.items()}.get(state_code, "")

        targets.append(Target(
            name=f"housing_vouchers_{state_code}",
            category=TargetCategory.HOUSING,
            value=vouchers,
            year=year,
            source="HUD Picture of Subsidized Households",
            source_url="https://www.huduser.gov/portal/datasets/assthsg.html",
            geography=state_code,
            state_fips=state_fips,
            is_count=True,
            rac_variable="housing_subsidy",
            rac_statute="42/1437f",
        ))

    for state_code, public_housing in STATE_PUBLIC_HOUSING.items():
        state_fips = {v: k for k, v in STATE_FIPS.items()}.get(state_code, "")

        targets.append(Target(
            name=f"public_housing_{state_code}",
            category=TargetCategory.HOUSING,
            value=public_housing,
            year=year,
            source="HUD Picture of Subsidized Households",
            geography=state_code,
            state_fips=state_fips,
            is_count=True,
            rac_variable="housing_subsidy",
            rac_statute="42/1437",
        ))

    return targets


def load_all_state_targets(year: int = 2021, data_path: Optional[Path] = None) -> List[Target]:
    """Load all state-level targets."""
    targets = []
    # From cosilico-data-sources parquet files
    targets.extend(load_state_demographics_targets(year, data_path))
    targets.extend(load_state_income_targets(year, data_path))
    targets.extend(load_state_tax_credit_targets(year, data_path))
    targets.extend(load_state_unemployment_targets(year, data_path))
    # Hardcoded state benefit targets
    targets.extend(load_state_snap_targets(year))
    targets.extend(load_state_medicaid_targets(year))
    targets.extend(load_state_ssi_targets(year))
    targets.extend(load_state_tanf_targets(year))
    targets.extend(load_state_housing_targets(year))
    return targets


def load_all_targets(year: int = 2021, include_states: bool = True) -> List[Target]:
    """
    Load all available targets for a year.

    Args:
        year: Target year
        include_states: If True, include state-level targets (~2500 additional)

    Returns:
        List of Target objects (~500 national + ~2500 state if included)
    """
    targets = []

    # IRS SOI income/deduction targets
    targets.extend(load_soi_targets(year))
    targets.extend(load_filing_status_targets(year))

    # Tax credits
    targets.extend(load_eitc_targets(year))
    targets.extend(load_ctc_targets(year))
    targets.extend(load_aca_targets(year))

    # Benefits
    targets.extend(load_snap_targets(year))
    targets.extend(load_medicaid_targets(year))
    targets.extend(load_ssi_targets(year))
    targets.extend(load_tanf_targets(year))
    targets.extend(load_housing_targets(year))
    targets.extend(load_wic_targets(year))
    targets.extend(load_other_benefit_targets(year))

    # Social Security / Medicare
    targets.extend(load_social_security_targets(year))

    # Demographics
    targets.extend(load_demographics_targets(year))
    targets.extend(load_race_ethnicity_targets(year))
    targets.extend(load_education_targets(year))
    targets.extend(load_disability_targets(year))
    targets.extend(load_household_composition_targets(year))
    targets.extend(load_poverty_targets(year))

    # Wealth / Assets
    targets.extend(load_wealth_targets(year))

    # Employment
    targets.extend(load_employment_industry_targets(year))

    # State-level targets
    if include_states:
        targets.extend(load_all_state_targets(year))

    return targets
