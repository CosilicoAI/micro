"""
Target Loaders

Load calibration targets from various sources:
- IRS Statistics of Income (SOI)
- Census ACS/CPS
- Admin data (SNAP, Medicaid)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional

from microplex.targets.database import Target, TargetCategory
from microplex.targets.rac_mapping import RAC_VARIABLE_MAP


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

    # Age distribution
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

    return targets


def load_all_targets(year: int = 2021) -> List[Target]:
    """Load all available targets for a year."""
    targets = []
    # IRS SOI income/deduction targets
    targets.extend(load_soi_targets(year))
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
    # Demographics
    targets.extend(load_demographics_targets(year))
    return targets
