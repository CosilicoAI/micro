"""Registry of all calibration targets for PE parity."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import pandas as pd
import numpy as np


class TargetCategory(Enum):
    """Categories of calibration targets."""
    GEOGRAPHY = "geography"
    INCOME = "income"
    BENEFITS = "benefits"
    DEMOGRAPHICS = "demographics"
    HEALTH = "health"
    TAX = "tax"


class TargetLevel(Enum):
    """Geographic level of target."""
    NATIONAL = "national"
    STATE = "state"
    CD = "cd"
    COUNTY = "county"
    TRACT = "tract"


@dataclass
class TargetSpec:
    """Specification for a calibration target."""
    name: str
    category: TargetCategory
    level: TargetLevel
    value: float
    year: int = 2024

    # How to compute from microdata
    column: Optional[str] = None  # Column to aggregate
    filter_column: Optional[str] = None  # Column to filter on
    filter_value: Optional[Any] = None  # Value to filter for
    aggregation: str = "sum"  # sum, count, mean

    # Metadata
    source: str = ""
    unit: str = ""
    description: str = ""

    # Data availability
    available_in_cps: bool = True
    requires_imputation: bool = False
    notes: str = ""


@dataclass
class TargetGroup:
    """A group of related targets."""
    name: str
    category: TargetCategory
    targets: List[TargetSpec] = field(default_factory=list)

    def add(self, target: TargetSpec):
        self.targets.append(target)
        return self

    def __len__(self):
        return len(self.targets)


class TargetRegistry:
    """Central registry of all calibration targets."""

    def __init__(self):
        self.groups: Dict[str, TargetGroup] = {}
        self._build_registry()

    def _build_registry(self):
        """Build the full target registry."""
        self._add_geography_targets()
        self._add_income_targets()
        self._add_benefit_targets()
        self._add_health_targets()
        self._add_tax_targets()
        self._add_demographic_targets()

    def _add_geography_targets(self):
        """Add geographic population targets."""
        # State populations (Census 2020)
        state_pops = TargetGroup("state_population", TargetCategory.GEOGRAPHY)

        CENSUS_2020 = {
            '01': 5024279, '02': 733391, '04': 7151502, '05': 3011524,
            '06': 39538223, '08': 5773714, '09': 3605944, '10': 989948,
            '11': 689545, '12': 21538187, '13': 10711908, '15': 1455271,
            '16': 1839106, '17': 12812508, '18': 6785528, '19': 3190369,
            '20': 2937880, '21': 4505836, '22': 4657757, '23': 1362359,
            '24': 6177224, '25': 7029917, '26': 10077331, '27': 5706494,
            '28': 2961279, '29': 6154913, '30': 1084225, '31': 1961504,
            '32': 3104614, '33': 1377529, '34': 9288994, '35': 2117522,
            '36': 20201249, '37': 10439388, '38': 779094, '39': 11799448,
            '40': 3959353, '41': 4237256, '42': 13002700, '44': 1097379,
            '45': 5118425, '46': 886667, '47': 6910840, '48': 29145505,
            '49': 3271616, '50': 643077, '51': 8631393, '53': 7705281,
            '54': 1793716, '55': 5893718, '56': 576851,
        }

        for fips, pop in CENSUS_2020.items():
            state_pops.add(TargetSpec(
                name=f"population_{fips}",
                category=TargetCategory.GEOGRAPHY,
                level=TargetLevel.STATE,
                value=pop,
                column=None,  # Use weight directly (count weighted people)
                filter_column="state_fips",
                filter_value=fips,
                aggregation="count",
                source="Census 2020",
                unit="persons",
            ))

        self.groups["state_population"] = state_pops

        # Congressional Districts (placeholder - 436 CDs)
        cd_pops = TargetGroup("cd_population", TargetCategory.GEOGRAPHY)
        # Would load from data/district_targets.parquet
        self.groups["cd_population"] = cd_pops

        # State Legislative Districts (placeholder)
        sldu_pops = TargetGroup("sldu_population", TargetCategory.GEOGRAPHY)
        sldl_pops = TargetGroup("sldl_population", TargetCategory.GEOGRAPHY)
        self.groups["sldu_population"] = sldu_pops
        self.groups["sldl_population"] = sldl_pops

    def _add_income_targets(self):
        """Add IRS SOI income targets."""
        income = TargetGroup("irs_soi_income", TargetCategory.INCOME)

        # IRS SOI income totals (2021 data)
        SOI_INCOME = {
            "employment_income": (9_022_352_941_000, "employment_income", True),
            "self_employment_income": (436_400_000_000, "self_employment_income", True),
            "social_security": (774_000_000_000, "social_security", True),
            "taxable_pension_income": (827_600_000_000, "taxable_pension_income", True),
            "tax_exempt_pension_income": (580_400_000_000, "tax_exempt_pension_income", True),
            "unemployment_compensation": (208_000_000_000, "unemployment_compensation", True),
            "dividend_income": (260_200_000_000, "dividend_income", False),  # Underreported
            "interest_income": (127_400_000_000, "interest_income", False),
            "rental_income": (46_000_000_000, "rental_income", True),
            "long_term_capital_gains": (1_137_000_000_000, "long_term_capital_gains", False),
            "short_term_capital_gains": (-72_000_000_000, "short_term_capital_gains", False),
            "partnership_s_corp_income": (976_000_000_000, "partnership_s_corp_income", False),
            "farm_income": (-26_141_944_000, "farm_income", False),
            "alimony_income": (8_500_000_000, "alimony_income", True),
        }

        for name, (value, column, in_cps) in SOI_INCOME.items():
            income.add(TargetSpec(
                name=name,
                category=TargetCategory.INCOME,
                level=TargetLevel.NATIONAL,
                value=value,
                column=column,
                aggregation="sum",
                source="IRS SOI",
                unit="USD",
                available_in_cps=in_cps,
                requires_imputation=not in_cps,
                notes="" if in_cps else "Underreported in CPS, requires imputation",
            ))

        self.groups["irs_soi_income"] = income

    def _add_benefit_targets(self):
        """Add benefit program targets."""
        benefits = TargetGroup("benefit_programs", TargetCategory.BENEFITS)

        # National benefit aggregates
        BENEFITS = {
            # (value, column, unit, source)
            "snap_spending": (103_100_000_000, "snap", "USD", "CBO"),
            "snap_participation": (41_209_000, "snap", "persons", "USDA"),
            "ssi_spending": (78_500_000_000, "ssi", "USD", "CBO"),
            "ssi_participation": (7_400_000, "ssi", "persons", "SSA"),
            "social_security_spending": (2_623_800_000_000, "social_security", "USD", "CBO"),
            "social_security_participation": (66_000_000, "social_security", "persons", "SSA"),
            "eitc_spending": (72_700_000_000, "eitc", "USD", "Treasury"),
            "unemployment_spending": (59_100_000_000, "unemployment_compensation", "USD", "CBO"),
        }

        for name, (value, column, unit, source) in BENEFITS.items():
            is_count = unit == "persons"
            benefits.add(TargetSpec(
                name=name,
                category=TargetCategory.BENEFITS,
                level=TargetLevel.NATIONAL,
                value=value,
                column=column,
                aggregation="count" if is_count else "sum",
                source=source,
                unit=unit,
                available_in_cps=True,
            ))

        self.groups["benefit_programs"] = benefits

    def _add_health_targets(self):
        """Add health insurance/Medicaid targets."""
        health = TargetGroup("health_insurance", TargetCategory.HEALTH)

        # Medicaid enrollment by state (placeholder structure)
        # Would be populated from PE calibration files
        MEDICAID_CATS = ["child", "aged", "disabled", "expansion_adults", "non_expansion_adults"]

        for cat in MEDICAID_CATS:
            health.add(TargetSpec(
                name=f"medicaid_{cat}_national",
                category=TargetCategory.HEALTH,
                level=TargetLevel.NATIONAL,
                value=0,  # Would load from PE
                column="medicaid",
                aggregation="count",
                source="HHS/CMS",
                unit="persons",
                available_in_cps=False,
                requires_imputation=True,
                notes="Requires eligibility modeling",
            ))

        # CHIP targets
        health.add(TargetSpec(
            name="chip_enrollment_national",
            category=TargetCategory.HEALTH,
            level=TargetLevel.NATIONAL,
            value=0,
            column="chip",
            aggregation="count",
            source="CMS",
            unit="persons",
            available_in_cps=False,
            requires_imputation=True,
        ))

        # ACA marketplace
        health.add(TargetSpec(
            name="aca_enrollment_national",
            category=TargetCategory.HEALTH,
            level=TargetLevel.NATIONAL,
            value=0,
            column="aca_enrolled",
            aggregation="count",
            source="CMS",
            unit="persons",
            available_in_cps=False,
            requires_imputation=True,
        ))

        self.groups["health_insurance"] = health

    def _add_tax_targets(self):
        """Add tax-related targets."""
        tax = TargetGroup("tax_aggregates", TargetCategory.TAX)

        TAX_TARGETS = {
            "income_tax_total": (4_412_800_000_000, "income_tax", "USD"),
            "payroll_tax_total": (2_605_200_000_000, "payroll_tax", "USD"),
            "eitc_claims": (25_000_000, "eitc", "returns"),  # Approximate
            "ctc_claims": (35_000_000, "ctc", "returns"),  # Approximate
        }

        for name, (value, column, unit) in TAX_TARGETS.items():
            tax.add(TargetSpec(
                name=name,
                category=TargetCategory.TAX,
                level=TargetLevel.NATIONAL,
                value=value,
                column=column,
                aggregation="count" if "claims" in name else "sum",
                source="CBO/IRS",
                unit=unit,
                available_in_cps=False,
                requires_imputation=True,
                notes="Requires tax calculation",
            ))

        self.groups["tax_aggregates"] = tax

    def _add_demographic_targets(self):
        """Add demographic distribution targets."""
        demo = TargetGroup("demographics", TargetCategory.DEMOGRAPHICS)

        # Age distribution by state (placeholder)
        # Would load from data/targets.parquet

        # Filing status distribution
        FILING_STATUS = {
            "single": 75_000_000,
            "married_joint": 55_000_000,
            "married_separate": 3_000_000,
            "head_of_household": 22_000_000,
        }

        for status, count in FILING_STATUS.items():
            demo.add(TargetSpec(
                name=f"filing_status_{status}",
                category=TargetCategory.DEMOGRAPHICS,
                level=TargetLevel.NATIONAL,
                value=count,
                column="filing_status",
                filter_value=status,
                aggregation="count",
                source="IRS SOI",
                unit="returns",
                available_in_cps=False,
                requires_imputation=True,
                notes="Requires tax unit modeling",
            ))

        self.groups["demographics"] = demo

    def get_group(self, name: str) -> Optional[TargetGroup]:
        """Get a target group by name."""
        return self.groups.get(name)

    def get_all_targets(self) -> List[TargetSpec]:
        """Get all targets as a flat list."""
        all_targets = []
        for group in self.groups.values():
            all_targets.extend(group.targets)
        return all_targets

    def get_available_targets(self) -> List[TargetSpec]:
        """Get targets that are available in CPS data."""
        return [t for t in self.get_all_targets() if t.available_in_cps]

    def get_targets_by_category(self, category: TargetCategory) -> List[TargetSpec]:
        """Get targets by category."""
        return [t for t in self.get_all_targets() if t.category == category]

    def get_targets_by_level(self, level: TargetLevel) -> List[TargetSpec]:
        """Get targets by geographic level."""
        return [t for t in self.get_all_targets() if t.level == level]

    def summary(self) -> Dict:
        """Get summary of registry contents."""
        all_targets = self.get_all_targets()
        available = self.get_available_targets()

        by_category = {}
        for cat in TargetCategory:
            by_category[cat.value] = len(self.get_targets_by_category(cat))

        by_level = {}
        for level in TargetLevel:
            by_level[level.value] = len(self.get_targets_by_level(level))

        return {
            "total_targets": len(all_targets),
            "available_in_cps": len(available),
            "requires_imputation": len(all_targets) - len(available),
            "by_category": by_category,
            "by_level": by_level,
            "groups": {name: len(group) for name, group in self.groups.items()},
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert registry to DataFrame."""
        records = []
        for target in self.get_all_targets():
            records.append({
                "name": target.name,
                "category": target.category.value,
                "level": target.level.value,
                "value": target.value,
                "column": target.column,
                "aggregation": target.aggregation,
                "source": target.source,
                "unit": target.unit,
                "available_in_cps": target.available_in_cps,
                "requires_imputation": target.requires_imputation,
                "notes": target.notes,
            })
        return pd.DataFrame(records)


def get_registry() -> TargetRegistry:
    """Get the global target registry."""
    return TargetRegistry()


def print_registry_summary():
    """Print a summary of available targets."""
    registry = get_registry()
    summary = registry.summary()

    print("=" * 70)
    print("MICROPLEX TARGET REGISTRY")
    print("=" * 70)
    print(f"\nTotal targets: {summary['total_targets']}")
    print(f"Available in CPS: {summary['available_in_cps']}")
    print(f"Requires imputation: {summary['requires_imputation']}")

    print("\nBy category:")
    for cat, count in summary['by_category'].items():
        print(f"  {cat}: {count}")

    print("\nBy level:")
    for level, count in summary['by_level'].items():
        print(f"  {level}: {count}")

    print("\nBy group:")
    for name, count in summary['groups'].items():
        print(f"  {name}: {count}")
