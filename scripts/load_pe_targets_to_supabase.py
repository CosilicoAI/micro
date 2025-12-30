"""
Load all PolicyEngine calibration targets into Supabase.

Sources:
- IRS SOI: Income distribution by AGI bracket
- Census: State populations, age distribution
- CBO: Benefit program spending
- HHS: Medicaid/CHIP enrollment
- CMS: ACA enrollment and spending
- USDA: SNAP participation and costs
"""

import os
import pandas as pd
import requests
from io import StringIO
from typing import Optional, List, Dict, Any

# Supabase connection - use Cosilico DB
SUPABASE_URL = "https://nsupqhfchdtqclomlrgs.supabase.co"
# Use service_role key for writes
SUPABASE_KEY = os.environ.get(
    "COSILICO_SUPABASE_SERVICE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5zdXBxaGZjaGR0cWNsb21scmdzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NjkzMTEwOCwiZXhwIjoyMDgyNTA3MTA4fQ.IZX2C6dM6CCuxzBeg3zoZSA31p_jy9XLjdxjaE126BU"
)


class SupabaseClient:
    """Simple REST client for Supabase microplex schema."""

    def __init__(self, url: str, key: str, schema: str = "microplex"):
        self.base_url = f"{url}/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept-Profile": schema,
            "Content-Profile": schema,
            "Prefer": "return=representation"
        }

    def select(self, table: str, columns: str = "*", filters: Dict = None, limit: int = None) -> List[Dict]:
        """Select rows from a table."""
        url = f"{self.base_url}/{table}?select={columns}"
        if filters:
            for k, v in filters.items():
                url += f"&{k}=eq.{v}"
        if limit:
            url += f"&limit={limit}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def insert(self, table: str, data: Dict) -> Dict:
        """Insert a row, returning the inserted data."""
        url = f"{self.base_url}/{table}"
        resp = requests.post(url, headers=self.headers, json=data)
        resp.raise_for_status()
        return resp.json()[0] if resp.json() else {}

    def upsert(self, table: str, data: Dict, on_conflict: str = None) -> Dict:
        """Upsert a row."""
        url = f"{self.base_url}/{table}"
        headers = {**self.headers, "Prefer": "resolution=merge-duplicates,return=representation"}
        if on_conflict:
            url += f"?on_conflict={on_conflict}"
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()[0] if resp.json() else {}

    def update(self, table: str, data: Dict, filters: Dict) -> List[Dict]:
        """Update rows matching filters."""
        url = f"{self.base_url}/{table}"
        for k, v in filters.items():
            url += f"?{k}=eq.{v}"
        resp = requests.patch(url, headers=self.headers, json=data)
        resp.raise_for_status()
        return resp.json()


# Initialize client
supabase = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)

# PE data URLs
PE_BASE = "https://raw.githubusercontent.com/PolicyEngine/policyengine-us-data/main/policyengine_us_data/storage/calibration_targets"

# State FIPS to name mapping
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
    "56": "WY"
}

STATE_NAME_TO_FIPS = {v: k for k, v in STATE_FIPS.items()}


def fetch_csv(filename: str) -> pd.DataFrame:
    """Fetch CSV from PE repo."""
    url = f"{PE_BASE}/{filename}"
    print(f"Fetching {url}")
    response = requests.get(url)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def get_or_create_source(jurisdiction: str, institution: str, dataset: str, name: str, url: str = None):
    """Get existing source or create new one."""
    result = supabase.select("sources", "id", {
        "jurisdiction": jurisdiction,
        "institution": institution,
        "dataset": dataset
    })

    if result:
        return result[0]["id"]

    data = {
        "jurisdiction": jurisdiction,
        "institution": institution,
        "dataset": dataset,
        "name": name,
    }
    if url:
        data["url"] = url

    inserted = supabase.insert("sources", data)
    return inserted["id"]


def get_or_create_stratum(name: str, jurisdiction: str, constraints: list = None):
    """Get existing stratum or create new one."""
    result = supabase.select("strata", "id", {
        "name": name,
        "jurisdiction": jurisdiction
    })

    if result:
        return result[0]["id"]

    inserted = supabase.insert("strata", {
        "name": name,
        "jurisdiction": jurisdiction,
        "description": name
    })
    stratum_id = inserted["id"]

    # Add constraints if provided
    if constraints:
        for c in constraints:
            supabase.insert("stratum_constraints", {
                "stratum_id": stratum_id,
                "variable": c["variable"],
                "operator": c["operator"],
                "value": c["value"]
            })

    return stratum_id


def insert_target(source_id: str, stratum_id: str, variable: str, value: float,
                  target_type: str = "amount", period: int = 2024, notes: str = None):
    """Insert a calibration target."""
    # Convert numpy/pandas types to native Python
    if hasattr(value, 'item'):
        value = value.item()
    value = float(value)

    # Check if exists - need to filter on multiple fields
    url = f"{supabase.base_url}/targets?select=id&source_id=eq.{source_id}&stratum_id=eq.{stratum_id}&variable=eq.{variable}&period=eq.{period}"
    resp = requests.get(url, headers=supabase.headers)
    result = resp.json() if resp.status_code == 200 else []

    if result:
        # Update existing
        supabase.update("targets", {"value": value, "notes": notes}, {"id": result[0]["id"]})
        return result[0]["id"]

    # Insert new
    data = {
        "source_id": source_id,
        "stratum_id": stratum_id,
        "variable": variable,
        "value": value,
        "target_type": target_type,
        "period": period
    }
    if notes:
        data["notes"] = notes

    inserted = supabase.insert("targets", data)
    return inserted.get("id")


def load_medicaid_targets():
    """Load Medicaid enrollment by state."""
    print("\n=== Loading Medicaid Targets ===")
    df = fetch_csv("medicaid_enrollment_2024.csv")

    source_id = get_or_create_source(
        "us", "hhs", "medicaid",
        "HHS Medicaid Enrollment Data",
        "https://www.medicaid.gov/medicaid/program-information/medicaid-and-chip-enrollment-data/index.html"
    )

    # National total
    national_stratum = get_or_create_stratum("US total population", "us")
    total_enrollment = df["enrollment"].sum()
    insert_target(source_id, national_stratum, "medicaid_enrollment", total_enrollment, "count", 2024)
    print(f"  National: {total_enrollment:,.0f} enrolled")

    # By state
    for _, row in df.iterrows():
        state = row["state"]
        fips = STATE_NAME_TO_FIPS.get(state)
        if not fips:
            continue

        stratum = get_or_create_stratum(
            f"State {state} population", f"us-{state.lower()}",
            [{"variable": "state_fips", "operator": "==", "value": fips}]
        )
        insert_target(source_id, stratum, "medicaid_enrollment", row["enrollment"], "count", 2024)

    print(f"  Loaded {len(df)} state targets")


def load_snap_targets():
    """Load SNAP participation and costs by state."""
    print("\n=== Loading SNAP Targets ===")
    df = fetch_csv("snap_state.csv")

    source_id = get_or_create_source(
        "us", "usda", "snap",
        "USDA SNAP State Activity Report",
        "https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap"
    )

    # Convert GEO_ID to state FIPS
    df["state_fips"] = df["GEO_ID"].str.extract(r"(\d{2})$")

    # National totals
    national_stratum = get_or_create_stratum("US total population", "us")
    total_hh = df["Households"].sum()
    total_cost = df["Cost"].sum()
    insert_target(source_id, national_stratum, "snap_households", total_hh, "count", 2024)
    insert_target(source_id, national_stratum, "snap_spending", total_cost, "amount", 2024)
    print(f"  National: {total_hh:,.0f} households, ${total_cost/1e9:.1f}B spending")

    # By state
    for _, row in df.iterrows():
        fips = row["state_fips"]
        state = STATE_FIPS.get(fips)
        if not state:
            continue

        stratum = get_or_create_stratum(
            f"State {state} population", f"us-{state.lower()}",
            [{"variable": "state_fips", "operator": "==", "value": fips}]
        )
        insert_target(source_id, stratum, "snap_households", row["Households"], "count", 2024)
        insert_target(source_id, stratum, "snap_spending", row["Cost"], "amount", 2024)

    print(f"  Loaded {len(df) * 2} state targets")


def load_aca_targets():
    """Load ACA marketplace enrollment and spending by state."""
    print("\n=== Loading ACA Targets ===")
    df = fetch_csv("aca_spending_and_enrollment_2024.csv")

    source_id = get_or_create_source(
        "us", "cms", "aca_marketplace",
        "CMS ACA Marketplace Enrollment",
        "https://www.cms.gov/newsroom/fact-sheets/marketplace-2024-open-enrollment-period-report-final-national-snapshot"
    )

    # National totals
    national_stratum = get_or_create_stratum("US total population", "us")
    total_enrollment = df["enrollment"].sum()
    total_spending = df["spending"].sum()
    insert_target(source_id, national_stratum, "aca_enrollment", total_enrollment, "count", 2024)
    insert_target(source_id, national_stratum, "aca_ptc_spending", total_spending, "amount", 2024)
    print(f"  National: {total_enrollment:,.0f} enrolled, ${total_spending/1e9:.1f}B PTC")

    # By state
    for _, row in df.iterrows():
        state = row["state"]
        fips = STATE_NAME_TO_FIPS.get(state)
        if not fips:
            continue

        stratum = get_or_create_stratum(
            f"State {state} population", f"us-{state.lower()}",
            [{"variable": "state_fips", "operator": "==", "value": fips}]
        )
        insert_target(source_id, stratum, "aca_enrollment", row["enrollment"], "count", 2024)
        insert_target(source_id, stratum, "aca_ptc_spending", row["spending"], "amount", 2024)

    print(f"  Loaded {len(df) * 2} state targets")


def load_population_targets():
    """Load Census population by state."""
    print("\n=== Loading Population Targets ===")
    df = fetch_csv("population_by_state.csv")

    source_id = get_or_create_source(
        "us", "census", "population_projections",
        "Census Bureau Population Projections",
        "https://www.census.gov/programs-surveys/popproj.html"
    )

    # National totals
    national_stratum = get_or_create_stratum("US total population", "us")
    total_pop = df["population"].sum()
    total_under5 = df["population_under_5"].sum()
    insert_target(source_id, national_stratum, "total_population", total_pop, "count", 2024)
    insert_target(source_id, national_stratum, "population_under_5", total_under5, "count", 2024)
    print(f"  National: {total_pop:,.0f} total, {total_under5:,.0f} under 5")

    # By state
    for _, row in df.iterrows():
        state = row["state"]
        fips = STATE_NAME_TO_FIPS.get(state)
        if not fips:
            continue

        stratum = get_or_create_stratum(
            f"State {state} population", f"us-{state.lower()}",
            [{"variable": "state_fips", "operator": "==", "value": fips}]
        )
        insert_target(source_id, stratum, "total_population", row["population"], "count", 2024)
        insert_target(source_id, stratum, "population_under_5", row["population_under_5"], "count", 2024)

    print(f"  Loaded {len(df) * 2} state targets")


def load_irs_income_targets():
    """Load IRS SOI income targets (national level)."""
    print("\n=== Loading IRS Income Targets ===")

    source_id = get_or_create_source(
        "us", "irs", "soi",
        "IRS Statistics of Income",
        "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics"
    )

    national_stratum = get_or_create_stratum("US total population", "us")

    # From PE's loss.py and our earlier analysis
    targets = {
        "employment_income": 9_022_400_000_000,
        "self_employment_income": 436_400_000_000,
        "taxable_pension_income": 827_600_000_000,
        "tax_exempt_pension_income": 580_400_000_000,
        "social_security": 774_000_000_000,
        "unemployment_compensation": 208_000_000_000,
        "dividend_income": 260_200_000_000,
        "interest_income": 127_400_000_000,
        "rental_income": 46_000_000_000,
        "long_term_capital_gains": 1_137_000_000_000,
        "short_term_capital_gains": -72_000_000_000,
        "partnership_s_corp_income": 976_000_000_000,
        "farm_income": -26_100_000_000,
        "alimony_income": 8_500_000_000,
    }

    for name, value in targets.items():
        insert_target(source_id, national_stratum, name, value, "amount", 2024,
                     notes="IRS SOI aggregate")
        print(f"  {name}: ${value/1e9:.1f}B")

    print(f"  Loaded {len(targets)} income targets")


def load_benefit_spending_targets():
    """Load CBO/Treasury benefit spending targets."""
    print("\n=== Loading Benefit Spending Targets ===")

    source_id = get_or_create_source(
        "us", "cbo", "budget_projections",
        "CBO Budget Projections",
        "https://www.cbo.gov/data/budget-economic-data"
    )

    national_stratum = get_or_create_stratum("US total population", "us")

    # From PE's loss.py
    targets = {
        "snap_spending": 103_100_000_000,
        "ssi_spending": 78_500_000_000,
        "social_security_spending": 2_623_800_000_000,
        "eitc_spending": 72_700_000_000,
        "unemployment_spending": 59_100_000_000,
        "medicaid_spending": 900_000_000_000,
        "aca_ptc_spending": 98_000_000_000,
    }

    for name, value in targets.items():
        insert_target(source_id, national_stratum, name, value, "amount", 2024,
                     notes="CBO/Treasury aggregate")
        print(f"  {name}: ${value/1e9:.1f}B")

    print(f"  Loaded {len(targets)} benefit targets")


def load_healthcare_targets():
    """Load healthcare coverage targets."""
    print("\n=== Loading Healthcare Targets ===")

    source_id = get_or_create_source(
        "us", "hhs", "healthcare_coverage",
        "HHS Healthcare Coverage Estimates",
        "https://www.hhs.gov/"
    )

    national_stratum = get_or_create_stratum("US total population", "us")

    # From PE's loss.py
    targets = {
        "health_insurance_premiums": 385_000_000_000,
        "other_medical_expenses": 278_000_000_000,
        "medicare_part_b_premiums": 112_000_000_000,
        "over_the_counter_health_expenses": 72_000_000_000,
        "medicaid_enrollment": 72_429_055,
        "aca_enrollment": 19_743_689,
    }

    for name, value in targets.items():
        target_type = "count" if "enrollment" in name else "amount"
        insert_target(source_id, national_stratum, name, value, target_type, 2024,
                     notes="HHS/CBO estimate")
        if target_type == "count":
            print(f"  {name}: {value:,.0f}")
        else:
            print(f"  {name}: ${value/1e9:.1f}B")

    print(f"  Loaded {len(targets)} healthcare targets")


def load_tax_targets():
    """Load JCT tax expenditure targets."""
    print("\n=== Loading Tax Expenditure Targets ===")

    source_id = get_or_create_source(
        "us", "jct", "tax_expenditures",
        "JCT Tax Expenditure Estimates",
        "https://www.jct.gov/publications/tax-expenditure-estimates/"
    )

    national_stratum = get_or_create_stratum("US total population", "us")

    # From PE's loss.py (JCT 2024 report)
    targets = {
        "salt_deduction": 21_247_000_000,
        "medical_expense_deduction": 11_400_000_000,
        "charitable_deduction": 65_301_000_000,
        "interest_deduction": 24_800_000_000,
        "qbi_deduction": 63_100_000_000,
    }

    for name, value in targets.items():
        insert_target(source_id, national_stratum, name, value, "amount", 2024,
                     notes="JCT tax expenditure estimate")
        print(f"  {name}: ${value/1e9:.1f}B")

    print(f"  Loaded {len(targets)} tax targets")


def load_eitc_targets():
    """Load EITC targets by number of children."""
    print("\n=== Loading EITC Distribution Targets ===")
    df = fetch_csv("eitc.csv")

    source_id = get_or_create_source(
        "us", "irs", "soi_eitc",
        "IRS SOI EITC Statistics",
        "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics"
    )

    for _, row in df.iterrows():
        n_children = int(row["count_children"])
        stratum = get_or_create_stratum(
            f"Tax filers with {n_children} qualifying children", "us",
            [{"variable": "eitc_qualifying_children", "operator": "==", "value": str(n_children)}]
        )
        insert_target(source_id, stratum, "eitc_returns", row["eitc_returns"], "count", 2020)
        insert_target(source_id, stratum, "eitc_total", row["eitc_total"], "amount", 2020)
        print(f"  {n_children} children: {row['eitc_returns']:,.0f} returns, ${row['eitc_total']/1e9:.1f}B")

    print(f"  Loaded {len(df) * 2} EITC targets")


def main():
    print("=" * 70)
    print("LOADING POLICYENGINE CALIBRATION TARGETS TO SUPABASE")
    print("=" * 70)

    try:
        # Test connection
        result = supabase.select("sources", "id", limit=1)
        print(f"Connected to Supabase (found {len(result)} existing sources)")
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        return

    # Load all targets
    load_irs_income_targets()
    load_benefit_spending_targets()
    load_healthcare_targets()
    load_tax_targets()
    load_eitc_targets()
    load_medicaid_targets()
    load_snap_targets()
    load_aca_targets()
    load_population_targets()

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)

    # Summary
    sources = supabase.select("sources", "id")
    strata = supabase.select("strata", "id")
    targets = supabase.select("targets", "id")

    print(f"\nTotal in database:")
    print(f"  Sources: {len(sources)}")
    print(f"  Strata: {len(strata)}")
    print(f"  Targets: {len(targets)}")


if __name__ == "__main__":
    main()
