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
import time
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError
from io import StringIO
from typing import Optional, List, Dict, Any, Union

# Supabase connection - use Cosilico DB
SUPABASE_URL = "https://nsupqhfchdtqclomlrgs.supabase.co"
# Use service_role key for writes
SUPABASE_KEY = os.environ.get(
    "COSILICO_SUPABASE_SERVICE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5zdXBxaGZjaGR0cWNsb21scmdzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NjkzMTEwOCwiZXhwIjoyMDgyNTA3MTA4fQ.IZX2C6dM6CCuxzBeg3zoZSA31p_jy9XLjdxjaE126BU"
)


class SupabaseClient:
    """Simple REST client for Supabase microplex schema with retries."""

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
        self.session = requests.Session()

    def _request_with_retry(self, method: str, url: str, max_retries: int = 8, **kwargs) -> requests.Response:
        """Make a request with exponential backoff retry on timeouts."""
        kwargs.setdefault("timeout", 30)
        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, **kwargs)
                if resp.status_code in [429, 500, 502, 503, 504]:
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 60)  # Cap at 60s
                        print(f"    (Retry {attempt+1}/{max_retries} after {resp.status_code}, waiting {wait_time}s)")
                        time.sleep(wait_time)
                        continue
                return resp
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError, Urllib3ReadTimeoutError, Exception) as e:
                # Catch all exceptions for network issues
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 60)  # Cap at 60s
                        print(f"    (Retry {attempt+1}/{max_retries} after {type(e).__name__}, waiting {wait_time}s)")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise
        return resp

    def select(self, table: str, columns: str = "*", filters: Dict = None, limit: int = None) -> List[Dict]:
        """Select rows from a table."""
        url = f"{self.base_url}/{table}?select={columns}"
        if filters:
            for k, v in filters.items():
                url += f"&{k}=eq.{v}"
        if limit:
            url += f"&limit={limit}"
        resp = self._request_with_retry("GET", url, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def insert(self, table: str, data: Dict) -> Dict:
        """Insert a row, returning the inserted data."""
        url = f"{self.base_url}/{table}"
        resp = self._request_with_retry("POST", url, headers=self.headers, json=data)
        resp.raise_for_status()
        return resp.json()[0] if resp.json() else {}

    def upsert(self, table: str, data: Dict, on_conflict: str = None) -> Dict:
        """Upsert a row."""
        url = f"{self.base_url}/{table}"
        headers = {**self.headers, "Prefer": "resolution=merge-duplicates,return=representation"}
        if on_conflict:
            url += f"?on_conflict={on_conflict}"
        resp = self._request_with_retry("POST", url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.json()[0] if resp.json() else {}

    def update(self, table: str, data: Dict, filters: Dict) -> List[Dict]:
        """Update rows matching filters."""
        url = f"{self.base_url}/{table}"
        filter_parts = [f"{k}=eq.{v}" for k, v in filters.items()]
        if filter_parts:
            url += "?" + "&".join(filter_parts)
        resp = self._request_with_retry("PATCH", url, headers=self.headers, json=data)
        resp.raise_for_status()
        return resp.json()

    def batch_upsert_strata(self, strata: List[Dict], return_mapping: bool = False) -> Any:
        """
        Batch upsert multiple strata in a single request.

        Args:
            strata: List of dicts with name, jurisdiction, description
            return_mapping: If True, return dict mapping (name, jurisdiction) -> id

        Returns:
            List of inserted/updated records, or mapping if return_mapping=True
        """
        if not strata:
            return {} if return_mapping else []

        url = f"{self.base_url}/strata?on_conflict=name,jurisdiction"
        headers = {
            **self.headers,
            "Prefer": "resolution=merge-duplicates,return=representation"
        }

        resp = self._request_with_retry("POST", url, headers=headers, json=strata)
        resp.raise_for_status()
        result = resp.json()

        if return_mapping:
            return {(r["name"], r["jurisdiction"]): r["id"] for r in result}
        return result

    def batch_upsert_targets(self, targets: List[Dict], chunk_size: int = 500) -> List[Dict]:
        """
        Batch upsert multiple targets, chunking large batches.

        Args:
            targets: List of target dicts
            chunk_size: Max records per request (default 500)

        Returns:
            List of all inserted/updated records
        """
        if not targets:
            return []

        results = []
        # Upsert on composite key
        url = f"{self.base_url}/targets?on_conflict=source_id,stratum_id,variable,period"
        headers = {
            **self.headers,
            "Prefer": "resolution=merge-duplicates,return=representation"
        }

        # Process in chunks
        for i in range(0, len(targets), chunk_size):
            chunk = targets[i:i + chunk_size]
            resp = self._request_with_retry("POST", url, headers=headers, json=chunk)
            resp.raise_for_status()
            results.extend(resp.json())

        return results


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
    # URL-encode the name for the query
    import urllib.parse
    encoded_name = urllib.parse.quote(name, safe='')

    # Try to find existing stratum
    url = f"{supabase.base_url}/strata?select=id&name=eq.{encoded_name}&jurisdiction=eq.{jurisdiction}"
    resp = supabase._request_with_retry("GET", url, headers=supabase.headers)
    if resp.status_code == 200 and resp.json():
        return resp.json()[0]["id"]

    # Try to insert, handle conflict gracefully
    try:
        inserted = supabase.insert("strata", {
            "name": name,
            "jurisdiction": jurisdiction,
            "description": name
        })
        stratum_id = inserted["id"]
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 409:
            # Conflict - stratum already exists, fetch it
            resp = supabase._request_with_retry("GET", url, headers=supabase.headers)
            if resp.status_code == 200 and resp.json():
                return resp.json()[0]["id"]
            raise  # Re-raise if we still can't find it
        raise

    # Add constraints if provided
    if constraints:
        for c in constraints:
            try:
                supabase.insert("stratum_constraints", {
                    "stratum_id": stratum_id,
                    "variable": c["variable"],
                    "operator": c["operator"],
                    "value": c["value"]
                })
            except requests.exceptions.HTTPError as e:
                if e.response.status_code != 409:  # Ignore duplicate constraints
                    raise

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
    resp = supabase._request_with_retry("GET", url, headers=supabase.headers)
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
    loaded = 0
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
        loaded += 1
        if loaded % 10 == 0:
            print(f"    Progress: {loaded} states processed...")

    print(f"  Loaded {loaded} state targets")


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
    loaded = 0
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
        loaded += 1
        if loaded % 10 == 0:
            print(f"    Progress: {loaded} states processed...")

    print(f"  Loaded {loaded * 2} state targets")


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
    loaded = 0
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
        loaded += 1
        if loaded % 10 == 0:
            print(f"    Progress: {loaded} states processed...")

    print(f"  Loaded {loaded * 2} state targets")


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
    loaded = 0
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
        loaded += 1
        if loaded % 10 == 0:
            print(f"    Progress: {loaded} states processed...")

    print(f"  Loaded {loaded * 2} state targets")


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


def load_soi_targets():
    """Load full IRS SOI targets by AGI bracket, filing status, year."""
    print("\n=== Loading SOI Targets by AGI Bracket ===")
    df = fetch_csv("soi_targets.csv")

    source_id = get_or_create_source(
        "us", "irs", "soi_detailed",
        "IRS Statistics of Income - Detailed",
        "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics"
    )

    loaded = 0
    # Group by unique combinations to create strata
    # Column names have spaces: "Filing status", "AGI lower bound", "AGI upper bound"
    for (year, filing_status, agi_lower, agi_upper, variable), group in df.groupby(
        ["Year", "Filing status", "AGI lower bound", "AGI upper bound", "Variable"]
    ):
        # Handle -inf/inf comparisons
        is_neg_inf_lower = (agi_lower == float('-inf')) or (str(agi_lower) == "-inf")
        is_pos_inf_upper = (agi_upper == float('inf')) or (str(agi_upper) == "inf")

        # Create stratum name
        if is_neg_inf_lower and is_pos_inf_upper:
            agi_range = "all AGI"
        elif is_neg_inf_lower:
            agi_range = f"AGI < ${float(agi_upper):,.0f}"
        elif is_pos_inf_upper:
            agi_range = f"AGI >= ${float(agi_lower):,.0f}"
        else:
            agi_range = f"AGI ${float(agi_lower):,.0f}-${float(agi_upper):,.0f}"

        stratum_name = f"Tax filers {filing_status} {agi_range}"

        constraints = []
        if filing_status != "All":
            constraints.append({"variable": "filing_status", "operator": "==", "value": filing_status})
        if not is_neg_inf_lower:
            constraints.append({"variable": "adjusted_gross_income", "operator": ">=", "value": str(agi_lower)})
        if not is_pos_inf_upper:
            constraints.append({"variable": "adjusted_gross_income", "operator": "<", "value": str(agi_upper)})

        stratum_id = get_or_create_stratum(stratum_name, "us", constraints if constraints else None)

        # Get the value - use first row (they should be same for this grouping)
        value = group["Value"].iloc[0]
        is_count = group["Count"].iloc[0] if "Count" in group.columns else False
        target_type = "count" if is_count else "amount"

        insert_target(source_id, stratum_id, variable, value, target_type, int(year))
        loaded += 1

        if loaded % 500 == 0:
            print(f"    Progress: {loaded} targets processed...")

    print(f"  Loaded {loaded} SOI targets")


def load_age_state_targets():
    """Load age distribution by state."""
    print("\n=== Loading Age by State Targets ===")
    df = fetch_csv("age_state.csv")

    source_id = get_or_create_source(
        "us", "census", "acs_age",
        "Census ACS Age Distribution",
        "https://data.census.gov/"
    )

    # Age columns in the file
    age_cols = [c for c in df.columns if c not in ["GEO_ID", "GEO_NAME"]]

    loaded = 0
    for _, row in df.iterrows():
        # Extract state FIPS from GEO_ID (format: 0400000US01)
        fips = row["GEO_ID"][-2:]
        state = STATE_FIPS.get(fips)
        if not state:
            continue

        # Create state stratum
        state_stratum = get_or_create_stratum(
            f"State {state} population", f"us-{state.lower()}",
            [{"variable": "state_fips", "operator": "==", "value": fips}]
        )

        # Insert target for each age group
        for age_col in age_cols:
            # Parse age range from column name (e.g., "0-4", "5-9", "85+")
            insert_target(source_id, state_stratum, f"population_age_{age_col}", row[age_col], "count", 2024)
            loaded += 1

        if (loaded // len(age_cols)) % 10 == 0 and loaded % len(age_cols) == 0:
            print(f"    Progress: {loaded // len(age_cols)} states processed...")

    print(f"  Loaded {loaded} age-state targets")


def load_agi_state_targets():
    """Load AGI distribution by state and bracket."""
    print("\n=== Loading AGI by State Targets ===")
    df = fetch_csv("agi_state.csv")

    source_id = get_or_create_source(
        "us", "irs", "soi_state",
        "IRS SOI State Data",
        "https://www.irs.gov/statistics/soi-tax-stats-historic-table-2"
    )

    loaded = 0
    for _, row in df.iterrows():
        # Skip rows with missing GEO_ID
        geo_id = row["GEO_ID"]
        if pd.isna(geo_id) or not isinstance(geo_id, str):
            continue

        # Extract state FIPS
        fips = geo_id[-2:]
        state = STATE_FIPS.get(fips)
        if not state:
            continue

        agi_lower = row["AGI_LOWER_BOUND"]
        agi_upper = row["AGI_UPPER_BOUND"]

        # Handle -inf/inf comparisons
        is_neg_inf_lower = pd.isna(agi_lower) or (str(agi_lower) == "-inf") or (agi_lower == float('-inf'))
        is_pos_inf_upper = pd.isna(agi_upper) or (str(agi_upper) == "inf") or (agi_upper == float('inf'))

        # Create stratum name
        if is_neg_inf_lower and is_pos_inf_upper:
            agi_range = "all AGI"
        elif is_neg_inf_lower:
            agi_range = f"AGI < ${float(agi_upper):,.0f}"
        elif is_pos_inf_upper:
            agi_range = f"AGI >= ${float(agi_lower):,.0f}"
        else:
            agi_range = f"AGI ${float(agi_lower):,.0f}-${float(agi_upper):,.0f}"

        stratum_name = f"State {state} {agi_range}"

        constraints = [{"variable": "state_fips", "operator": "==", "value": fips}]
        if not is_neg_inf_lower:
            constraints.append({"variable": "adjusted_gross_income", "operator": ">=", "value": str(agi_lower)})
        if not is_pos_inf_upper:
            constraints.append({"variable": "adjusted_gross_income", "operator": "<", "value": str(agi_upper)})

        stratum_id = get_or_create_stratum(stratum_name, f"us-{state.lower()}", constraints)

        is_count = row["IS_COUNT"] if "IS_COUNT" in row else False
        target_type = "count" if is_count else "amount"
        variable = row["VARIABLE"] if "VARIABLE" in row else "adjusted_gross_income"

        insert_target(source_id, stratum_id, variable, row["VALUE"], target_type, 2024)
        loaded += 1

        if loaded % 100 == 0:
            print(f"    Progress: {loaded} targets processed...")

    print(f"  Loaded {loaded} AGI-state targets")


def load_spm_agi_targets():
    """Load SPM threshold by AGI decile."""
    print("\n=== Loading SPM by AGI Decile Targets ===")
    df = fetch_csv("spm_threshold_agi.csv")

    source_id = get_or_create_source(
        "us", "census", "spm",
        "Census SPM Thresholds",
        "https://www.census.gov/topics/income-poverty/supplemental-poverty-measure.html"
    )

    loaded = 0
    for _, row in df.iterrows():
        decile = int(row["decile"])
        spm_lower = row["lower_spm_threshold"]
        spm_upper = row["upper_spm_threshold"]

        stratum_name = f"Decile {decile} (SPM ${spm_lower:,.0f}-${spm_upper:,.0f})"

        constraints = [
            {"variable": "spm_threshold", "operator": ">=", "value": str(spm_lower)},
            {"variable": "spm_threshold", "operator": "<", "value": str(spm_upper)}
        ]

        stratum_id = get_or_create_stratum(stratum_name, "us", constraints)

        insert_target(source_id, stratum_id, "adjusted_gross_income", row["adjusted_gross_income"], "amount", 2024)
        insert_target(source_id, stratum_id, "count", row["count"], "count", 2024)
        loaded += 2

    print(f"  Loaded {loaded} SPM-AGI targets")


def load_real_estate_tax_targets():
    """Load real estate taxes by state."""
    print("\n=== Loading Real Estate Tax by State Targets ===")
    df = fetch_csv("real_estate_taxes_by_state_acs.csv")

    source_id = get_or_create_source(
        "us", "census", "acs_real_estate_taxes",
        "Census ACS Real Estate Taxes",
        "https://data.census.gov/"
    )

    loaded = 0
    for _, row in df.iterrows():
        state = row["state_code"]
        fips = STATE_NAME_TO_FIPS.get(state)
        if not fips:
            continue

        stratum = get_or_create_stratum(
            f"State {state} population", f"us-{state.lower()}",
            [{"variable": "state_fips", "operator": "==", "value": fips}]
        )

        # Value is in billions, convert to dollars
        value = row["real_estate_taxes_bn"] * 1e9
        insert_target(source_id, stratum, "real_estate_taxes", value, "amount", 2024)
        loaded += 1

        if loaded % 10 == 0:
            print(f"    Progress: {loaded} states processed...")

    print(f"  Loaded {loaded} real estate tax targets")


def load_census_projection_targets():
    """Load Census population projections by race/sex/year (totals only, not age-specific)."""
    print("\n=== Loading Census Population Projections ===")
    df = fetch_csv("np2023_d5_mid.csv")

    source_id = get_or_create_source(
        "us", "census", "population_projections_detailed",
        "Census Population Projections 2023",
        "https://www.census.gov/programs-surveys/popproj.html"
    )

    # RACE_HISP mapping (from Census)
    race_map = {0: "All", 1: "White non-Hispanic", 2: "Black", 3: "AIAN",
                4: "Asian", 5: "NHPI", 6: "Two+", 7: "Hispanic"}
    sex_map = {0: "Both", 1: "Male", 2: "Female"}
    nativity_map = {0: "All", 1: "Native", 2: "Foreign-born"}

    loaded = 0
    # Only load total population by demographic group (not age-specific to keep size reasonable)
    for _, row in df.iterrows():
        year = int(row["YEAR"])
        nativity = int(row["NATIVITY"])
        race = int(row["RACE_HISP"])
        sex = int(row["SEX"])

        # Create stratum - skip constraints to avoid conflicts, year in name suffices
        parts = []

        if nativity != 0:
            parts.append(nativity_map.get(nativity, f"Nativity {nativity}"))
        if race != 0:
            parts.append(race_map.get(race, f"Race {race}"))
        if sex != 0:
            parts.append(sex_map.get(sex, f"Sex {sex}"))

        base_name = " ".join(parts) if parts else "Total population"
        stratum_name = f"Census projection {base_name} ({year})"
        stratum_id = get_or_create_stratum(stratum_name, "us", None)  # No constraints

        # Insert total population only (skip age-specific to keep manageable)
        insert_target(source_id, stratum_id, "total_population", row["TOTAL_POP"], "count", year)
        loaded += 1

        if loaded % 500 == 0:
            print(f"    Progress: {loaded} targets processed...")

    print(f"  Loaded {loaded} Census projection targets")


def load_healthcare_age_targets():
    """Load healthcare spending by age bracket."""
    print("\n=== Loading Healthcare Spending by Age ===")
    df = fetch_csv("healthcare_spending.csv")

    source_id = get_or_create_source(
        "us", "bls", "healthcare_spending_age",
        "BLS Consumer Expenditure Survey - Healthcare",
        "https://www.bls.gov/cex/"
    )

    loaded = 0
    for _, row in df.iterrows():
        age_lower = int(row["age_10_year_lower_bound"])

        # Create age bracket stratum - use "healthcare" prefix to avoid conflicts
        if age_lower == 80:
            stratum_name = f"Healthcare population age 80+"
            constraints = [{"variable": "age", "operator": ">=", "value": "80"}]
        else:
            age_upper = age_lower + 10
            stratum_name = f"Healthcare population age {age_lower}-{age_upper-1}"
            constraints = [
                {"variable": "age", "operator": ">=", "value": str(age_lower)},
                {"variable": "age", "operator": "<", "value": str(age_upper)}
            ]

        stratum_id = get_or_create_stratum(stratum_name, "us", constraints)

        # Insert each spending category
        for col in ["health_insurance_premiums_without_medicare_part_b",
                    "over_the_counter_health_expenses", "other_medical_expenses",
                    "medicare_part_b_premiums"]:
            if col in row:
                insert_target(source_id, stratum_id, col, row[col], "amount", 2024)
                loaded += 1

    print(f"  Loaded {loaded} healthcare-age targets")


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
    # National aggregates
    load_irs_income_targets()
    load_benefit_spending_targets()
    load_healthcare_targets()
    load_tax_targets()

    # By number of children
    load_eitc_targets()

    # By state
    load_medicaid_targets()
    load_snap_targets()
    load_aca_targets()
    load_population_targets()
    load_age_state_targets()
    load_agi_state_targets()
    load_real_estate_tax_targets()

    # By demographic/bracket
    load_soi_targets()
    load_spm_agi_targets()
    load_healthcare_age_targets()

    # Census projections (large - do last)
    load_census_projection_targets()

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
