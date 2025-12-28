"""Panel/longitudinal synthesis for the Microplex.

Architecture for generating lifetime trajectories (e.g., earnings histories
for Social Security modeling).

Key insight from social-security-model:
1. Train QRF on PSID panel data to learn earnings trajectories
2. Apply to CPS cross-section to generate 35-year earnings histories
3. Calibrate to SSA administrative targets

Two-stage hierarchical + temporal synthesis:
1. Synthesize households (size, geography, housing) - HierarchicalSynthesizer
2. Synthesize persons within households - HierarchicalSynthesizer
3. Synthesize trajectories for each person - TrajectoryModel

This module implements Stage 3: trajectory synthesis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory synthesis."""

    # Time horizon
    start_age: int = 18
    end_age: int = 70
    years_of_history: int = 35

    # Variables to model trajectories for
    trajectory_vars: List[str] = None

    # Conditioning variables (from cross-section)
    condition_vars: List[str] = None

    # Training data source
    panel_data: str = "psid"  # or "sipp"

    def __post_init__(self):
        if self.trajectory_vars is None:
            self.trajectory_vars = ["earnings"]
        if self.condition_vars is None:
            self.condition_vars = ["education", "gender", "birth_cohort"]


class TrajectoryModel:
    """Synthesizes lifetime trajectories for individuals.

    Architecture: Generate full trajectory ALL AT ONCE (not sequentially).

    Why all-at-once vs sequential:
    - Sequential (AR): P(earnings_t | earnings_{t-1}, ...) compounds errors
    - All-at-once: P(earnings_18:70 | demographics) preserves correlations

    The model learns a latent "trajectory type" that captures the shape:
    - Type A: Steady growth (professional career)
    - Type B: Peak-then-decline (manual labor)
    - Type C: Volatile (self-employment, gig work)
    - Type D: Interrupted (disability, caregiving)

    Uses microplex's ConditionalMAF with:
    - Input (conditions): demographics, education, birth cohort
    - Output (targets): earnings at each age (35-50 dimensions)

    Training data: PSID (ideal, 50+ years) or SIPP (4-year panels, public)
    """

    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.is_fitted_ = False

    def fit(
        self,
        panel_data: pd.DataFrame,
        id_col: str = "person_id",
        time_col: str = "year",
        weight_col: Optional[str] = None,
    ) -> "TrajectoryModel":
        """Fit trajectory model on panel data (e.g., PSID).

        The panel data should have:
        - Multiple observations per person over time
        - Demographic characteristics
        - Economic variables (earnings, hours, etc.)

        Args:
            panel_data: Long-format panel DataFrame
            id_col: Column identifying individuals
            time_col: Column identifying time period
            weight_col: Optional weight column

        Returns:
            self
        """
        # TODO: Implement - this is the core panel synthesis logic
        # Key steps:
        # 1. Reshape panel to person-level with trajectory features
        # 2. Learn P(trajectory | initial_conditions)
        # 3. Store model for generation

        print("TrajectoryModel.fit() - not yet implemented")
        print(f"  Panel data: {len(panel_data):,} observations")
        print(f"  Unique individuals: {panel_data[id_col].nunique():,}")

        self.is_fitted_ = True
        return self

    def generate(
        self,
        cross_section: pd.DataFrame,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate trajectories for cross-sectional individuals.

        Takes a cross-section (e.g., from CPS) and generates full
        earnings histories for each person.

        Args:
            cross_section: Cross-sectional data with demographics
            seed: Random seed for reproducibility

        Returns:
            Wide-format DataFrame with earnings at each age
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        # TODO: Implement trajectory generation
        # Key steps:
        # 1. For each person, sample initial conditions
        # 2. Generate trajectory using learned model
        # 3. Reshape to wide format (earnings_age_18, earnings_age_19, ...)

        print("TrajectoryModel.generate() - not yet implemented")
        print(f"  Cross-section: {len(cross_section):,} individuals")

        # Placeholder: return cross-section with dummy trajectories
        result = cross_section.copy()
        for age in range(self.config.start_age, self.config.end_age + 1):
            result[f"earnings_age_{age}"] = np.nan

        return result


def create_psid_panel() -> pd.DataFrame:
    """Load and prepare PSID panel data for trajectory modeling.

    The PSID (Panel Study of Income Dynamics) is the gold standard
    for earnings trajectory modeling because it has 50+ years of
    longitudinal data on the same individuals.

    Returns:
        Long-format panel DataFrame
    """
    # TODO: Implement PSID loading
    # PSID data access requires registration
    # Alternative: use SIPP which has shorter panels but is public

    print("PSID loading not yet implemented")
    print("Consider using SIPP panels as alternative")

    return pd.DataFrame()


def demo_trajectory_synthesis():
    """Demonstrate the trajectory synthesis workflow."""
    print("=" * 70)
    print("TRAJECTORY SYNTHESIS DEMO")
    print("=" * 70)

    # Configuration
    config = TrajectoryConfig(
        trajectory_vars=["earnings"],
        condition_vars=["education", "gender", "age"],
        years_of_history=35,
    )

    # Create mock panel data (simulating PSID)
    print("\n1. Creating mock panel data...")
    np.random.seed(42)
    n_persons = 1000
    n_years = 10

    panel_data = []
    for person_id in range(n_persons):
        education = np.random.randint(1, 5)
        gender = np.random.randint(0, 2)
        base_earnings = 20000 + education * 10000

        for year in range(n_years):
            age = 25 + year
            # Simple earnings growth with noise
            earnings = base_earnings * (1.03 ** year) * np.random.lognormal(0, 0.3)
            panel_data.append({
                "person_id": person_id,
                "year": 2010 + year,
                "age": age,
                "education": education,
                "gender": gender,
                "earnings": earnings,
            })

    panel_df = pd.DataFrame(panel_data)
    print(f"  Created panel: {len(panel_df):,} observations")
    print(f"  {n_persons:,} persons × {n_years} years")

    # Fit trajectory model
    print("\n2. Fitting trajectory model...")
    model = TrajectoryModel(config)
    model.fit(panel_df)

    # Generate trajectories for cross-section
    print("\n3. Generating trajectories for cross-section...")
    cross_section = panel_df[panel_df.year == 2015][
        ["person_id", "education", "gender", "age", "earnings"]
    ].copy()
    print(f"  Cross-section: {len(cross_section):,} individuals")

    trajectories = model.generate(cross_section, seed=42)
    print(f"  Generated {len(trajectories):,} trajectories")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Implement TrajectoryModel using microplex's ConditionalMAF
   - Input: initial demographics + current state
   - Output: full trajectory vector (earnings at each age)

2. Load PSID or SIPP panel data for training
   - PSID: longer histories, requires registration
   - SIPP: shorter panels (4 years), publicly available

3. Add calibration to SSA administrative targets
   - Match aggregate earnings distributions by cohort
   - Match benefit claiming patterns

4. Integrate with HierarchicalSynthesizer
   - Household → Person → Trajectory
   - Each person gets a full lifetime earnings history
    """)


if __name__ == "__main__":
    demo_trajectory_synthesis()
