"""
Generate Pareto frontier chart: Sparsity vs Error for both calibration methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from microplex.calibration import SparseCalibrator, HardConcreteCalibrator


def generate_synthetic_population(n_records: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic population with known structure."""
    np.random.seed(seed)

    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
    state_probs = np.array([0.12, 0.09, 0.07, 0.06, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03])
    state_probs = state_probs / state_probs.sum()

    age_groups = ["0-17", "18-34", "35-54", "55-64", "65+"]
    age_probs = [0.22, 0.22, 0.26, 0.13, 0.17]

    income_brackets = ["<25k", "25-50k", "50-100k", "100k+"]
    income_probs = [0.20, 0.25, 0.35, 0.20]

    data = pd.DataFrame({
        "state": np.random.choice(states, n_records, p=state_probs),
        "age_group": np.random.choice(age_groups, n_records, p=age_probs),
        "income_bracket": np.random.choice(income_brackets, n_records, p=income_probs),
        "income": np.random.lognormal(10.5, 1.0, n_records),
        "weight": np.ones(n_records),
    })

    return data


def compute_targets(data: pd.DataFrame) -> tuple:
    """Compute calibration targets from data."""
    marginal_targets = {}

    for var in ["state", "age_group", "income_bracket"]:
        marginal_targets[var] = {}
        for val in data[var].unique():
            marginal_targets[var][val] = float((data[var] == val).sum())

    continuous_targets = {"income": float(data["income"].sum())}

    return marginal_targets, continuous_targets


def run_comparison(n_records: int = 5000):
    """Run both methods across sparsity range and collect results."""
    print(f"Generating {n_records} synthetic records...")
    pop = generate_synthetic_population(n_records=n_records)
    marginal_targets, continuous_targets = compute_targets(pop)

    # Cross-category results
    cc_results = []
    sparsity_levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\nRunning Cross-Category Selection...")
    for target_sparsity in sparsity_levels:
        try:
            cal = SparseCalibrator(target_sparsity=target_sparsity)
            result = cal.fit_transform(pop.copy(), marginal_targets, continuous_targets)
            val = cal.validate(result)
            cc_results.append({
                "sparsity": cal.get_sparsity(),
                "mean_error": val["mean_error"],
                "max_error": val["max_error"],
            })
            print(f"  target={target_sparsity:.0%} → actual={cal.get_sparsity():.1%}, error={val['mean_error']:.2%}")
        except Exception as e:
            print(f"  target={target_sparsity:.0%} failed: {e}")

    # Hard Concrete results - sweep lambda_l0
    hc_results = []
    lambda_values = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

    print("\nRunning Hard Concrete L0...")
    for lam in lambda_values:
        try:
            cal = HardConcreteCalibrator(
                lambda_l0=lam,
                epochs=1000,
                lr=0.1,
                verbose=False,
            )
            result = cal.fit_transform(pop.copy(), marginal_targets, continuous_targets)
            val = cal.validate(result)
            hc_results.append({
                "lambda": lam,
                "sparsity": cal.get_sparsity(),
                "mean_error": val["mean_error"],
                "max_error": val["max_error"],
            })
            print(f"  λ={lam:.0e} → sparsity={cal.get_sparsity():.1%}, error={val['mean_error']:.2%}")
        except Exception as e:
            print(f"  λ={lam:.0e} failed: {e}")

    return pd.DataFrame(cc_results), pd.DataFrame(hc_results)


def plot_pareto(cc_df: pd.DataFrame, hc_df: pd.DataFrame, output_path: str = "sparse_calibration_pareto.png"):
    """Plot Pareto frontier for both methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by sparsity for line plots
    cc_df = cc_df.sort_values("sparsity")
    hc_df = hc_df.sort_values("sparsity")

    # Plot Cross-Category
    ax.plot(
        cc_df["sparsity"] * 100,
        cc_df["mean_error"] * 100,
        "o-",
        color="#2ecc71",
        linewidth=2,
        markersize=8,
        label="Cross-Category + IPF",
    )

    # Plot Hard Concrete
    ax.plot(
        hc_df["sparsity"] * 100,
        hc_df["mean_error"] * 100,
        "s-",
        color="#3498db",
        linewidth=2,
        markersize=8,
        label="Hard Concrete L0",
    )

    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Mean Relative Error (%)", fontsize=12)
    ax.set_title("Sparse Calibration: Sparsity vs Accuracy Tradeoff", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 100)
    ax.set_ylim(bottom=0)

    # Add annotation
    ax.annotate(
        "Lower is better →",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        color="gray",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved chart to {output_path}")

    return fig


if __name__ == "__main__":
    cc_df, hc_df = run_comparison(n_records=5000)

    print("\n" + "=" * 60)
    print("Cross-Category Results:")
    print(cc_df.to_string(index=False))
    print("\nHard Concrete Results:")
    print(hc_df.to_string(index=False))

    plot_pareto(cc_df, hc_df)
