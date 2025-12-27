"""
CPS Multivariate Benchmark - Joint Holdout Similarity.

This runs the proper multivariate evaluation on CPS-like data:
- MMD (Maximum Mean Discrepancy)
- Energy Distance
- Authenticity (Synthetic → Holdout distance)
- Coverage (Holdout → Synthetic distance)

These metrics evaluate similarity to holdout records in the FULL JOINT SPACE,
not just marginal distributions.
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_cps_benchmark import generate_cps_like_data
from compare_qrf import SequentialQRF, SequentialQRFWithZeroInflation
from multivariate_metrics import (
    compute_multivariate_metrics,
    compare_methods_multivariate,
    print_multivariate_metrics_report,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def main():
    """Run multivariate benchmark on CPS-like data."""

    print("=" * 80)
    print("CPS MULTIVARIATE HOLDOUT BENCHMARK")
    print("=" * 80)
    print("\nEvaluating joint distribution similarity, not marginals.")
    print("Metrics: MMD, Energy Distance, Authenticity, Coverage\n")

    # Configuration
    n_train = 20000
    n_test = 5000

    # Target and condition variables (same as CPS benchmark)
    target_vars = [
        "wage_income",
        "self_emp_income",
        "ssi_income",
        "uc_income",
        "snap_benefit",
        "eitc",
        "agi",
        "federal_tax",
    ]

    condition_vars = [
        "age",
        "education",
        "is_employed",
        "marital_status",
    ]

    all_vars = target_vars + condition_vars

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Generate CPS-like data
    print(f"Generating CPS-like data...")
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")

    full_data = generate_cps_like_data(n_train + n_test, seed=42)

    # Convert categorical to numeric for distance calculations
    full_data['marital_status_num'] = pd.Categorical(full_data['marital_status']).codes

    train_data = full_data.iloc[:n_train].copy()
    test_data = full_data.iloc[n_train:].copy()
    test_conditions = test_data[condition_vars].copy()

    print("\nTarget variable zero-inflation rates:")
    for var in target_vars:
        zero_pct = (train_data[var] == 0).mean() * 100
        print(f"  {var}: {zero_pct:.1f}% zeros")

    # Train models
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)

    synthetic_datasets = {}

    # 1. microplex (MAF)
    print("\n[1/3] Training microplex (MAF)...")
    try:
        from microplex import Synthesizer

        start = time.time()
        microplex = Synthesizer(
            target_vars=target_vars,
            condition_vars=condition_vars,
            hidden_dim=64,
            n_layers=6,
            zero_inflated=True,  # Enable zero-inflation handling
        )
        microplex.fit(train_data, epochs=50, batch_size=256, verbose=False)
        train_time = time.time() - start

        start = time.time()
        synthetic = microplex.generate(test_conditions)
        gen_time = time.time() - start

        synthetic_datasets['microplex'] = synthetic
        print(f"  ✓ microplex: train={train_time:.1f}s, gen={gen_time:.2f}s")

    except Exception as e:
        print(f"  ✗ microplex failed: {e}")

    # 2. Sequential QRF (baseline)
    print("\n[2/3] Training Sequential QRF...")
    try:
        start = time.time()
        qrf = SequentialQRF(target_vars, condition_vars)
        qrf.fit(train_data, verbose=False)
        train_time = time.time() - start

        start = time.time()
        synthetic = qrf.generate(test_conditions)
        gen_time = time.time() - start

        synthetic_datasets['qrf_sequential'] = synthetic
        print(f"  ✓ qrf_sequential: train={train_time:.1f}s, gen={gen_time:.2f}s")

    except Exception as e:
        print(f"  ✗ qrf_sequential failed: {e}")

    # 3. QRF + Zero Inflation
    print("\n[3/3] Training QRF + Zero Inflation...")
    try:
        start = time.time()
        qrf_zi = SequentialQRFWithZeroInflation(target_vars, condition_vars)
        qrf_zi.fit(train_data, verbose=False)
        train_time = time.time() - start

        start = time.time()
        synthetic = qrf_zi.generate(test_conditions)
        gen_time = time.time() - start

        synthetic_datasets['qrf_zero_inflation'] = synthetic
        print(f"  ✓ qrf_zero_inflation: train={train_time:.1f}s, gen={gen_time:.2f}s")

    except Exception as e:
        print(f"  ✗ qrf_zero_inflation failed: {e}")

    if not synthetic_datasets:
        print("\n✗ All methods failed!")
        return

    print(f"\n✓ Successfully trained {len(synthetic_datasets)} method(s)")

    # Run multivariate comparison
    print("\n" + "=" * 80)
    print("COMPUTING MULTIVARIATE METRICS")
    print("=" * 80)

    # Use target vars only for joint evaluation (conditioning vars are same across all)
    eval_vars = target_vars

    comparison_df = compare_methods_multivariate(
        train_data=train_data,
        holdout_data=test_data,
        synthetic_datasets=synthetic_datasets,
        variables=eval_vars,
    )

    # Save results
    csv_path = output_dir / "cps_multivariate_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Create visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CPS Multivariate Benchmark: Joint Holdout Similarity", fontsize=16, fontweight="bold")

    methods = comparison_df['Method'].values
    colors = {
        'microplex': '#00d4ff',
        'qrf_sequential': '#707088',
        'qrf_zero_inflation': '#00ff88',
    }
    bar_colors = [colors.get(m, 'steelblue') for m in methods]

    # MMD (lower = better)
    ax = axes[0, 0]
    ax.bar(methods, comparison_df['MMD'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('MMD Statistic')
    ax.set_title('Maximum Mean Discrepancy\n(lower = better joint distribution match)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Energy Distance (lower = better)
    ax = axes[0, 1]
    ax.bar(methods, comparison_df['Energy Distance'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('Energy Distance')
    ax.set_title('Energy Distance\n(lower = better joint distribution match)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Authenticity (lower = more realistic individual records)
    ax = axes[1, 0]
    ax.bar(methods, comparison_df['Authenticity (mean)'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('Mean Distance to Nearest Holdout Record')
    ax.set_title('Authenticity (Synth → Holdout)\n(lower = more realistic records)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Coverage (lower = better coverage)
    ax = axes[1, 1]
    ax.bar(methods, comparison_df['Coverage (mean)'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('Mean Distance to Nearest Synthetic Record')
    ax.set_title('Coverage (Holdout → Synth)\n(lower = better data manifold coverage)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "cps_multivariate_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {fig_path}")
    plt.close()

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    best_mmd = comparison_df.loc[comparison_df['MMD'].idxmin()]
    best_energy = comparison_df.loc[comparison_df['Energy Distance'].idxmin()]
    best_auth = comparison_df.loc[comparison_df['Authenticity (mean)'].idxmin()]
    best_cov = comparison_df.loc[comparison_df['Coverage (mean)'].idxmin()]

    print(f"\nBest MMD (joint distribution match): {best_mmd['Method']} ({best_mmd['MMD']:.6f})")
    print(f"Best Energy Distance: {best_energy['Method']} ({best_energy['Energy Distance']:.6f})")
    print(f"Best Authenticity (individual realism): {best_auth['Method']} ({best_auth['Authenticity (mean)']:.4f})")
    print(f"Best Coverage (data manifold): {best_cov['Method']} ({best_cov['Coverage (mean)']:.4f})")

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("  - cps_multivariate_metrics.csv")
    print("  - cps_multivariate_comparison.png")


if __name__ == "__main__":
    main()
