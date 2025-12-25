"""
Run multivariate realism benchmark comparing microplex vs other methods.

Tests whether synthetic records are realistic in the full joint space,
not just marginally correct.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare import (
    MicroBenchmark,
    CTGANBenchmark,
    TVAEBenchmark,
    GaussianCopulaBenchmark,
)
from compare_qrf import SequentialQRF, SequentialQRFWithZeroInflation
from multivariate_metrics import (
    compute_multivariate_metrics,
    compare_methods_multivariate,
)
from run_benchmarks import generate_realistic_microdata

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


class QRFSequentialBenchmark:
    """Benchmark wrapper for Sequential QRF."""

    def __init__(self, target_vars, condition_vars, **kwargs):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.model = None

    def fit(self, data, **kwargs):
        self.model = SequentialQRF(self.target_vars, self.condition_vars)
        self.model.fit(data, verbose=False)

    def generate(self, conditions):
        return self.model.generate(conditions)


class QRFZeroInflationBenchmark:
    """Benchmark wrapper for Sequential QRF with Zero Inflation handling."""

    def __init__(self, target_vars, condition_vars, **kwargs):
        self.target_vars = target_vars
        self.condition_vars = condition_vars
        self.model = None

    def fit(self, data, **kwargs):
        self.model = SequentialQRFWithZeroInflation(self.target_vars, self.condition_vars)
        self.model.fit(data, verbose=False)

    def generate(self, conditions):
        return self.model.generate(conditions)


def create_multivariate_visualizations(
    comparison_df: pd.DataFrame,
    output_dir: Path,
):
    """Create visualizations for multivariate metrics comparison."""

    # 1. Distance metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multivariate Realism Metrics Comparison", fontsize=16, fontweight="bold")

    methods = comparison_df['Method'].values
    colors = {
        'microplex': '#27ae60',
        'qrf_sequential': '#e74c3c',
        'qrf_zero_inflation': '#f39c12',
        'ctgan': '#3498db',
        'tvae': '#9b59b6',
        'copula': '#1abc9c',
    }
    bar_colors = [colors.get(m, 'steelblue') for m in methods]

    # Authenticity (lower = more realistic)
    ax = axes[0, 0]
    ax.bar(methods, comparison_df['Authenticity (mean)'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('Mean Distance', fontsize=11)
    ax.set_title('Authenticity (Synth → Holdout)\n(lower = more realistic)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Coverage (lower = better coverage)
    ax = axes[0, 1]
    ax.bar(methods, comparison_df['Coverage (mean)'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('Mean Distance', fontsize=11)
    ax.set_title('Coverage (Holdout → Synth)\n(lower = better coverage)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Privacy Ratio (higher = better generalization)
    ax = axes[1, 0]
    ax.bar(methods, comparison_df['Privacy Ratio'], color=bar_colors, alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ideal = 1.0')
    ax.set_ylabel('Distance Ratio (Holdout/Train)', fontsize=11)
    ax.set_title('Privacy Ratio\n(> 1 = good generalization)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # MMD (lower = better distribution match)
    ax = axes[1, 1]
    ax.bar(methods, comparison_df['MMD'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('MMD Statistic', fontsize=11)
    ax.set_title('Maximum Mean Discrepancy\n(lower = better)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "multivariate_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved multivariate comparison to {output_dir / 'multivariate_comparison.png'}")
    plt.close()

    # 2. Privacy analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Privacy & Overfitting Analysis", fontsize=16, fontweight="bold")

    # Min authenticity distance (privacy concern if too low)
    ax = axes[0]
    bars = ax.bar(methods, comparison_df['Authenticity (min)'], color=bar_colors, alpha=0.8)
    ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Privacy threshold')
    ax.set_ylabel('Min Distance to Real Record', fontsize=11)
    ax.set_title('Minimum Authenticity Distance\n(< 0.1 = privacy risk)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Fraction closer to training data
    ax = axes[1]
    ax.bar(methods, comparison_df['Closer to Train (%)'], color=bar_colors, alpha=0.8)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random baseline')
    ax.set_ylabel('Percentage', fontsize=11)
    ax.set_title('Records Closer to Train than Holdout\n(< 50% = good)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "privacy_analysis.png", dpi=300, bbox_inches="tight")
    print(f"Saved privacy analysis to {output_dir / 'privacy_analysis.png'}")
    plt.close()

    # 3. Multivariate distribution tests
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multivariate Distribution Tests", fontsize=16, fontweight="bold")

    # MMD
    ax = axes[0]
    ax.bar(methods, comparison_df['MMD'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('MMD Statistic', fontsize=11)
    ax.set_title('Maximum Mean Discrepancy (RBF Kernel)\n(0 = identical distributions)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Energy Distance
    ax = axes[1]
    ax.bar(methods, comparison_df['Energy Distance'], color=bar_colors, alpha=0.8)
    ax.set_ylabel('Energy Distance', fontsize=11)
    ax.set_title('Energy Distance\n(0 = identical distributions)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_tests.png", dpi=300, bbox_inches="tight")
    print(f"Saved distribution tests to {output_dir / 'distribution_tests.png'}")
    plt.close()


def main():
    """Run multivariate realism benchmark."""

    print("=" * 80)
    print("MULTIVARIATE REALISM BENCHMARK")
    print("=" * 80)
    print("\nTesting: Do synthetic records look like plausible real records")
    print("in the full joint space, or just marginally correct?\n")

    # Configuration
    n_train = 5000
    n_test = 1000
    epochs = 50

    target_vars = ["income", "assets", "debt", "savings"]
    condition_vars = ["age", "education", "region"]
    all_vars = target_vars + condition_vars

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Generate data
    print(f"Generating realistic microdata...")
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}\n")

    full_data = generate_realistic_microdata(n_train + n_test, seed=42)
    train_data = full_data.iloc[:n_train].copy()
    test_data = full_data.iloc[n_train:].copy()
    test_conditions = test_data[condition_vars].copy()

    # Define benchmarks to run
    benchmarks_to_run = {
        'microplex': MicroBenchmark,
        'qrf_sequential': QRFSequentialBenchmark,
        'copula': GaussianCopulaBenchmark,
    }

    # Try to include deep learning methods if available
    try:
        from compare import CTGANBenchmark, TVAEBenchmark
        benchmarks_to_run['ctgan'] = CTGANBenchmark
        benchmarks_to_run['tvae'] = TVAEBenchmark
    except ImportError:
        print("⚠️  SDV not available, skipping CT-GAN and TVAE")

    # Train models and generate synthetic data
    synthetic_datasets = {}

    for method_name, benchmark_cls in benchmarks_to_run.items():
        print(f"\n{'=' * 80}")
        print(f"Training: {method_name}")
        print('=' * 80)

        try:
            benchmark = benchmark_cls(target_vars, condition_vars)

            # Training
            print(f"  Training {method_name}...")
            if method_name.startswith('qrf'):
                benchmark.fit(train_data)
            else:
                benchmark.fit(train_data, epochs=epochs)

            # Generation
            print(f"  Generating synthetic data...")
            synthetic = benchmark.generate(test_conditions)
            synthetic_datasets[method_name] = synthetic

            print(f"  ✓ {method_name} completed")

        except Exception as e:
            print(f"  ✗ {method_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not synthetic_datasets:
        print("\n✗ All methods failed!")
        return

    print(f"\n✓ Successfully trained {len(synthetic_datasets)} method(s)")

    # Run multivariate metrics comparison
    print("\n" + "=" * 80)
    print("COMPUTING MULTIVARIATE METRICS")
    print("=" * 80)

    comparison_df = compare_methods_multivariate(
        train_data=train_data,
        holdout_data=test_data,
        synthetic_datasets=synthetic_datasets,
        variables=all_vars,  # Use all variables for joint space
    )

    # Save results
    csv_path = output_dir / "multivariate_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    create_multivariate_visualizations(comparison_df, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    best_auth = comparison_df.loc[comparison_df['Authenticity (mean)'].idxmin()]
    print(f"\nMost realistic records (lowest authenticity distance):")
    print(f"  {best_auth['Method']}: {best_auth['Authenticity (mean)']:.4f}")

    best_cov = comparison_df.loc[comparison_df['Coverage (mean)'].idxmin()]
    print(f"\nBest coverage (lowest coverage distance):")
    print(f"  {best_cov['Method']}: {best_cov['Coverage (mean)']:.4f}")

    best_priv = comparison_df.loc[comparison_df['Privacy Ratio'].idxmax()]
    print(f"\nBest generalization (highest privacy ratio):")
    print(f"  {best_priv['Method']}: {best_priv['Privacy Ratio']:.4f}")

    best_mmd = comparison_df.loc[comparison_df['MMD'].idxmin()]
    print(f"\nBest multivariate distribution match (lowest MMD):")
    print(f"  {best_mmd['Method']}: {best_mmd['MMD']:.6f}")

    # Check for privacy concerns
    print("\nPrivacy concerns (min distance < 0.1):")
    privacy_issues = comparison_df[comparison_df['Authenticity (min)'] < 0.1]
    if len(privacy_issues) > 0:
        for _, row in privacy_issues.iterrows():
            print(f"  ⚠️  {row['Method']}: min distance = {row['Authenticity (min)']:.4f}")
    else:
        print("  ✓ No privacy concerns detected")

    # Check for overfitting
    print("\nOverfitting concerns (> 50% closer to train):")
    overfitting_issues = comparison_df[comparison_df['Closer to Train (%)'] > 50]
    if len(overfitting_issues) > 0:
        for _, row in overfitting_issues.iterrows():
            print(f"  ⚠️  {row['Method']}: {row['Closer to Train (%)']:.1f}% closer to train")
    else:
        print("  ✓ No overfitting concerns detected")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - multivariate_metrics.csv: Detailed metrics table")
    print(f"  - multivariate_comparison.png: Main comparison chart")
    print(f"  - privacy_analysis.png: Privacy & overfitting analysis")
    print(f"  - distribution_tests.png: MMD & energy distance")
    print("\n")


if __name__ == "__main__":
    main()
