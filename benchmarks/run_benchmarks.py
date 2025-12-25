"""
Comprehensive benchmark suite for microplex.

Generates realistic economic microdata with zero-inflation and runs
comparisons against CT-GAN, TVAE, and Gaussian Copula methods.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare import (
    BenchmarkResult,
    CTGANBenchmark,
    GaussianCopulaBenchmark,
    MicroBenchmark,
    TVAEBenchmark,
    compute_correlation_fidelity,
    compute_marginal_fidelity,
    compute_zero_fidelity,
)

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def generate_realistic_microdata(
    n_samples: int = 10000, seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic economic microdata with zero-inflation.

    Simulates household survey data with:
    - Demographics (age, education, region)
    - Economic outcomes (income, assets, debt)
    - Zero-inflated variables (assets, debt)
    - Realistic correlations

    This mimics CPS/ACS-style survey data.
    """
    np.random.seed(seed)

    # Condition variables (demographics)
    age = np.random.normal(45, 15, n_samples).clip(18, 90)
    education = np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.3, 0.35, 0.25])
    region = np.random.choice([1, 2, 3, 4], n_samples, p=[0.18, 0.24, 0.37, 0.21])

    # Target variables (economic outcomes)
    # Income: log-normal with education/age effects
    base_income = 30000 + education * 15000 + (age - 18) * 800
    income = np.maximum(0, np.random.lognormal(
        np.log(base_income) - 0.5, 0.7, n_samples
    ))

    # Assets: zero-inflated (40% have no assets)
    has_assets = np.random.random(n_samples) > 0.4
    base_assets = income * 2 + education * 10000
    assets = np.where(
        has_assets,
        np.maximum(0, np.random.lognormal(np.log(base_assets) - 1.0, 1.2, n_samples)),
        0
    )

    # Debt: zero-inflated (50% have no debt)
    has_debt = np.random.random(n_samples) > 0.5
    base_debt = income * 0.5 + education * 5000
    debt = np.where(
        has_debt,
        np.maximum(0, np.random.lognormal(np.log(base_debt) - 1.5, 1.0, n_samples)),
        0
    )

    # Savings: can be negative, correlated with income and assets
    savings = 0.1 * income + 0.05 * assets - 0.1 * debt + np.random.normal(
        0, 5000, n_samples
    )

    return pd.DataFrame({
        "age": age,
        "education": education,
        "region": region,
        "income": income,
        "assets": assets,
        "debt": debt,
        "savings": savings,
    })


def run_single_benchmark(
    method_name: str,
    benchmark_cls,
    train_data: pd.DataFrame,
    test_conditions: pd.DataFrame,
    target_vars: list,
    condition_vars: list,
    epochs: int = 100,
) -> BenchmarkResult:
    """Run a single benchmark method."""
    print(f"\nBenchmarking {method_name}...")
    print("-" * 60)

    benchmark = benchmark_cls(target_vars, condition_vars)

    # Training
    print(f"  Training {method_name}...")
    start = time.time()
    try:
        if method_name == "micro":
            benchmark.fit(train_data, epochs=epochs)
        else:
            benchmark.fit(train_data, epochs=epochs)
    except Exception as e:
        print(f"  FAILED during training: {e}")
        raise
    train_time = time.time() - start
    print(f"  Training completed in {train_time:.1f}s")

    # Generation
    print(f"  Generating synthetic data...")
    start = time.time()
    try:
        synthetic = benchmark.generate(test_conditions)
    except Exception as e:
        print(f"  FAILED during generation: {e}")
        raise
    generate_time = time.time() - start
    print(f"  Generation completed in {generate_time:.1f}s")

    # Compute metrics
    print(f"  Computing fidelity metrics...")
    ks_stats, mean_ks = compute_marginal_fidelity(train_data, synthetic, target_vars)
    corr_error = compute_correlation_fidelity(train_data, synthetic, target_vars)
    zero_errors, mean_zero = compute_zero_fidelity(train_data, synthetic, target_vars)

    result = BenchmarkResult(
        method=method_name,
        dataset="economic_microdata",
        ks_stats=ks_stats,
        mean_ks=mean_ks,
        correlation_error=corr_error,
        zero_fraction_error=zero_errors,
        mean_zero_error=mean_zero,
        train_time=train_time,
        generate_time=generate_time,
        n_train=len(train_data),
        n_generate=len(test_conditions),
    )

    print(f"\n  Results:")
    print(f"    Marginal fidelity (mean KS): {mean_ks:.4f}")
    print(f"    Correlation error: {corr_error:.4f}")
    print(f"    Zero-fraction error: {mean_zero:.4f}")
    print(f"    Training time: {train_time:.1f}s")
    print(f"    Generation time: {generate_time:.1f}s")

    return result, synthetic


def create_visualizations(
    results: list,
    real_data: pd.DataFrame,
    synthetic_data: dict,
    output_dir: Path,
):
    """Create comprehensive benchmark visualization report."""

    # 1. Summary metrics bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = [r.method for r in results]

    # KS statistic (lower is better)
    axes[0, 0].bar(methods, [r.mean_ks for r in results], color="steelblue")
    axes[0, 0].set_ylabel("Mean KS Statistic")
    axes[0, 0].set_title("Marginal Distribution Fidelity (lower is better)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Correlation error (lower is better)
    axes[0, 1].bar(methods, [r.correlation_error for r in results], color="coral")
    axes[0, 1].set_ylabel("Correlation Matrix Error")
    axes[0, 1].set_title("Joint Distribution Fidelity (lower is better)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Zero-fraction error (lower is better)
    axes[1, 0].bar(methods, [r.mean_zero_error for r in results], color="seagreen")
    axes[1, 0].set_ylabel("Mean Zero-Fraction Error")
    axes[1, 0].set_title("Zero-Inflation Handling (lower is better)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Training time (lower is better)
    axes[1, 1].bar(methods, [r.train_time for r in results], color="purple")
    axes[1, 1].set_ylabel("Training Time (s)")
    axes[1, 1].set_title("Training Speed (lower is better)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", dpi=300, bbox_inches="tight")
    print(f"Saved summary metrics to {output_dir / 'summary_metrics.png'}")
    plt.close()

    # 2. Distribution comparison for each method
    target_vars = ["income", "assets", "debt", "savings"]

    for method_name, synthetic in synthetic_data.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Distribution Comparison: {method_name.upper()}", fontsize=16)

        for idx, var in enumerate(target_vars):
            ax = axes[idx // 2, idx % 2]

            # Plot histograms
            ax.hist(
                real_data[var],
                bins=50,
                alpha=0.5,
                label="Real",
                density=True,
                color="blue",
            )
            ax.hist(
                synthetic[var],
                bins=50,
                alpha=0.5,
                label="Synthetic",
                density=True,
                color="red",
            )

            # Add KS statistic
            ks_stat = next(r for r in results if r.method == method_name).ks_stats[var]
            ax.text(
                0.95,
                0.95,
                f"KS: {ks_stat:.4f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.set_xlabel(var.capitalize())
            ax.set_ylabel("Density")
            ax.legend()
            ax.set_xlim(0, np.percentile(real_data[var], 95))

        plt.tight_layout()
        plt.savefig(
            output_dir / f"distributions_{method_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved distribution comparison for {method_name}")
        plt.close()

    # 3. Zero-inflation comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    zero_vars = ["assets", "debt"]
    width = 0.15
    x = np.arange(len(zero_vars))

    # Real zero fractions
    real_zeros = [(real_data[var] == 0).mean() for var in zero_vars]

    for i, method_name in enumerate(methods):
        synthetic = synthetic_data[method_name]
        synth_zeros = [(synthetic[var] == 0).mean() for var in zero_vars]
        axes[0].bar(
            x + i * width, synth_zeros, width, label=method_name, alpha=0.8
        )

    axes[0].bar(x + len(methods) * width, real_zeros, width, label="Real", alpha=0.8)
    axes[0].set_ylabel("Zero Fraction")
    axes[0].set_title("Zero-Inflation Preservation")
    axes[0].set_xticks(x + width * len(methods) / 2)
    axes[0].set_xticklabels([v.capitalize() for v in zero_vars])
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Zero-fraction error
    for i, method_name in enumerate(methods):
        result = next(r for r in results if r.method == method_name)
        errors = [result.zero_fraction_error[var] for var in zero_vars]
        axes[1].bar(x + i * width, errors, width, label=method_name, alpha=0.8)

    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("Zero-Fraction Error")
    axes[1].set_xticks(x + width * (len(methods) - 1) / 2)
    axes[1].set_xticklabels([v.capitalize() for v in zero_vars])
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "zero_inflation.png", dpi=300, bbox_inches="tight")
    print(f"Saved zero-inflation comparison to {output_dir / 'zero_inflation.png'}")
    plt.close()

    # 4. Timing comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training time
    train_times = [r.train_time for r in results]
    ax1.bar(methods, train_times, color="steelblue")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Training Time")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # Generation time
    gen_times = [r.generate_time for r in results]
    ax2.bar(methods, gen_times, color="coral")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Generation Time")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "timing.png", dpi=300, bbox_inches="tight")
    print(f"Saved timing comparison to {output_dir / 'timing.png'}")
    plt.close()


def save_results_table(results: list, output_dir: Path):
    """Save results as CSV and markdown table."""

    # Create DataFrame
    rows = []
    for r in results:
        rows.append({
            "Method": r.method,
            "Mean KS": f"{r.mean_ks:.4f}",
            "Corr Error": f"{r.correlation_error:.4f}",
            "Zero Error": f"{r.mean_zero_error:.4f}",
            "Train Time (s)": f"{r.train_time:.1f}",
            "Gen Time (s)": f"{r.generate_time:.1f}",
        })

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results table to {csv_path}")

    # Save as markdown
    md_path = output_dir / "results.md"
    with open(md_path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Metrics Explanation\n\n")
        f.write("- **Mean KS**: Average Kolmogorov-Smirnov statistic across all target variables (lower is better)\n")
        f.write("- **Corr Error**: Frobenius norm of correlation matrix difference (lower is better)\n")
        f.write("- **Zero Error**: Mean absolute error in zero-fractions for zero-inflated variables (lower is better)\n")
        f.write("- **Train Time**: Time to train the model in seconds\n")
        f.write("- **Gen Time**: Time to generate synthetic samples in seconds\n")

    print(f"Saved results markdown to {md_path}")

    # Print to console
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


def main():
    """Run comprehensive benchmark suite."""

    print("=" * 80)
    print("MICROPLEX BENCHMARK SUITE")
    print("=" * 80)

    # Configuration
    n_train = 10000
    n_test = 2000
    epochs = 50  # Reduced for faster benchmarking

    target_vars = ["income", "assets", "debt", "savings"]
    condition_vars = ["age", "education", "region"]

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate data
    print(f"\nGenerating realistic microdata...")
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")

    full_data = generate_realistic_microdata(n_train + n_test, seed=42)
    train_data = full_data.iloc[:n_train].copy()
    test_data = full_data.iloc[n_train:].copy()
    test_conditions = test_data[condition_vars].copy()

    # Print data statistics
    print("\nData statistics:")
    print(f"  Income: ${train_data['income'].mean():,.0f} ± ${train_data['income'].std():,.0f}")
    print(f"  Assets: ${train_data['assets'].mean():,.0f} (zero-fraction: {(train_data['assets'] == 0).mean():.1%})")
    print(f"  Debt: ${train_data['debt'].mean():,.0f} (zero-fraction: {(train_data['debt'] == 0).mean():.1%})")
    print(f"  Savings: ${train_data['savings'].mean():,.0f}")

    # Save data
    train_data.to_csv(output_dir / "train_data.csv", index=False)
    test_data.to_csv(output_dir / "test_data.csv", index=False)
    print(f"\nSaved training and test data to {output_dir}")

    # Run benchmarks
    benchmarks = {
        "micro": MicroBenchmark,
        "ctgan": CTGANBenchmark,
        "tvae": TVAEBenchmark,
        "copula": GaussianCopulaBenchmark,
    }

    results = []
    synthetic_data = {}

    for method_name, benchmark_cls in benchmarks.items():
        try:
            result, synthetic = run_single_benchmark(
                method_name,
                benchmark_cls,
                train_data,
                test_conditions,
                target_vars,
                condition_vars,
                epochs=epochs,
            )
            results.append(result)
            synthetic_data[method_name] = synthetic
        except Exception as e:
            print(f"\nERROR: {method_name} benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("\nERROR: All benchmarks failed!")
        return

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    create_visualizations(results, train_data, synthetic_data, output_dir)

    # Save results
    save_results_table(results, output_dir)

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Best method for each metric
    best_ks = min(results, key=lambda r: r.mean_ks)
    print(f"\nBest marginal fidelity: {best_ks.method} (KS: {best_ks.mean_ks:.4f})")

    best_corr = min(results, key=lambda r: r.correlation_error)
    print(f"Best correlation preservation: {best_corr.method} (error: {best_corr.correlation_error:.4f})")

    best_zero = min(results, key=lambda r: r.mean_zero_error)
    print(f"Best zero-inflation handling: {best_zero.method} (error: {best_zero.mean_zero_error:.4f})")

    fastest_train = min(results, key=lambda r: r.train_time)
    print(f"Fastest training: {fastest_train.method} ({fastest_train.train_time:.1f}s)")

    fastest_gen = min(results, key=lambda r: r.generate_time)
    print(f"Fastest generation: {fastest_gen.method} ({fastest_gen.generate_time:.1f}s)")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.csv: Summary table")
    print(f"  - results.md: Markdown report")
    print(f"  - summary_metrics.png: Overall comparison")
    print(f"  - distributions_*.png: Per-method distribution comparisons")
    print(f"  - zero_inflation.png: Zero-handling comparison")
    print(f"  - timing.png: Performance comparison")


if __name__ == "__main__":
    main()
