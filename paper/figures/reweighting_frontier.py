#!/usr/bin/env python3
"""Generate reweighting frontier figure: records used vs out-of-sample error.

Reads reweighting_frontier.json and produces a figure showing the
accuracy-sparsity tradeoff for each reweighting method family.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(__file__).parent.parent.parent / "benchmarks" / "results" / "reweighting_frontier.json"
OUTPUT_PATH = Path(__file__).parent / "reweighting_frontier.pdf"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    n_records = data["n_records"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # --- SparseCalibrator frontier ---
    sc = data["methods"]["SparseCalibrator"]
    sc_x = [p["n_active"] for p in sc]
    sc_y = [p["test_error"] for p in sc]
    # Sort by n_active
    order = np.argsort(sc_x)
    sc_x = [sc_x[i] for i in order]
    sc_y = [sc_y[i] for i in order]
    ax.plot(sc_x, sc_y, "s-", color="#2ca02c", linewidth=1.5, markersize=5,
            label="SparseCalibrator", zorder=3)

    # --- HardConcrete frontier ---
    hc = data["methods"]["HardConcrete"]
    hc_x = [p["n_active"] for p in hc]
    hc_y = [p["test_error"] for p in hc]
    order = np.argsort(hc_x)
    hc_x = [hc_x[i] for i in order]
    hc_y = [hc_y[i] for i in order]
    ax.plot(hc_x, hc_y, "o-", color="#1f77b4", linewidth=1.5, markersize=5,
            label="HardConcrete", zorder=3)

    # --- Dense methods as reference points ---
    for name, marker, color in [
        ("IPF", "D", "#d62728"),
        ("Entropy", "^", "#ff7f0e"),
    ]:
        pts = data["methods"][name]
        for p in pts:
            ax.scatter(p["n_active"], p["test_error"], marker=marker, color=color,
                       s=80, zorder=4, label=name, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Active records (non-zero weight)", fontsize=11)
    ax.set_ylabel("Out-of-sample error\n(held-out sex margin)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(5, n_records * 1.2)
    ax.set_ylim(0, max(
        max(p["test_error"] for p in sc),
        max(p["test_error"] for p in hc),
        max(p["test_error"] for pts in [data["methods"]["IPF"], data["methods"]["Entropy"]] for p in pts),
    ) * 1.15)

    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add annotation for total records
    ax.axvline(n_records, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(n_records * 0.85, ax.get_ylim()[1] * 0.95, f"N={n_records:,}",
            ha="right", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH} and {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
