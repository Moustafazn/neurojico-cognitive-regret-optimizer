#!/usr/bin/env python3
"""
Generate supplementary figures for the NCRO manuscript.

Produces:
  results/figures/variant_comparison.png   — Section 8.6
  results/figures/scalability_plot.png     — Section 9.1

Reads from:
  results/tables/variant_summary.csv
  results/tables/scalability_analysis.csv

Style matches the existing ablation figures (experiment.py):
  figsize=(10, 6), dpi=300, fontsize conventions, same colour palette.

Usage:
  python generate_figures.py
"""

import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#D81B60", "#546E7A",
]

VARIANT_LABELS = {
    "full": "Full (coeff-swap)",
    "random": "Random",
    "obl": "OBL",
    "swap_no_regret": "Swap (no regret)",
}


def load_csv(filename):
    fp = os.path.join(TABLES_DIR, filename)
    if not os.path.exists(fp):
        print(f"  WARNING: {fp} not found")
        return []
    with open(fp, newline="") as f:
        return list(csv.DictReader(f))


# ──────────────────────────────────────────────────────────────
# Figure 9: Variant comparison (Section 8.6)
# ──────────────────────────────────────────────────────────────

def generate_variant_comparison():
    """Grouped bar chart: mean fitness per variant on Sphere and Schwefel."""
    rows = load_csv("variant_summary.csv")
    if not rows:
        return

    # Only Sphere and Schwefel show meaningful variant differences
    target_functions = ["Sphere", "Schwefel"]
    variant_order = ["full", "random", "obl", "swap_no_regret"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, func in zip(axes, target_functions):
        func_rows = {r["Variant"]: r for r in rows
                     if r["Function"] == func}
        if not func_rows:
            continue

        means = []
        stds = []
        labels = []
        for v in variant_order:
            if v in func_rows:
                means.append(float(func_rows[v]["Mean"]))
                stds.append(float(func_rows[v]["Std"]))
                labels.append(VARIANT_LABELS.get(v, v))

        x = np.arange(len(labels))
        bar_colors = COLORS[:len(labels)]
        bars = ax.bar(x, means, width=0.6, color=bar_colors,
                      edgecolor="black", linewidth=0.5)

        # Value annotations
        for bar, val in zip(bars, means):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        height * 1.05 if func == "Schwefel" else height * 1.8,
                        f"{val:.2e}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel("Mean Fitness", fontsize=11)
        ax.set_title(f"{func} (D = 30)", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        if func == "Sphere":
            ax.set_yscale("log")

    plt.tight_layout()
    fp = os.path.join(FIGURES_DIR, "variant_comparison.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {fp}")


# ──────────────────────────────────────────────────────────────
# Figure 10: Scalability across dimensions (Section 9.1)
# ──────────────────────────────────────────────────────────────

def generate_scalability_plot():
    """Line plot: mean fitness vs dimension for each benchmark function."""
    rows = load_csv("scalability_analysis.csv")
    if not rows:
        return

    functions = ["Sphere", "Rastrigin", "Ackley", "Griewank"]
    func_data = {f: {"dims": [], "means": []} for f in functions}

    for r in rows:
        fn = r["Function"]
        if fn in func_data:
            func_data[fn]["dims"].append(int(r["Dim"]))
            func_data[fn]["means"].append(float(r["Mean"]))

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "D", "^"]
    for i, fn in enumerate(functions):
        d = func_data[fn]
        if not d["dims"]:
            continue
        # Sort by dimension
        order = np.argsort(d["dims"])
        dims = np.array(d["dims"])[order]
        means = np.array(d["means"])[order]

        # Offset zero values for log display
        plot_means = np.where(means == 0, 1e-320, means)

        ax.semilogy(dims, plot_means, linewidth=2, marker=markers[i],
                    markersize=8, label=fn, color=COLORS[i])

        # Annotate each point
        for dim, mean in zip(dims, means):
            if mean == 0:
                label_text = "0.0"
            else:
                label_text = f"{mean:.1e}"
            ax.annotate(label_text,
                        (dim, max(mean, 1e-320)),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=7, ha="center", fontweight="bold")

    ax.set_xlabel("Dimension (D)", fontsize=11)
    ax.set_ylabel("Mean Fitness (log scale)", fontsize=11)
    ax.set_title("NCRO Scalability Across Dimensions (30 runs)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks([10, 30, 50, 100])
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fp = os.path.join(FIGURES_DIR, "scalability_plot.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {fp}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating supplementary figures...")
    generate_variant_comparison()
    generate_scalability_plot()
    print("Done.")


if __name__ == "__main__":
    main()
