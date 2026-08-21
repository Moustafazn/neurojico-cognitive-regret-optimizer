"""
Experiment runner and visualization for NCRO.

Provides:
  - run_experiment()  — multi-run experiment with statistics
  - plot_results()    — convergence curves and search behavior metrics
  - plot_ablation()   — ablation study comparison plots
  - Results saved to JSON files and high-quality PNG images in results/ folder
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

from .optimizer import NCROOptimizer
from .benchmarks import BENCHMARK_FUNCTIONS


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment(
    function_name: str,
    dimension: int = 30,
    population_size: int = 30,
    max_iter: int = 1000,
    num_runs: int = 30,
    algorithm_name: str = "NCRO",
    optimizer_class=None,
    optimizer_kwargs: dict | None = None,
) -> dict:
    """
    Run an optimizer on a benchmark function for multiple independent runs.

    Parameters
    ----------
    function_name : str
        Name of the benchmark function (must be a key in BENCHMARK_FUNCTIONS).
    dimension : int
        Dimensionality of the problem.
    population_size : int
        Number of agents.
    max_iter : int
        Maximum iterations per run.
    num_runs : int
        Number of independent runs for statistical evaluation.
    algorithm_name : str
        Name of the algorithm (for display and file naming).
    optimizer_class : class, optional
        Optimizer class to use. Defaults to NCROOptimizer.
    optimizer_kwargs : dict, optional
        Extra keyword arguments for the optimizer.

    Returns
    -------
    dict
        Dictionary containing aggregated performance metrics.
    """
    _ensure_results_dir()

    if optimizer_class is None:
        optimizer_class = NCROOptimizer
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    bench = BENCHMARK_FUNCTIONS[function_name]
    func = bench["function"]
    bounds = bench["bounds"]
    optimal = bench["optimal"]

    # Accumulators
    all_best_fitness = []
    all_convergence = []
    all_exploration_ratios = []
    all_exploitation_ratios = []
    all_cf_success_rates = []
    all_avg_regrets = []

    print(f"\n{'=' * 65}")
    print(f"  Running {algorithm_name} on {function_name}")
    print(f"  D={dimension}  N={population_size}  T={max_iter}  Runs={num_runs}")
    print(f"{'=' * 65}")

    for run in range(num_runs):

        optimizer = optimizer_class(
            objective_function=func,
            dimension=dimension,
            bounds=bounds,
            population_size=population_size,
            max_iter=max_iter,
            seed=run,
            **optimizer_kwargs,
        )

        result = optimizer.optimize()

        all_best_fitness.append(result.best_fitness)
        all_convergence.append(result.convergence_curve)

        # Compute per-run metrics
        # NCRO-family algorithms provide exploration/exploitation/regret metrics
        total_decisions = population_size * max_iter
        has_ncro_metrics = (
            hasattr(result, "exploration_counts")
            and result.exploration_counts is not None
            and np.any(result.exploration_counts > 0)
        )

        if has_ncro_metrics:
            er = float(np.sum(result.exploration_counts)) / total_decisions
            xr = float(np.sum(result.exploitation_counts)) / total_decisions
            # CSR = mean fraction of agents with CF success per iteration
            # counterfactual_success_counts[t] = number of agents with CF success at iter t
            csr = float(np.mean(result.counterfactual_success_counts / population_size))
            avg_regret = float(np.mean(result.regret_values))
        else:
            er = 0.0
            xr = 0.0
            csr = 0.0
            avg_regret = 0.0

        all_exploration_ratios.append(er)
        all_exploitation_ratios.append(xr)
        all_cf_success_rates.append(csr)
        all_avg_regrets.append(avg_regret)

        # Progress reporting
        if (run + 1) % 5 == 0 or run == 0:
            print(f"  Run {run + 1:3d}/{num_runs}:  Best = {result.best_fitness:.6e}")

    # Aggregate statistics
    best_fitnesses = np.array(all_best_fitness)
    convergence_curves = np.array(all_convergence)

    results = {
        "algorithm": algorithm_name,
        "function": function_name,
        "dimension": dimension,
        "population_size": population_size,
        "max_iter": max_iter,
        "num_runs": num_runs,
        "best": float(np.min(best_fitnesses)),
        "worst": float(np.max(best_fitnesses)),
        "mean": float(np.mean(best_fitnesses)),
        "std": float(np.std(best_fitnesses, ddof=1) if len(best_fitnesses) > 1 else 0.0),
        "median": float(np.median(best_fitnesses)),
        "accuracy": float(abs(np.min(best_fitnesses) - optimal)),
        "convergence_mean": np.mean(convergence_curves, axis=0),
        "convergence_std": np.std(convergence_curves, axis=0),
        "exploration_ratio": float(np.mean(all_exploration_ratios)),
        "exploitation_ratio": float(np.mean(all_exploitation_ratios)),
        "cf_success_rate": float(np.mean(all_cf_success_rates)),
        "avg_regret": float(np.mean(all_avg_regrets)),
        "all_best_fitness": best_fitnesses,
    }

    # Print summary — show NCRO metrics for any NCRO variant
    is_ncro = algorithm_name.startswith("NCRO")
    print(f"\n  -- {algorithm_name} on {function_name} Summary --")
    print(f"  Best Fitness : {results['best']:.6e}")
    print(f"  Mean Fitness : {results['mean']:.6e}")
    print(f"  Std Dev      : {results['std']:.6e}")
    print(f"  Accuracy     : {results['accuracy']:.6e}")
    if is_ncro:
        print(f"  Exploration  : {results['exploration_ratio']:.4f}")
        print(f"  Exploitation : {results['exploitation_ratio']:.4f}")
        print(f"  CSR          : {results['cf_success_rate']:.4f}")
        print(f"  Avg Regret   : {results['avg_regret']:.6e}")

    return results


def save_results_json(results: dict, algorithm_name: str, function_name: str) -> None:
    """Save experiment results to a JSON file in the results/ folder."""
    _ensure_results_dir()

    # Prepare JSON-serializable data (no numpy arrays)
    json_data = {
        "algorithm": results["algorithm"],
        "function": results["function"],
        "dimension": results["dimension"],
        "population_size": results["population_size"],
        "max_iter": results["max_iter"],
        "num_runs": results["num_runs"],
        "best": results["best"],
        "worst": results["worst"],
        "mean": results["mean"],
        "std": results["std"],
        "median": results["median"],
        "accuracy": results["accuracy"],
        "exploration_ratio": results["exploration_ratio"],
        "exploitation_ratio": results["exploitation_ratio"],
        "cf_success_rate": results["cf_success_rate"],
        "avg_regret": results["avg_regret"],
        "convergence_mean": results["convergence_mean"].tolist(),
        "all_best_fitness": results["all_best_fitness"].tolist(),
    }

    # Clean filename (replace spaces, special chars)
    safe_alg = algorithm_name.lower().replace(" ", "_").replace("-", "_")
    safe_func = function_name.lower().replace(" ", "_")
    filename = f"{safe_alg}_{safe_func}_results.json"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Results saved -> {filepath}")


def plot_results(results: dict, function_name: str, algorithm_name: str = "NCRO") -> None:
    """
    Generate and save high-quality visualization plots.

    Creates a figure with:
      - Left:  Convergence curve (log scale) with std-dev band
      - Right: Search-behavior metrics bar chart (ER, XR, CSR)

    Saves to results/ folder as high-quality PNG (300 DPI).
    """
    _ensure_results_dir()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{algorithm_name} Performance on {function_name}",
        fontsize=14,
        fontweight="bold",
    )

    # -- Convergence Curve --
    ax1 = axes[0]
    iters = np.arange(1, len(results["convergence_mean"]) + 1)
    mean_c = results["convergence_mean"]
    std_c = results["convergence_std"]

    # Protect against zero/negative values for log scale
    mean_c_safe = np.maximum(mean_c, 1e-30)
    ax1.semilogy(iters, mean_c_safe, "b-", linewidth=1.5, label="Mean")
    ax1.fill_between(
        iters,
        np.maximum(mean_c - std_c, 1e-30),
        mean_c + std_c,
        alpha=0.2,
        color="blue",
    )
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Best Fitness (log scale)", fontsize=11)
    ax1.set_title("Convergence Curve", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # -- Search Behavior Metrics --
    ax2 = axes[1]
    metrics = {
        "ER\n(Exploration)": results["exploration_ratio"],
        "XR\n(Exploitation)": results["exploitation_ratio"],
        "CSR\n(CF Success)": results["cf_success_rate"],
    }
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    bars = ax2.bar(metrics.keys(), metrics.values(), color=colors, width=0.5)
    ax2.set_ylabel("Ratio", fontsize=11)
    ax2.set_title("Search Behavior Metrics", fontsize=12)
    ax2.set_ylim(0, 1.0)
    for bar, val in zip(bars, metrics.values()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    safe_alg = algorithm_name.lower().replace(" ", "_").replace("-", "_")
    safe_func = function_name.lower().replace(" ", "_")
    filename = f"{safe_alg}_{safe_func}_results.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved -> {filepath}")


def plot_comparison(
    all_results: dict,
    function_name: str,
    algorithm_names: list[str],
) -> None:
    """
    Plot convergence curves of multiple algorithms on the same function.
    """
    _ensure_results_dir()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(
        f"Algorithm Comparison on {function_name}",
        fontsize=14,
        fontweight="bold",
    )

    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00ACC1", "#D81B60", "#546E7A", "#FFB300", "#00897B",
    ]

    for i, alg_name in enumerate(algorithm_names):
        if alg_name in all_results:
            res = all_results[alg_name]
            mean_c = np.maximum(res["convergence_mean"], 1e-30)
            iters = np.arange(1, len(mean_c) + 1)
            color = colors[i % len(colors)]
            ax.semilogy(iters, mean_c, linewidth=1.5, label=alg_name, color=color)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best Fitness (log scale)", fontsize=11)
    ax.set_title("Convergence Comparison", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_func = function_name.lower().replace(" ", "_")
    filename = f"comparison_{safe_func}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Comparison plot saved -> {filepath}")


def plot_ablation(
    ablation_results: dict,
    function_name: str,
    variant_names: list[str],
) -> None:
    """
    Plot ablation study: convergence curves + bar chart of final fitness.

    Parameters
    ----------
    ablation_results : dict
        Maps variant_name -> results dict.
    function_name : str
        Name of the benchmark function.
    variant_names : list[str]
        List of variant names to plot.
    """
    _ensure_results_dir()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Ablation Study on {function_name}",
        fontsize=14,
        fontweight="bold",
    )

    colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1",
        "#D81B60", "#546E7A", "#FFB300", "#00897B",
    ]

    # -- Convergence curves --
    ax1 = axes[0]
    for i, name in enumerate(variant_names):
        if name in ablation_results:
            res = ablation_results[name]
            mean_c = np.maximum(res["convergence_mean"], 1e-30)
            iters = np.arange(1, len(mean_c) + 1)
            color = colors[i % len(colors)]
            short = name.replace("NCRO (Full)", "Full").replace("NCRO_", "")
            ax1.semilogy(iters, mean_c, linewidth=2, label=short, color=color)
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Best Fitness (log scale)", fontsize=11)
    ax1.set_title("Convergence Comparison", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # -- Bar chart of mean final fitness --
    ax2 = axes[1]
    means = []
    labels = []
    bar_colors = []
    for i, name in enumerate(variant_names):
        if name in ablation_results:
            means.append(ablation_results[name]["mean"])
            labels.append(name.replace("NCRO (Full)", "Full").replace("NCRO_", "No\n"))
            bar_colors.append(colors[i % len(colors)])

    bars = ax2.bar(labels, means, color=bar_colors, width=0.6)
    ax2.set_ylabel("Mean Best Fitness", fontsize=11)
    ax2.set_title("Final Performance", fontsize=12)
    ax2.set_yscale("log")
    for bar, val in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{val:.2e}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    safe_func = function_name.lower().replace(" ", "_")
    filename = f"ablation_{safe_func}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Ablation plot saved -> {filepath}")


def save_summary_json(all_results: dict) -> None:
    """
    Save a final summary JSON with results from all algorithms and functions.

    Parameters
    ----------
    all_results : dict
        Nested dict: all_results[function_name][algorithm_name] -> results dict
    """
    _ensure_results_dir()

    summary = {}
    for func_name, alg_dict in all_results.items():
        summary[func_name] = {}
        for alg_name, res in alg_dict.items():
            summary[func_name][alg_name] = {
                "best": res["best"],
                "worst": res["worst"],
                "mean": res["mean"],
                "std": res["std"],
                "median": res["median"],
                "accuracy": res["accuracy"],
            }

    filepath = os.path.join(RESULTS_DIR, "final_summary.json")
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Final summary saved -> {filepath}")
