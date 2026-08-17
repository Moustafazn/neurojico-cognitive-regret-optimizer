"""
Statistical tests for comparing optimization algorithms.

Implements:
  - Wilcoxon signed-rank test (paired comparison — same seeds)
  - Friedman test (multi-algorithm comparison)
  - Summary table generation

Uses signed-rank (not rank-sum) because experiments are paired by seed.
"""

import json
import os
import numpy as np
from scipy import stats


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def wilcoxon_signedrank_test(
    results_a: dict,
    results_b: dict,
    alpha: float = 0.05,
) -> dict:
    """
    Perform Wilcoxon signed-rank test between two algorithms.

    Uses signed-rank (not rank-sum) because experiments are paired
    by seed — each run i uses the same random seed for both algorithms.

    Parameters
    ----------
    results_a, results_b : dict
        Results from run_experiment() containing 'all_best_fitness'.
    alpha : float
        Significance level.

    Returns
    -------
    dict with statistic, p_value, significant, winner
    """
    a = np.asarray(results_a["all_best_fitness"], dtype=float)
    b = np.asarray(results_b["all_best_fitness"], dtype=float)

    # Ensure same length (should be, but safety check)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    # Handle edge case: identical arrays
    diff = a - b
    if np.all(diff == 0):
        return {
            "algorithm_a": results_a["algorithm"],
            "algorithm_b": results_b["algorithm"],
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "winner": "No significant difference (identical results)",
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
        }

    try:
        stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        # Fallback if wilcoxon fails (e.g., all differences are zero)
        stat, p_value = 0.0, 1.0

    if p_value < alpha:
        winner = results_a["algorithm"] if np.mean(a) < np.mean(b) else results_b["algorithm"]
        significant = True
    else:
        winner = "No significant difference"
        significant = False

    return {
        "algorithm_a": results_a["algorithm"],
        "algorithm_b": results_b["algorithm"],
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(significant),
        "winner": winner,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
    }


def friedman_test(
    all_results: dict,
    algorithm_names: list[str],
) -> dict:
    """
    Perform Friedman test across multiple algorithms on one function.

    Parameters
    ----------
    all_results : dict
        Maps algorithm_name -> results dict.
    algorithm_names : list[str]
        Names of algorithms to compare.

    Returns
    -------
    dict with statistic, p_value, rankings
    """
    data = []
    valid_names = []
    for alg in algorithm_names:
        if alg in all_results:
            data.append(np.asarray(all_results[alg]["all_best_fitness"], dtype=float))
            valid_names.append(alg)

    if len(data) < 3:
        return {"error": "Need at least 3 algorithms for Friedman test"}

    # Friedman test requires same number of observations
    min_len = min(len(d) for d in data)
    if min_len < 2:
        return {"error": "Need at least 2 runs per algorithm"}
    data = [d[:min_len] for d in data]

    try:
        stat, p_value = stats.friedmanchisquare(*data)
    except Exception:
        stat, p_value = 0.0, 1.0

    # Compute average ranks
    n_runs = min_len
    n_algs = len(data)
    ranks = np.zeros((n_runs, n_algs))
    for run in range(n_runs):
        values = [data[a][run] for a in range(n_algs)]
        ranks[run] = stats.rankdata(values)

    avg_ranks = np.mean(ranks, axis=0)
    rankings = {valid_names[i]: float(avg_ranks[i]) for i in range(n_algs)}

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "rankings": rankings,
        "significant": bool(p_value < 0.05),
    }


def run_statistical_analysis(
    all_results: dict,
    function_name: str,
    algorithm_names: list[str],
    reference_algorithm: str = "NCRO",
) -> dict:
    """
    Run complete statistical analysis for one benchmark function.

    Parameters
    ----------
    all_results : dict
        Maps algorithm_name -> results dict.
    function_name : str
        Name of the function.
    algorithm_names : list[str]
        All algorithm names.
    reference_algorithm : str
        Algorithm to compare against (default: NCRO-V2).

    Returns
    -------
    dict with all statistical test results.
    """
    analysis = {
        "function": function_name,
        "pairwise_wilcoxon": [],
        "friedman": None,
    }

    # Pairwise Wilcoxon tests: reference vs each other algorithm
    if reference_algorithm in all_results:
        for alg in algorithm_names:
            if alg != reference_algorithm and alg in all_results:
                test_result = wilcoxon_signedrank_test(
                    all_results[reference_algorithm],
                    all_results[alg],
                )
                analysis["pairwise_wilcoxon"].append(test_result)

                sig_marker = "*" if test_result["significant"] else " "
                print(f"    {reference_algorithm} vs {alg}: "
                      f"p={test_result['p_value']:.4e} {sig_marker} "
                      f"Winner: {test_result['winner']}")

    # Friedman test across all algorithms
    if len([a for a in algorithm_names if a in all_results]) >= 3:
        friedman = friedman_test(all_results, algorithm_names)
        analysis["friedman"] = friedman
        if "error" not in friedman:
            print(f"    Friedman: chi2={friedman['statistic']:.4f}, "
                  f"p={friedman['p_value']:.4e}")
            if "rankings" in friedman:
                rank_str = ", ".join(f"{k}: {v:.2f}" for k, v in friedman["rankings"].items())
                print(f"    Rankings: {rank_str}")

    return analysis


def save_statistical_results(all_analyses: dict) -> None:
    """Save all statistical analysis results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    filepath = os.path.join(RESULTS_DIR, "statistical_analysis.json")
    with open(filepath, "w") as f:
        json.dump(all_analyses, f, indent=2)

    print(f"\n  Statistical analysis saved -> {filepath}")
