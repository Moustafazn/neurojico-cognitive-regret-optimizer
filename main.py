# ==================================================================
#  NCRO: Full Experiment Suite
# ==================================================================
#  Part 1: Benchmarks NCRO against SOTA algorithms
#  Part 2: Ablation study (NCRO Full vs 4 ablated variants)
#
#  Tests on 8 benchmark functions × 3 dimensions (10, 30, 50)
#  NCRO uses 3N FEs/iter. Fair comparison: others get 3× iterations.
#
#  Output structure (research-paper ready):
#    Tables (CSV):
#      - benchmark_comparison.csv     All algorithms × functions × dims
#      - ablation_study.csv           All ablation variants
#      - friedman_rankings.csv        Average Friedman rankings
#      - statistical_tests.csv        Wilcoxon p-values + significance
#
#    Plots (key representatives only):
#      - comparison_{func}_D{d}.png   Convergence for key functions × D=30
#      - ablation_{func}_D{d}.png     Ablation for key functions × D=30
#      - ranking_summary.png          Bar chart of avg Friedman rankings
# ==================================================================

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ncro import run_experiment
from ncro.experiment import plot_comparison, plot_ablation, RESULTS_DIR
from ncro.comparisons import (
    PSOOptimizer, DEOptimizer,
    GWOOptimizer, WOAOptimizer,
    SHADEOptimizer, LSHADEOptimizer,
    jSOOptimizer, NLSHADELBCOptimizer, LSRTDEOptimizer,
)
from ncro.ablation import (
    NCRO_NoRegret, NCRO_NoCounterfactual,
    NCRO_NoAdaptiveEE, NCRO_NoRegretMemory,
)
from ncro.statistics import (
    run_statistical_analysis, save_statistical_results,
    wilcoxon_signedrank_test, friedman_test,
)


# ──────────────────────────────────────────────────────────────
# CSV output helpers
# ──────────────────────────────────────────────────────────────

def save_comparison_csv(all_results, funcs, dims, alg_names):
    """Table 1: Complete benchmark comparison (Mean ± Std)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fp = os.path.join(RESULTS_DIR, "benchmark_comparison.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Function", "Dim", "Algorithm", "Best", "Worst",
                     "Mean", "Std", "Median", "Accuracy"])
        for D in dims:
            for fn in funcs:
                key = f"{fn}_D{D}"
                if key not in all_results: continue
                for alg in alg_names:
                    if alg not in all_results[key]: continue
                    r = all_results[key][alg]
                    w.writerow([fn, D, alg, f"{r['best']:.6e}", f"{r['worst']:.6e}",
                                f"{r['mean']:.6e}", f"{r['std']:.6e}",
                                f"{r['median']:.6e}", f"{r['accuracy']:.6e}"])
    print(f"  Table saved -> {fp}")


def save_ablation_csv(all_ablation, funcs, dims, var_names):
    """Table 2: Ablation study (Mean ± Std + NCRO metrics)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fp = os.path.join(RESULTS_DIR, "ablation_study.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Function", "Dim", "Variant", "Best", "Worst",
                     "Mean", "Std", "Median", "ER", "XR", "CSR", "AvgRegret"])
        for D in dims:
            for fn in funcs:
                key = f"{fn}_D{D}"
                if key not in all_ablation: continue
                for var in var_names:
                    if var not in all_ablation[key]: continue
                    r = all_ablation[key][var]
                    w.writerow([fn, D, var, f"{r['best']:.6e}", f"{r['worst']:.6e}",
                                f"{r['mean']:.6e}", f"{r['std']:.6e}",
                                f"{r['median']:.6e}",
                                f"{r['exploration_ratio']:.4f}",
                                f"{r['exploitation_ratio']:.4f}",
                                f"{r['cf_success_rate']:.4f}",
                                f"{r['avg_regret']:.6e}"])
    print(f"  Table saved -> {fp}")


def save_rankings_csv(all_results, funcs, dims, alg_names):
    """Table 3: Friedman average rankings per function×dim + overall."""
    from scipy import stats as sp_stats
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fp = os.path.join(RESULTS_DIR, "friedman_rankings.csv")

    all_ranks = {alg: [] for alg in alg_names}

    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Function", "Dim"] + alg_names + ["Friedman_p"])
        for D in dims:
            for fn in funcs:
                key = f"{fn}_D{D}"
                if key not in all_results: continue
                res = all_results[key]
                # Gather data for Friedman
                data = []
                valid = []
                for alg in alg_names:
                    if alg in res:
                        data.append(np.asarray(res[alg]["all_best_fitness"]))
                        valid.append(alg)
                if len(data) < 3: continue

                min_len = min(len(d) for d in data)
                data = [d[:min_len] for d in data]
                try:
                    _, p_val = sp_stats.friedmanchisquare(*data)
                except: p_val = 1.0

                # Compute per-run ranks
                n_runs = min_len
                ranks = np.zeros((n_runs, len(valid)))
                for run in range(n_runs):
                    vals = [data[a][run] for a in range(len(valid))]
                    ranks[run] = sp_stats.rankdata(vals)
                avg_ranks = np.mean(ranks, axis=0)

                row = [fn, D]
                rank_dict = dict(zip(valid, avg_ranks))
                for alg in alg_names:
                    if alg in rank_dict:
                        row.append(f"{rank_dict[alg]:.2f}")
                        all_ranks[alg].append(rank_dict[alg])
                    else:
                        row.append("—")
                row.append(f"{p_val:.4e}")
                w.writerow(row)

        # Overall average ranking
        w.writerow([])
        row = ["OVERALL", "Avg"]
        for alg in alg_names:
            if all_ranks[alg]:
                row.append(f"{np.mean(all_ranks[alg]):.2f}")
            else:
                row.append("—")
        row.append("")
        w.writerow(row)

    print(f"  Rankings saved -> {fp}")
    return all_ranks


def save_statistical_csv(all_results, funcs, dims, alg_names, ref="NCRO"):
    """Table 4: Wilcoxon signed-rank p-values + win/loss/tie."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fp = os.path.join(RESULTS_DIR, "statistical_tests.csv")

    wins = {alg: 0 for alg in alg_names if alg != ref}
    losses = {alg: 0 for alg in alg_names if alg != ref}
    ties = {alg: 0 for alg in alg_names if alg != ref}

    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Function", "Dim", "Comparison", "p_value", "Significant", "Winner"])
        for D in dims:
            for fn in funcs:
                key = f"{fn}_D{D}"
                if key not in all_results or ref not in all_results[key]: continue
                for alg in alg_names:
                    if alg == ref or alg not in all_results[key]: continue
                    test = wilcoxon_signedrank_test(
                        all_results[key][ref], all_results[key][alg])
                    sig = "Yes" if test["significant"] else "No"
                    w.writerow([fn, D, f"{ref} vs {alg}",
                                f"{test['p_value']:.4e}", sig, test["winner"]])
                    if test["significant"]:
                        if test["winner"] == ref: wins[alg] += 1
                        else: losses[alg] += 1
                    else:
                        ties[alg] += 1

        # Win/Loss/Tie summary
        w.writerow([])
        w.writerow(["SUMMARY", "", "Opponent", "NCRO Wins", "NCRO Losses", "Ties"])
        for alg in alg_names:
            if alg == ref: continue
            w.writerow(["", "", alg, wins[alg], losses[alg], ties[alg]])

    print(f"  Stats saved -> {fp}")


def plot_ranking_summary(all_ranks, alg_names):
    """Bar chart of overall average Friedman rankings."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    names = [a for a in alg_names if all_ranks.get(a)]
    avg = [np.mean(all_ranks[a]) for a in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#E53935" if "NCRO" in n else "#1E88E5" for n in names]
    bars = ax.bar(names, avg, color=colors, width=0.6)
    ax.set_ylabel("Average Friedman Rank (lower is better)", fontsize=11)
    ax.set_title("Overall Algorithm Rankings Across All Functions & Dimensions", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, avg):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fp = os.path.join(RESULTS_DIR, "ranking_summary.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Ranking plot saved -> {fp}")


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":

    POPULATION_SIZE = 30
    NCRO_ITER = 500
    NUM_RUNS = 30

    benchmark_functions = [
        "Sphere", "Rastrigin", "Rosenbrock", "Ackley",
        "Griewank", "Schwefel", "Zakharov", "Levy",
    ]
    dimensions = [10, 30, 50]

    # Key functions for convergence plots (not all 8×3=24)
    KEY_PLOT_FUNCTIONS = ["Sphere", "Rastrigin", "Ackley", "Rosenbrock"]
    KEY_PLOT_DIM = 30

    algorithms = {
        "NCRO":         {"class": None, "iter_mult": 1},
        "PSO":          {"class": PSOOptimizer, "iter_mult": 3},
        "DE":           {"class": DEOptimizer, "iter_mult": 3},
        "GWO":          {"class": GWOOptimizer, "iter_mult": 3},
        "WOA":          {"class": WOAOptimizer, "iter_mult": 3},
        "SHADE":        {"class": SHADEOptimizer, "iter_mult": 3},
        "L-SHADE":      {"class": LSHADEOptimizer, "iter_mult": 3},
        "jSO":          {"class": jSOOptimizer, "iter_mult": 3},
        "NL-SHADE-LBC": {"class": NLSHADELBCOptimizer, "iter_mult": 3},
        "L-SRTDE":      {"class": LSRTDEOptimizer, "iter_mult": 3},
    }

    ablation_variants = {
        "NCRO (Full)":    {"class": None, "iter": NCRO_ITER},
        "NCRO_NoRegret":  {"class": NCRO_NoRegret, "iter": NCRO_ITER},
        "NCRO_NoCF":      {"class": NCRO_NoCounterfactual, "iter": NCRO_ITER},
        "NCRO_NoAdaptEE": {"class": NCRO_NoAdaptiveEE, "iter": NCRO_ITER},
        "NCRO_NoMemory":  {"class": NCRO_NoRegretMemory, "iter": NCRO_ITER},
    }

    algorithm_names = list(algorithms.keys())
    ablation_names = list(ablation_variants.keys())
    all_results = {}
    all_stats = {}

    # ==============================================================
    # PART 1: BENCHMARK COMPARISON
    # ==============================================================
    print("\n" + "#" * 80)
    print("  PART 1: BENCHMARK COMPARISON")
    print("#" * 80)

    for D in dimensions:
        for function_name in benchmark_functions:
            key = f"{function_name}_D{D}"
            all_results[key] = {}

            for alg_name, cfg in algorithms.items():
                iters = NCRO_ITER * cfg["iter_mult"]
                results = run_experiment(
                    function_name=function_name, dimension=D,
                    population_size=POPULATION_SIZE, max_iter=iters,
                    num_runs=NUM_RUNS, algorithm_name=alg_name,
                    optimizer_class=cfg["class"],
                )
                all_results[key][alg_name] = results

            # Plot ONLY for key representative functions at key dimension
            if function_name in KEY_PLOT_FUNCTIONS and D == KEY_PLOT_DIM:
                plot_comparison(all_results[key], f"{function_name}_D{D}", algorithm_names)

    # ==============================================================
    # PART 2: ABLATION STUDY
    # ==============================================================
    print("\n\n" + "#" * 80)
    print("  PART 2: ABLATION STUDY")
    print("#" * 80)

    all_ablation = {}
    for D in dimensions:
        for function_name in benchmark_functions:
            key = f"{function_name}_D{D}"
            all_ablation[key] = {}

            for var_name, cfg in ablation_variants.items():
                results = run_experiment(
                    function_name=function_name, dimension=D,
                    population_size=POPULATION_SIZE, max_iter=cfg["iter"],
                    num_runs=NUM_RUNS, algorithm_name=var_name,
                    optimizer_class=cfg["class"],
                )
                all_ablation[key][var_name] = results

            # Plot ONLY for key representative functions at key dimension
            if function_name in KEY_PLOT_FUNCTIONS and D == KEY_PLOT_DIM:
                plot_ablation(all_ablation[key], f"{function_name}_D{D}", ablation_names)

    # ==============================================================
    # SAVE ALL RESULTS
    # ==============================================================
    print("\n\n" + "#" * 80)
    print("  SAVING RESULTS")
    print("#" * 80)

    # Table 1: Benchmark comparison CSV
    save_comparison_csv(all_results, benchmark_functions, dimensions, algorithm_names)

    # Table 2: Ablation study CSV
    save_ablation_csv(all_ablation, benchmark_functions, dimensions, ablation_names)

    # Table 3: Friedman rankings CSV
    all_ranks = save_rankings_csv(all_results, benchmark_functions, dimensions, algorithm_names)

    # Table 4: Statistical tests CSV (Wilcoxon p-values + W/L/T)
    save_statistical_csv(all_results, benchmark_functions, dimensions, algorithm_names, ref="NCRO")

    # Plot: Ranking summary bar chart
    plot_ranking_summary(all_ranks, algorithm_names)

    # JSON: Statistical analysis (for reference)
    for D in dimensions:
        for fn in benchmark_functions:
            key = f"{fn}_D{D}"
            if key in all_results:
                stat = run_statistical_analysis(
                    all_results[key], f"{fn}_D{D}", algorithm_names, "NCRO")
                all_stats[key] = stat
            abl_key = f"ablation_{key}"
            if key in all_ablation:
                abl_stat = run_statistical_analysis(
                    all_ablation[key], f"ablation_{fn}_D{D}", ablation_names, "NCRO (Full)")
                all_stats[abl_key] = abl_stat
    save_statistical_results(all_stats)

    # ==============================================================
    # PRINT FINAL TABLES
    # ==============================================================
    print("\n\n" + "=" * 160)
    print("BENCHMARK COMPARISON — Mean Fitness (lower is better)")
    print("=" * 160)
    for D in dimensions:
        print(f"\n--- D = {D} ---")
        header = f"{'Function':<14}"
        for alg in algorithm_names: header += f"{alg:>14}"
        print(header)
        print("-" * 154)
        for fn in benchmark_functions:
            key = f"{fn}_D{D}"
            row = f"{fn:<14}"
            for alg in algorithm_names:
                if alg in all_results.get(key, {}):
                    row += f"{all_results[key][alg]['mean']:>14.2e}"
                else: row += f"{'—':>14}"
            print(row)

    print("\n\n" + "=" * 100)
    print("ABLATION STUDY — Mean Fitness")
    print("=" * 100)
    for D in dimensions:
        print(f"\n--- D = {D} ---")
        header = f"{'Function':<14}"
        for var in ablation_names:
            short = var.replace("NCRO (Full)", "Full").replace("NCRO_", "")
            header += f"{short:>17}"
        print(header)
        print("-" * 99)
        for fn in benchmark_functions:
            key = f"{fn}_D{D}"
            row = f"{fn:<14}"
            for var in ablation_names:
                if var in all_ablation.get(key, {}):
                    row += f"{all_ablation[key][var]['mean']:>17.2e}"
                else: row += f"{'—':>17}"
            print(row)

    print("\n\n" + "=" * 80)
    print("OVERALL FRIEDMAN RANKINGS (lower rank = better)")
    print("=" * 80)
    for alg in algorithm_names:
        if all_ranks.get(alg):
            print(f"  {alg:<18}: {np.mean(all_ranks[alg]):.2f}")

    print("\n" + "=" * 80)
    print("\nResults saved to results/ folder:")
    print("  CSV Tables:")
    print("    benchmark_comparison.csv   — All algorithms × functions × dims")
    print("    ablation_study.csv         — All ablation variants")
    print("    friedman_rankings.csv      — Average rankings per test case")
    print("    statistical_tests.csv      — Wilcoxon p-values + win/loss/tie")
    print("  Plots:")
    print("    comparison_*.png           — Convergence for key functions (D=30)")
    print("    ablation_*.png             — Ablation for key functions (D=30)")
    print("    ranking_summary.png        — Overall Friedman ranking bar chart")
