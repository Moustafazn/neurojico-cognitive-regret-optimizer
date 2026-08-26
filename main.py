# ==================================================================
#  NCRO: Full Experiment Suite (IEEE Journal Version)
# ==================================================================
#  Part 1: Benchmarks NCRO against comparison algorithms
#  Part 2: Ablation study (NCRO Full vs 5 ablated variants)
#  Part 3: Scalability analysis (D=10,30,50,100)
#  Part 4: Computational complexity & runtime evaluation
#
#  Tests on 8 benchmark functions × 4 dimensions (10, 30, 50, 100)
#  NCRO uses 3N FEs/iter. Fair comparison: others get 3× iterations.
#
#  Output structure (research-paper ready):
#    Tables (CSV):
#      - benchmark_comparison.csv     All algorithms × functions × dims
#      - ablation_study.csv           All ablation variants
#      - friedman_rankings.csv        Average Friedman rankings
#      - statistical_tests.csv        Wilcoxon p-values + significance
#      - scalability_analysis.csv     NCRO performance across D
#      - runtime_analysis.csv         Runtime per run & per iteration
#
#    Plots (key representatives only):
#      - comparison_{func}_D{d}.png   Convergence for key functions × D=30
#      - ablation_{func}_D{d}.png     Ablation for key functions × D=30
#      - ranking_summary.png          Bar chart of avg Friedman rankings
#      - scalability_convergence.png  NCRO convergence across dimensions
# ==================================================================

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ncro import run_experiment
from ncro.experiment import plot_comparison, plot_ablation, RESULTS_DIR
from ncro.comparisons import (
    PSOOptimizer, DEOptimizer,
    GWOOptimizer, WOAOptimizer,
    HHOOptimizer, ABCOptimizer, SCAOptimizer,
    SHADEOptimizer, LSHADEOptimizer,
    jSOOptimizer, NLSHADELBCOptimizer, LSRTDEOptimizer,
)
from ncro.ablation import (
    NCRO_NoRegret, NCRO_NoCFMem, NCRO_NoCounterfactual,
    NCRO_NoAdaptiveEE, NCRO_NoRegretMemory,
    NCRO_NoMomentum,
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


def save_runtime_csv(runtime_data):
    """Table 5: Runtime analysis — average time per run & per iteration."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fp = os.path.join(RESULTS_DIR, "runtime_analysis.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Algorithm", "Function", "Dim", "Avg_Runtime_sec",
                     "Avg_Time_Per_Iter_ms", "Iterations", "Num_Runs"])
        for entry in runtime_data:
            w.writerow([
                entry["algorithm"], entry["function"], entry["dim"],
                f"{entry['avg_runtime']:.4f}",
                f"{entry['avg_time_per_iter']*1000:.4f}",
                entry["iterations"], entry["num_runs"],
            ])
    print(f"  Runtime saved -> {fp}")


def save_scalability_csv(all_results, funcs, dims):
    """Table 6: NCRO scalability across dimensions."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fp = os.path.join(RESULTS_DIR, "scalability_analysis.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Function", "Dim", "Best", "Mean", "Std", "Median", "Worst"])
        for fn in funcs:
            for D in dims:
                key = f"{fn}_D{D}"
                if key not in all_results or "NCRO" not in all_results[key]:
                    continue
                r = all_results[key]["NCRO"]
                w.writerow([fn, D, f"{r['best']:.6e}", f"{r['mean']:.6e}",
                            f"{r['std']:.6e}", f"{r['median']:.6e}",
                            f"{r['worst']:.6e}"])
    print(f"  Scalability saved -> {fp}")


def plot_ranking_summary(all_ranks, alg_names):
    """Bar chart of overall average Friedman rankings. """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    names = [a for a in alg_names if all_ranks.get(a)]
    avg = [np.mean(all_ranks[a]) for a in names]

    # Save data to JSON
    import json as _json
    json_path = os.path.join(RESULTS_DIR, "ranking_summary_data.json")
    with open(json_path, "w") as f:
        _json.dump({"names": names, "avg_ranks": avg}, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 5))
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


def plot_scalability(all_results, funcs, dims):
    """Plot NCRO convergence curves across dimensions for scalability analysis."""
    import json as _json
    os.makedirs(RESULTS_DIR, exist_ok=True)
    key_funcs = ["Sphere", "Rastrigin", "Ackley", "Griewank"]
    suffix_letters = ["a", "b", "c", "d"]
    colors = {10: "#1E88E5", 30: "#43A047", 50: "#FB8C00", 100: "#E53935"}

    # Save scalability data to JSON for regeneration
    json_data = {"key_funcs": key_funcs, "dims": dims, "curves": {}}
    for fn in key_funcs:
        json_data["curves"][fn] = {}
        for D in dims:
            key = f"{fn}_D{D}"
            if key in all_results and "NCRO" in all_results[key]:
                res = all_results[key]["NCRO"]
                json_data["curves"][fn][str(D)] = {
                    "convergence_mean": [float(v) for v in res["convergence_mean"]],
                }
    json_path = os.path.join(RESULTS_DIR, "scalability_convergence_data.json")
    with open(json_path, "w") as f:
        _json.dump(json_data, f, indent=2)

    for idx, fn in enumerate(key_funcs):
        letter = suffix_letters[idx]
        fig, ax = plt.subplots(figsize=(8, 6))
        for D in dims:
            key = f"{fn}_D{D}"
            if key in all_results and "NCRO" in all_results[key]:
                res = all_results[key]["NCRO"]
                mean_c = np.maximum(res["convergence_mean"], 1e-30)
                iters = np.arange(1, len(mean_c) + 1)
                ax.semilogy(iters, mean_c, linewidth=1.5, label=f"D={D}", color=colors[D])
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel("Best Fitness (log scale)", fontsize=11)
        ax.set_title(f"NCRO Scalability — {fn}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fp = os.path.join(RESULTS_DIR, f"scalability_convergence_{letter}.png")
        plt.savefig(fp, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Scalability plot saved -> {fp}")


def plot_runtime_comparison(runtime_data, alg_names, dims):
    """Bar chart comparing average runtime across algorithms for D=30."""
    import json as _json
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Average runtime per algorithm at D=30
    alg_times = {}
    for entry in runtime_data:
        if entry["dim"] == 30:
            alg = entry["algorithm"]
            if alg not in alg_times:
                alg_times[alg] = []
            alg_times[alg].append(entry["avg_runtime"])

    if not alg_times:
        return

    names = [a for a in alg_names if a in alg_times]
    avg_times = [np.mean(alg_times[a]) for a in names]

    # Save data to JSON
    json_path = os.path.join(RESULTS_DIR, "runtime_comparison_data.json")
    with open(json_path, "w") as f:
        _json.dump({"names": names, "avg_times": avg_times}, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#E53935" if "NCRO" in n else "#1E88E5" for n in names]
    bars = ax.bar(names, avg_times, color=colors, width=0.6)
    ax.set_ylabel("Average Runtime per Run (seconds)", fontsize=11)
    ax.set_title("Computational Cost Comparison (D=30, T=500/1500)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}s", ha="center", fontweight="bold", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fp = os.path.join(RESULTS_DIR, "runtime_comparison.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Runtime plot saved -> {fp}")


# ──────────────────────────────────────────────────────────────
# Timed experiment runner (wraps run_experiment with timing)
# ──────────────────────────────────────────────────────────────

def run_timed_experiment(runtime_data, **kwargs):
    """Run experiment and measure wall-clock time."""
    alg_name = kwargs.get("algorithm_name", "NCRO")
    fn_name = kwargs.get("function_name", "")
    dim = kwargs.get("dimension", 30)
    max_iter = kwargs.get("max_iter", 500)
    num_runs = kwargs.get("num_runs", 30)

    t_start = time.perf_counter()
    results = run_experiment(**kwargs)
    t_end = time.perf_counter()

    total_time = t_end - t_start
    avg_runtime = total_time / num_runs
    avg_time_per_iter = avg_runtime / max_iter

    runtime_data.append({
        "algorithm": alg_name,
        "function": fn_name,
        "dim": dim,
        "avg_runtime": avg_runtime,
        "avg_time_per_iter": avg_time_per_iter,
        "iterations": max_iter,
        "num_runs": num_runs,
    })

    return results


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":

    POPULATION_SIZE = 30
    NCRO_ITER = 500
    NUM_RUNS = 30

    benchmark_functions = [
        # Unimodal
        "Sphere", "Rosenbrock", "Zakharov",
        # Multimodal
        "Rastrigin", "Ackley", "Griewank", "Schwefel", "Levy",
        # Hybrid
        "Hybrid1", "Hybrid2",
        # Composition
        "Composition1", "Composition2",
    ]
    # D ∈ {10, 30, 50, 100} per professor's plan
    dimensions = [10, 30, 50, 100]

    # Key functions for convergence plots (one per category)
    KEY_PLOT_FUNCTIONS = [
        "Sphere", "Rastrigin", "Ackley", "Rosenbrock",  # unimodal + multimodal
        "Hybrid1", "Composition1",                        # hybrid + composition
    ]
    KEY_PLOT_DIM = 30

    # Professor's required algorithms: PSO, DE, GWO, WOA, HHO, ABC, SCA
    algorithms = {
        "NCRO":         {"class": None,              "iter_mult": 1},
        "PSO":          {"class": PSOOptimizer,      "iter_mult": 3},
        "DE":           {"class": DEOptimizer,       "iter_mult": 3},
        "GWO":          {"class": GWOOptimizer,      "iter_mult": 3},
        "WOA":          {"class": WOAOptimizer,      "iter_mult": 3},
        "HHO":          {"class": HHOOptimizer,      "iter_mult": 3},
        "ABC":          {"class": ABCOptimizer,      "iter_mult": 3},
        "SCA":          {"class": SCAOptimizer,      "iter_mult": 3},
    }

    # Professor's ablation variants (Section 6.8)
    ablation_variants = {
        "NCRO-Full":  {"class": None,                    "iter": NCRO_ITER},
        "NCRO-R":     {"class": NCRO_NoRegret,           "iter": NCRO_ITER},
        "NCRO-C":     {"class": NCRO_NoCFMem,            "iter": NCRO_ITER},
        "NCRO-L":     {"class": NCRO_NoCounterfactual,   "iter": NCRO_ITER},
        "NCRO-M":     {"class": NCRO_NoMomentum,         "iter": NCRO_ITER},
    }

    algorithm_names = list(algorithms.keys())
    ablation_names = list(ablation_variants.keys())
    all_results = {}
    all_stats = {}
    runtime_data = []  # Collect timing information

    # ==============================================================
    # PART 1: BENCHMARK COMPARISON (with runtime measurement)
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
                results = run_timed_experiment(
                    runtime_data,
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
                results = run_timed_experiment(
                    runtime_data,
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

    # Table 5: Runtime analysis CSV
    save_runtime_csv(runtime_data)

    # Table 6: Scalability analysis CSV
    save_scalability_csv(all_results, benchmark_functions, dimensions)

    # Plot: Ranking summary bar chart
    plot_ranking_summary(all_ranks, algorithm_names)

    # Plot: Scalability convergence across dimensions
    plot_scalability(all_results, benchmark_functions, dimensions)

    # Plot: Runtime comparison
    plot_runtime_comparison(runtime_data, algorithm_names, dimensions)

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
                    all_ablation[key], f"ablation_{fn}_D{D}", ablation_names, "NCRO-Full")
                all_stats[abl_key] = abl_stat
    save_statistical_results(all_stats)

    # ==============================================================
    # PART 5: CONTROLLED-VARIANT EXPERIMENT
    # ==============================================================
    # Tests whether NCRO's coefficient-swapped counterfactual is
    # superior to alternative strategies (random, OBL, swap-no-regret)
    print("\n\n" + "#" * 80)
    print("  PART 5: CONTROLLED-VARIANT EXPERIMENT")
    print("#" * 80)

    from run_variant_experiment import run_experiment as run_variant_exp, save_results as save_variant_results

    variant_functions = ["Sphere", "Rastrigin", "Ackley", "Rosenbrock", "Griewank", "Schwefel"]
    variant_rows = run_variant_exp(
        function_names=variant_functions,
        dimension=30,
        population_size=POPULATION_SIZE,
        max_fes=NCRO_ITER * POPULATION_SIZE * 3,  # 45,000 FEs
        num_runs=NUM_RUNS,
        base_seed=2026,
    )
    save_variant_results(variant_rows)

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

    print("\n\n" + "=" * 120)
    print("ABLATION STUDY — Mean Fitness")
    print("=" * 120)
    for D in dimensions:
        print(f"\n--- D = {D} ---")
        header = f"{'Function':<14}"
        for var in ablation_names:
            short = var.replace("NCRO (Full)", "Full").replace("NCRO_", "")
            header += f"{short:>17}"
        print(header)
        print("-" * 119)
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

    print("\n\n" + "=" * 80)
    print("RUNTIME ANALYSIS — Average per run (seconds)")
    print("=" * 80)
    # Aggregate by algorithm
    alg_runtimes = {}
    for entry in runtime_data:
        alg = entry["algorithm"]
        if alg in algorithm_names:  # Only benchmark algorithms
            if alg not in alg_runtimes:
                alg_runtimes[alg] = []
            alg_runtimes[alg].append(entry["avg_runtime"])
    for alg in algorithm_names:
        if alg in alg_runtimes:
            avg_t = np.mean(alg_runtimes[alg])
            print(f"  {alg:<18}: {avg_t:.4f} sec/run")

    print("\n" + "=" * 80)
    print("\nResults saved to results/ folder:")
    print("  CSV Tables:")
    print("    benchmark_comparison.csv   — All algorithms × functions × dims")
    print("    ablation_study.csv         — All ablation variants (6 total)")
    print("    friedman_rankings.csv      — Average rankings per test case")
    print("    statistical_tests.csv      — Wilcoxon p-values + win/loss/tie")
    print("    runtime_analysis.csv       — Runtime per run & per iteration")
    print("    scalability_analysis.csv   — NCRO performance across D={10,30,50,100}")
    print("    variant_raw.csv            — Controlled-variant per-run results")
    print("    variant_summary.csv        — Controlled-variant summary statistics")
    print("    variant_wilcoxon.csv       — Controlled-variant Wilcoxon tests")
    print("    variant_friedman.csv       — Controlled-variant Friedman tests")
    print("  Plots (each figure saved individually for paper arrangement):")
    print("    comparison_*.png                — Convergence for key functions (D=30)")
    print("    ablation_*_a.png               — Ablation convergence curves")
    print("    ablation_*_b.png               — Ablation final performance bars")
    print("    ranking_summary.png            — Overall Friedman ranking bar chart")
    print("    scalability_convergence_a.png  — NCRO scalability: Sphere")
    print("    scalability_convergence_b.png  — NCRO scalability: Rastrigin")
    print("    scalability_convergence_c.png  — NCRO scalability: Ackley")
    print("    scalability_convergence_d.png  — NCRO scalability: Griewank")
    print("    runtime_comparison.png         — Runtime comparison bar chart")
    print("  JSON data (for regeneration by generate_paper_figures.py):")
    print("    *_data.json                    — Plot data for all figures")
