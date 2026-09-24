#!/usr/bin/env python3
"""
NCRO Controlled-Variant Experiment
===================================

Tests whether NCRO's coefficient-swapped counterfactual direction is
specifically superior to alternative strategies, complementing the
ablation study.

Implements four equal-FE-budget variants using the actual optimizer class:
  1. full            : Full NCRO (NCROOptimizer as-is)
  2. swap_no_regret  : Same motion equation but regret/CF memories frozen
  3. random          : D_C replaced by a random perturbation direction
  4. obl             : D_C replaced by opposition-based direction

All variants share the same FE budget.

Output:
  results/tables/variant_raw.csv
  results/tables/variant_summary.csv
  results/tables/variant_wilcoxon.csv
  results/tables/variant_friedman.csv

Usage:
  python -m ncro.variant_experiment
  python -m ncro.variant_experiment --functions Sphere Rastrigin --runs 30
"""

import argparse
import collections
import csv
import os
import time

import numpy as np

try:
    from scipy.stats import friedmanchisquare, wilcoxon as scipy_wilcoxon
except ImportError:
    friedmanchisquare = None
    scipy_wilcoxon = None

from .benchmarks import BENCHMARK_FUNCTIONS
from .optimizer import NCROOptimizer, NCROResult

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
VARIANTS = ("full", "swap_no_regret", "random", "obl")


# ──────────────────────────────────────────────────────────────
# Variant optimizers (subclass, override only what differs)
# ──────────────────────────────────────────────────────────────

class NCRO_SwapNoRegret(NCROOptimizer):
    """Same motion equation but regret/CF memories stay at zero."""

    def _update_memories_rank(self, i, F_old, F_new, M_R, C_mem):
        # No learning — memories frozen at 0
        return 0.0


class NCRO_RandomCF(NCROOptimizer):
    """D_C is a random perturbation instead of the swapped-coefficient direction."""

    def optimize(self) -> NCROResult:
        # Patch _build_candidate to use random D_C
        orig_build = self._build_candidate
        rng = np.random.default_rng(self.seed)

        def _random_dc_build(X_i, X_prev_i, E_i, H_i, D_C, M_R_i,
                             C_mem_i, alpha_t, beta_t, gamma_t, tau):
            sr = self._scalar_range(self.U - self.L)
            D_C_rand = rng.standard_normal(self.D) * sr * 0.1
            return orig_build(X_i, X_prev_i, E_i, H_i, D_C_rand,
                              M_R_i, C_mem_i, alpha_t, beta_t,
                              gamma_t, tau)

        self._build_candidate = _random_dc_build
        result = super().optimize()
        self._build_candidate = orig_build
        return result


class NCRO_OBL_CF(NCROOptimizer):
    """D_C is the opposition-based direction (L+U-X) instead of swapped."""

    def optimize(self) -> NCROResult:
        orig_build = self._build_candidate
        L, U = self.L, self.U

        def _obl_dc_build(X_i, X_prev_i, E_i, H_i, D_C, M_R_i,
                          C_mem_i, alpha_t, beta_t, gamma_t, tau):
            D_C_obl = (L + U - 2.0 * X_i)
            return orig_build(X_i, X_prev_i, E_i, H_i, D_C_obl,
                              M_R_i, C_mem_i, alpha_t, beta_t,
                              gamma_t, tau)

        self._build_candidate = _obl_dc_build
        result = super().optimize()
        self._build_candidate = orig_build
        return result


VARIANT_CLASSES = {
    "full": NCROOptimizer,
    "swap_no_regret": NCRO_SwapNoRegret,
    "random": NCRO_RandomCF,
    "obl": NCRO_OBL_CF,
}


# ──────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────

def run_experiment(function_names, dimension=30, population_size=30,
                   max_fes=45000, num_runs=30, base_seed=2026):
    """Run the full controlled-variant experiment."""
    os.makedirs(TABLES_DIR, exist_ok=True)

    # 1 FE per agent per iteration in the new architecture
    init_fes = 2 * population_size + population_size // 5
    max_iter = max(1, (max_fes - init_fes) // (1 * population_size))

    raw_rows = []

    for fn_name in function_names:
        bench = BENCHMARK_FUNCTIONS[fn_name]
        func_obj = bench["function"]
        bounds = bench["bounds"]
        optimal = bench["optimal"]

        for run in range(num_runs):
            seed = base_seed + run
            for variant in VARIANTS:
                cls = VARIANT_CLASSES[variant]
                opt = cls(
                    objective_function=func_obj,
                    dimension=dimension,
                    bounds=bounds,
                    population_size=population_size,
                    max_iter=max_iter,
                    seed=seed,
                )
                t0 = time.perf_counter()
                result = opt.optimize()
                elapsed = time.perf_counter() - t0

                row = {
                    "function": fn_name,
                    "variant": variant,
                    "seed": seed,
                    "best_fitness": result.best_fitness,
                    "error": abs(result.best_fitness - optimal),
                    "iterations": max_iter,
                    "runtime_seconds": elapsed,
                }
                raw_rows.append(row)
                print(
                    f"  {fn_name:12s} run={run+1:02d}/{num_runs} "
                    f"variant={variant:15s} best={result.best_fitness:.6e}"
                )

    return raw_rows


# ──────────────────────────────────────────────────────────────
# Save results + statistical tests
# ──────────────────────────────────────────────────────────────

def save_wilcoxon_results(raw_rows):
    """Recompute paired statistics only; ties use the existing NumPy tolerance.

    Ties are excluded from wins/losses and zeroed for the signed-rank test.
    Raw objective values and other summary files are never modified here.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    functions = sorted(set(r["function"] for r in raw_rows))
    wilcoxon_rows = []
    for fn in functions:
        seed_values = collections.defaultdict(dict)
        for row in raw_rows:
            if row["function"] == fn:
                seed_values[row["seed"]][row["variant"]] = row["best_fitness"]

        seeds = sorted(seed_values.keys())
        for var in VARIANTS[1:]:
            paired_seeds = [s for s in seeds
                            if "full" in seed_values[s] and var in seed_values[s]]
            full_vals = np.array([seed_values[s]["full"] for s in paired_seeds])
            var_vals = np.array([seed_values[s][var] for s in paired_seeds])
            if len(paired_seeds) < 2:
                continue

            tie_mask = np.isclose(full_vals, var_vals, rtol=1e-5, atol=1e-8)
            if np.all(tie_mask):
                stat, pval = 0.0, 1.0
            elif scipy_wilcoxon is not None:
                differences = np.where(tie_mask, 0.0, full_vals - var_vals)
                stat, pval = scipy_wilcoxon(differences)
            else:
                stat, pval = float("nan"), float("nan")

            wins = int(np.sum((full_vals < var_vals) & ~tie_mask))
            losses = int(np.sum((full_vals > var_vals) & ~tie_mask))
            ties = int(np.sum(tie_mask))

            assert wins + losses + ties == len(full_vals)
            wilcoxon_rows.append({
                "function": fn,
                "comparison": f"full_vs_{var}",
                "statistic": f"{stat:.4f}",
                "p_value": f"{pval:.6e}",
                "full_wins": wins,
                "full_losses": losses,
                "ties": ties,
            })

    wilcoxon_path = os.path.join(TABLES_DIR, "variant_wilcoxon.csv")
    with open(wilcoxon_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "function", "comparison", "statistic", "p_value",
            "full_wins", "full_losses", "ties"
        ])
        writer.writeheader()
        writer.writerows(wilcoxon_rows)
    print(f"  Saved: {wilcoxon_path}")
    return wilcoxon_rows



def stable_sample_std(values):
    """Sample standard deviation without squaring tiny raw magnitudes."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        return 0.0
    return float(scale * np.std(arr / scale, ddof=1))


def save_results(raw_rows):
    """Save raw results, summary, and statistical tests to CSV."""
    os.makedirs(TABLES_DIR, exist_ok=True)

    # 1. Raw results
    raw_path = os.path.join(TABLES_DIR, "variant_raw.csv")
    fieldnames = ["function", "variant", "seed", "best_fitness", "error",
                  "iterations", "runtime_seconds"]
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\n  Saved: {raw_path}")

    # 2. Summary statistics
    groups = collections.defaultdict(list)
    for row in raw_rows:
        key = (row["function"], row["variant"])
        groups[key].append(row["best_fitness"])

    summary_path = os.path.join(TABLES_DIR, "variant_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Function", "Variant", "Runs", "Best", "Mean",
                         "Median", "Std", "Worst"])
        for (fn, var), values in sorted(groups.items()):
            arr = np.array(values)
            writer.writerow([
                fn, var, len(arr),
                f"{np.min(arr):.6e}", f"{np.mean(arr):.6e}",
                f"{np.median(arr):.6e}",
                f"{stable_sample_std(arr):.6e}" if len(arr) > 1 else "0",
                f"{np.max(arr):.6e}",
            ])
    print(f"  Saved: {summary_path}")

    # 3. Wilcoxon signed-rank tests (full vs each alternative)
    functions = sorted(set(r["function"] for r in raw_rows))
    save_wilcoxon_results(raw_rows)

    # 4. Friedman test across all 4 variants
    friedman_rows = []
    for fn in functions:
        seed_values = collections.defaultdict(dict)
        for row in raw_rows:
            if row["function"] == fn:
                seed_values[row["seed"]][row["variant"]] = row["best_fitness"]

        seeds = sorted(seed_values.keys())
        arrays = []
        for var in VARIANTS:
            arr = [seed_values[s].get(var) for s in seeds]
            if None in arr:
                break
            arrays.append(np.array(arr))

        if (len(arrays) == len(VARIANTS)
                and friedmanchisquare is not None
                and len(seeds) >= 2):
            stat, pval = friedmanchisquare(*arrays)
        else:
            stat, pval = float("nan"), float("nan")

        friedman_rows.append({
            "function": fn,
            "statistic": f"{stat:.4f}" if not np.isnan(stat) else "NaN",
            "p_value": f"{pval:.6e}" if not np.isnan(pval) else "NaN",
        })

    friedman_path = os.path.join(TABLES_DIR, "variant_friedman.csv")
    with open(friedman_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["function", "statistic", "p_value"])
        writer.writeheader()
        writer.writerows(friedman_rows)
    print(f"  Saved: {friedman_path}")

    # 5. Print summary table
    print("\n" + "=" * 90)
    print("  NCRO Controlled-Variant Experiment Summary")
    print("=" * 90)
    print(f"  {'Function':<14s} {'Variant':<18s} "
          f"{'Best':>12s} {'Mean':>12s} {'Std':>12s} {'Worst':>12s}")
    print("  " + "-" * 82)
    for (fn, var), values in sorted(groups.items()):
        arr = np.array(values)
        print(f"  {fn:<14s} {var:<18s} {np.min(arr):>12.4e} "
              f"{np.mean(arr):>12.4e} {stable_sample_std(arr):>12.4e} "
              f"{np.max(arr):>12.4e}")
    print("=" * 90)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NCRO Controlled-Variant Experiment"
    )
    parser.add_argument(
        "--functions", nargs="+",
        default=["Sphere", "Rastrigin", "Ackley",
                 "Rosenbrock", "Griewank", "Schwefel"],
        help="Benchmark functions to test"
    )
    parser.add_argument("--dimension", type=int, default=30)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--max-fes", type=int, default=45000)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stats-only", action="store_true",
                        help="Recompute only variant_wilcoxon.csv from saved variant_raw.csv")
    args = parser.parse_args()
    if args.stats_only:
        with open(os.path.join(TABLES_DIR, "variant_raw.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["seed"] = int(row["seed"])
            row["best_fitness"] = float(row["best_fitness"])
        save_wilcoxon_results(rows)
        return

    print("=" * 60)
    print("  NCRO Controlled-Variant Experiment")
    print(f"  Functions: {args.functions}")
    print(f"  D={args.dimension}  N={args.population}  "
          f"MaxFEs={args.max_fes}  Runs={args.runs}")
    print(f"  Variants: {VARIANTS}")
    print(f"  Iterations: {args.max_fes // args.population}")
    print("=" * 60)

    raw_rows = run_experiment(
        function_names=args.functions,
        dimension=args.dimension,
        population_size=args.population,
        max_fes=args.max_fes,
        num_runs=args.runs,
        base_seed=args.seed,
    )

    save_results(raw_rows)


if __name__ == "__main__":
    main()
