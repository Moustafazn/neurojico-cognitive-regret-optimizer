#!/usr/bin/env python3
"""
NCRO Controlled-Variant Experiment
===================================

Tests whether NCRO's coefficient-swapped counterfactual is specifically
superior to alternative strategies, complementing the ablation study.

The `full` variant is IDENTICAL to ncro/optimizer.py (NCROOptimizer),
including scouting, noise, blend, diversity recovery, and all mechanisms.

Implements four equal-FE-budget variants:
  1. full            : Full NCRO (identical to optimizer.py)
  2. swap_no_regret  : Same swapped CF candidate, but ALL learning disabled
  3. random          : CF candidate replaced by uniform random candidate
  4. obl             : CF candidate replaced by classical bound-based opposite

All variants use 45,000 FEs (NCRO: 500 iter × 3N; equivalent budget).

Output:
  results/variant_raw.csv       — per-run results
  results/variant_summary.csv   — aggregated statistics
  results/variant_wilcoxon.csv  — Wilcoxon signed-rank tests (full vs each)
  results/variant_friedman.csv  — Friedman ranking tests

Usage:
  venv/bin/python run_variant_experiment.py
  venv/bin/python run_variant_experiment.py --functions Sphere Rastrigin Ackley --runs 30
"""

import argparse
import csv
import os
import time
import numpy as np

try:
    from scipy.stats import friedmanchisquare, wilcoxon as scipy_wilcoxon
except ImportError:
    friedmanchisquare = None
    scipy_wilcoxon = None

from ncro.benchmarks import BENCHMARK_FUNCTIONS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
VARIANTS = ("full", "swap_no_regret", "random", "obl")


# ──────────────────────────────────────────────────────────────
# NCRO parameters (matching optimizer.py exactly)
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "c1": 2.0, "c2": 2.0,
    "alpha_max": 2.0, "alpha_min": 0.15,
    "beta_min": 0.15, "beta_max": 1.8,
    "gamma_max": 1.0, "gamma_min": 0.05,
    "rho": 0.90, "rho_c": 0.90,
    "eta_R": 0.30, "eta_C": 0.20, "eta_P": 0.20,
    "q_min": 0.10, "q_max": 0.90,
    "epsilon": 1e-12,
}


# ──────────────────────────────────────────────────────────────
# Core variant runner (matches optimizer.py for 'full')
# ──────────────────────────────────────────────────────────────

def run_ncro_variant(func, lower, upper, dimension, population_size,
                     max_iter, variant, seed, config=None):
    """
    Run one trial of an NCRO variant.
    The 'full' variant is identical to NCROOptimizer.optimize().
    Other variants differ ONLY in how Y_C is generated and whether learning is enabled.
    """
    if config is None:
        config = DEFAULT_CONFIG

    rng = np.random.default_rng(seed)
    N, D, T = population_size, dimension, max_iter
    L, U = lower, upper
    eps = config["epsilon"]
    search_range = U - L

    # Initialize population (identical to optimizer.py)
    X = rng.uniform(L, U, size=(N, D))
    F = np.array([func(X[i]) for i in range(N)])

    P = X.copy()           # Personal best positions
    PF = F.copy()          # Personal best fitness
    g_idx = int(np.argmin(F))
    G = P[g_idx].copy()
    GF = F[g_idx]

    # Memories
    M_R = np.zeros(N)           # Regret memory
    C_mem = np.zeros(N)         # Counterfactual success memory
    progress_mem = np.zeros(N)  # Progress memory

    # Momentum: track previous positions
    X_prev = X.copy()

    start_time = time.perf_counter()

    for t in range(T):
        tau = t / max(1, T - 1)

        # Time-varying parameters (identical to optimizer.py)
        alpha_t = config["alpha_max"] * (1 - tau) + config["alpha_min"] * tau
        beta_t = config["beta_min"] * (1 - tau) + config["beta_max"] * tau
        gamma_t = config["gamma_max"] * (1 - tau) + config["gamma_min"] * tau
        sigma_t = 1.0 * (1 - tau) + 0.01 * tau
        q0_t = config["q_max"] * (1 - tau) + config["q_min"] * tau

        # Population diversity (identical to optimizer.py)
        pop_diversity = np.mean(np.std(X, axis=0)) / (search_range + eps)

        newX = np.empty_like(X)
        newF = np.empty(N)

        for i in range(N):
            # ─── EXPLORATION DIRECTION ───
            # For full/random/obl: use scouting when stuck (identical to optimizer.py)
            # For swap_no_regret: always standard differential (no M_R to trigger scouting)

            use_scouting = (variant != "swap_no_regret")
            agent_stuck = use_scouting and (M_R[i] > 0.25 and progress_mem[i] < 0.01)
            diversity_collapsed = use_scouting and (pop_diversity < 0.005)

            if agent_stuck or diversity_collapsed:
                # Long-distance scouting (identical to optimizer.py)
                scout_type = rng.integers(4)
                center = (L + U) / 2.0
                if scout_type == 0:
                    target = np.clip(2.0 * center - G, L, U)
                elif scout_type == 1:
                    target = rng.uniform(L, U, D)
                elif scout_type == 2:
                    target = np.clip(2.0 * center - P[i], L, U)
                else:
                    r_agent = rng.integers(N)
                    target = P[r_agent].copy()
                E_i = target - X[i]
                regret_scale = 0.3 + 0.7 * M_R[i]
                E_i *= regret_scale
            else:
                # Standard differential exploration
                r1, r2 = rng.choice(N, 2, replace=False)
                E_i = X[r1] - X[r2]

                # Regret-proportional minimum step (skip for swap_no_regret)
                if variant != "swap_no_regret":
                    min_step = M_R[i] * 0.02 * search_range
                    E_norm = np.linalg.norm(E_i)
                    if E_norm < min_step and E_norm > eps:
                        E_i = E_i * (min_step / E_norm)

            # Exploitation direction (identical to optimizer.py)
            u1, u2 = rng.random(), rng.random()
            H_i = config["c1"] * u1 * (P[i] - X[i]) + config["c2"] * u2 * (G - X[i])

            # ─── ADAPTIVE q ───
            if variant == "swap_no_regret":
                q_i = float(np.clip(q0_t, config["q_min"], config["q_max"]))
            else:
                q_i = q0_t + config["eta_R"] * M_R[i] + config["eta_C"] * C_mem[i] - config["eta_P"] * progress_mem[i]
                q_i = float(np.clip(q_i, config["q_min"], config["q_max"]))

            # ─── ACTUAL CANDIDATE Y_A (identical for all variants) ───
            Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
            noise = 0.02 * sigma_t * search_range / np.sqrt(D) * rng.standard_normal(D)
            Y_A = np.clip(Y_A + noise, L, U)
            F_A = func(Y_A)

            # ─── COUNTERFACTUAL CANDIDATE Y_C (VARIES BY VARIANT) ───
            if variant in ("full", "swap_no_regret"):
                # Coefficient-swapped counterfactual (identical to optimizer.py)
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i
                Y_C = np.clip(Y_C, L, U)
            elif variant == "random":
                # Replace CF with uniform random candidate
                Y_C = rng.uniform(L, U, D)
            elif variant == "obl":
                # Classical opposition-based learning
                Y_C = np.clip(L + U - Y_A, L, U)

            F_C = func(Y_C)

            # ─── REGRET & CF SUCCESS (disabled for swap_no_regret) ───
            if variant == "swap_no_regret":
                regret = 0.0
                success = 0.0
                # M_R, C_mem stay at 0
            else:
                regret = max(0.0, F_A - F_C) / (abs(F_A) + abs(F_C) + eps)
                regret = float(np.clip(regret, 0, 1))
                success = float(F_C < F_A - eps)
                M_R[i] = config["rho"] * M_R[i] + (1 - config["rho"]) * regret
                C_mem[i] = config["rho_c"] * C_mem[i] + (1 - config["rho_c"]) * success

            # Counterfactual direction
            D_C = Y_C - X[i]

            # ─── MOMENTUM (identical to optimizer.py for full/random/obl) ───
            V_i = X[i] - X_prev[i]
            if variant == "swap_no_regret":
                w_momentum = 0.4 * (1 - tau)  # no M_R modulation
            else:
                w_momentum = (1 - M_R[i]) * 0.4 * (1 - tau)

            # ─── MOTION EQUATION (identical to optimizer.py) ───
            if variant == "swap_no_regret":
                # No regret modulation, no CF direction
                candidate = (
                    X[i]
                    + w_momentum * V_i
                    + alpha_t * E_i
                    + beta_t * H_i
                )
            else:
                candidate = (
                    X[i]
                    + w_momentum * V_i
                    + alpha_t * (1 + M_R[i]) * E_i
                    + beta_t * (1 - M_R[i]) * H_i
                    + gamma_t * C_mem[i] * D_C
                )

            # ─── BLEND (identical to optimizer.py) ───
            if rng.random() < q_i:
                candidate = 0.75 * candidate + 0.25 * Y_A
            else:
                candidate = 0.75 * candidate + 0.25 * Y_C

            candidate = np.clip(candidate, L, U)
            F_cand = func(candidate)

            # ─── SELECTION (identical to optimizer.py) ───
            if variant != "swap_no_regret" and pop_diversity < 0.01 and M_R[i] > 0.2:
                # Diversity-aware: force movement
                move_candidates = [Y_A, Y_C, candidate]
                move_values = [F_A, F_C, F_cand]
                best_move = int(np.argmin(move_values))
                newX[i] = move_candidates[best_move]
                newF[i] = move_values[best_move]
            else:
                # Standard greedy 4-way selection
                candidates = [X[i], Y_A, Y_C, candidate]
                values = [F[i], F_A, F_C, F_cand]
                best_idx = int(np.argmin(values))
                newX[i] = candidates[best_idx]
                newF[i] = values[best_idx]

        # Update population
        F_old = F.copy()
        X_prev = X.copy()
        X = newX
        F = newF

        # Update personal bests
        improved = F < PF
        P[improved] = X[improved]
        PF[improved] = F[improved]

        # Update global best
        k = int(np.argmin(PF))
        if PF[k] < GF:
            G = P[k].copy()
            GF = PF[k]

        # ─── DIVERSITY RECOVERY (identical to optimizer.py, skip for swap_no_regret) ───
        if variant != "swap_no_regret" and pop_diversity < 0.002 and tau < 0.8:
            n_reset = max(1, N // 5)
            worst_idx = np.argsort(F)[-n_reset:]
            for wi in worst_idx:
                X[wi] = rng.uniform(L, U, D)
                F[wi] = func(X[wi])

        # Update progress memory
        if variant != "swap_no_regret":
            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(
                accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0
            )

    elapsed = time.perf_counter() - start_time
    return {
        "best_fitness": float(GF),
        "iterations": T,
        "runtime_seconds": elapsed,
    }


# ──────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────

def run_experiment(function_names, dimension=30, population_size=30,
                   max_fes=45000, num_runs=30, base_seed=2026):
    """Run the full controlled-variant experiment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Compute iterations from FE budget: NCRO uses 3N FEs/iter
    max_iter = max_fes // (3 * population_size)

    raw_rows = []

    for fn_name in function_names:
        bench = BENCHMARK_FUNCTIONS[fn_name]
        func_obj = bench["function"]
        lower, upper = bench["bounds"]
        optimal = bench["optimal"]

        for run in range(num_runs):
            seed = base_seed + run
            for variant in VARIANTS:
                result = run_ncro_variant(
                    func_obj, lower, upper, dimension, population_size,
                    max_iter, variant, seed
                )
                row = {
                    "function": fn_name,
                    "variant": variant,
                    "seed": seed,
                    "best_fitness": result["best_fitness"],
                    "error": abs(result["best_fitness"] - optimal),
                    "iterations": result["iterations"],
                    "runtime_seconds": result["runtime_seconds"],
                }
                raw_rows.append(row)
                print(
                    f"  {fn_name:12s} run={run+1:02d}/{num_runs} "
                    f"variant={variant:15s} best={result['best_fitness']:.6e}"
                )

    return raw_rows


def save_results(raw_rows):
    """Save raw results, summary, and statistical tests to CSV."""
    import collections

    # 1. Raw results
    raw_path = os.path.join(RESULTS_DIR, "variant_raw.csv")
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

    summary_path = os.path.join(RESULTS_DIR, "variant_summary.csv")
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
                f"{np.std(arr, ddof=1):.6e}" if len(arr) > 1 else "0",
                f"{np.max(arr):.6e}",
            ])
    print(f"  Saved: {summary_path}")

    # 3. Wilcoxon signed-rank tests (full vs each alternative)
    wilcoxon_rows = []
    functions = sorted(set(r["function"] for r in raw_rows))
    for fn in functions:
        seed_values = collections.defaultdict(dict)
        for row in raw_rows:
            if row["function"] == fn:
                seed_values[row["seed"]][row["variant"]] = row["best_fitness"]

        seeds = sorted(seed_values.keys())
        full_vals = np.array([seed_values[s]["full"] for s in seeds if "full" in seed_values[s]])

        for var in VARIANTS[1:]:
            var_vals = np.array([seed_values[s][var] for s in seeds if var in seed_values[s]])
            if len(full_vals) != len(var_vals) or len(full_vals) < 2:
                continue

            if np.allclose(full_vals, var_vals):
                stat, pval = 0.0, 1.0
            elif scipy_wilcoxon is not None:
                stat, pval = scipy_wilcoxon(full_vals, var_vals,
                                            alternative="two-sided",
                                            zero_method="wilcox")
            else:
                stat, pval = float("nan"), float("nan")

            wilcoxon_rows.append({
                "function": fn,
                "comparison": f"full vs {var}",
                "statistic": f"{stat:.4f}",
                "p_value": f"{pval:.6e}",
                "full_wins": int(np.sum(full_vals < var_vals)),
                "full_losses": int(np.sum(full_vals > var_vals)),
                "ties": int(np.sum(full_vals == var_vals)),
            })

    wilcoxon_path = os.path.join(RESULTS_DIR, "variant_wilcoxon.csv")
    with open(wilcoxon_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "function", "comparison", "statistic", "p_value",
            "full_wins", "full_losses", "ties"
        ])
        writer.writeheader()
        writer.writerows(wilcoxon_rows)
    print(f"  Saved: {wilcoxon_path}")

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

        if len(arrays) == len(VARIANTS) and friedmanchisquare is not None and len(seeds) >= 2:
            stat, pval = friedmanchisquare(*arrays)
        else:
            stat, pval = float("nan"), float("nan")

        friedman_rows.append({
            "function": fn,
            "statistic": f"{stat:.4f}" if not np.isnan(stat) else "NaN",
            "p_value": f"{pval:.6e}" if not np.isnan(pval) else "NaN",
        })

    friedman_path = os.path.join(RESULTS_DIR, "variant_friedman.csv")
    with open(friedman_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["function", "statistic", "p_value"])
        writer.writeheader()
        writer.writerows(friedman_rows)
    print(f"  Saved: {friedman_path}")

    # 5. Print summary table
    print("\n" + "=" * 90)
    print("  NCRO Controlled-Variant Experiment Summary")
    print("=" * 90)
    print(f"  {'Function':<14s} {'Variant':<18s} {'Best':>12s} {'Mean':>12s} {'Std':>12s} {'Worst':>12s}")
    print("  " + "-" * 82)
    for (fn, var), values in sorted(groups.items()):
        arr = np.array(values)
        print(f"  {fn:<14s} {var:<18s} {np.min(arr):>12.4e} {np.mean(arr):>12.4e} "
              f"{np.std(arr, ddof=1):>12.4e} {np.max(arr):>12.4e}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="NCRO Controlled-Variant Experiment"
    )
    parser.add_argument(
        "--functions", nargs="+",
        default=["Sphere", "Rastrigin", "Ackley", "Rosenbrock", "Griewank", "Schwefel"],
        help="Benchmark functions to test"
    )
    parser.add_argument("--dimension", type=int, default=30)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--max-fes", type=int, default=45000)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    print("=" * 60)
    print("  NCRO Controlled-Variant Experiment")
    print(f"  Functions: {args.functions}")
    print(f"  D={args.dimension}  N={args.population}  MaxFEs={args.max_fes}  Runs={args.runs}")
    print(f"  Variants: {VARIANTS}")
    print(f"  Iterations: {args.max_fes // (3 * args.population)}")
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
