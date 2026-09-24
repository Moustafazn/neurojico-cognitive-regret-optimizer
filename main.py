#!/usr/bin/env python3
"""
NCRO Complete Experiment — Single Entry Point
===============================================

Runs ALL experiments needed for the manuscript:

  Section 7  →  ncro/microgrid_experiment.py   (microgrid comparison)
  Section 8  →  ablation study + variant experiment
  Section 9  →  scalability + runtime analysis

Usage
-----
  python main.py            # full run
  python main.py --quick    # fast validation (fewer runs)
"""

import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ncro import run_experiment
from ncro.experiment import plot_ablation, RESULTS_DIR, TABLES_DIR, FIGURES_DIR
from ncro.ablation import (
    NCRO_NoRegret, NCRO_NoCFMem, NCRO_NoCounterfactual,
    NCRO_NoAdaptiveEE, NCRO_NoRegretMemory, NCRO_NoMomentum,
)


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

BENCHMARK_FUNCTIONS_LIST = [
    "Sphere", "Rosenbrock", "Zakharov",
    "Rastrigin", "Ackley", "Griewank", "Schwefel", "Levy",
]

ABLATION_VARIANTS = {
    "NCRO (Full)":      {"class": None},
    "NCRO_NoRegret":    {"class": NCRO_NoRegret},
    "NCRO_NoCFMem":     {"class": NCRO_NoCFMem},
    "NCRO_NoCounterfactual": {"class": NCRO_NoCounterfactual},
    "NCRO_NoMomentum":  {"class": NCRO_NoMomentum},
    "NCRO_NoAdaptiveEE": {"class": NCRO_NoAdaptiveEE},
    "NCRO_NoRegretMemory": {"class": NCRO_NoRegretMemory},
}

DIMENSIONS = [10, 30, 50, 100]
KEY_SCALABILITY_FUNCS = ["Sphere", "Rastrigin", "Ackley", "Griewank"]


# ──────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────

def save_ablation_csv(all_ablation, funcs, dims, var_names):
    os.makedirs(TABLES_DIR, exist_ok=True)
    fp = os.path.join(TABLES_DIR, "ablation_study.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Function", "Dim", "Variant", "Best", "Worst",
                     "Mean", "Std", "Median", "ER", "XR", "CSR", "AvgRegret"])
        for D in dims:
            for fn in funcs:
                key = f"{fn}_D{D}"
                if key not in all_ablation:
                    continue
                for var in var_names:
                    if var not in all_ablation[key]:
                        continue
                    r = all_ablation[key][var]
                    w.writerow([fn, D, var,
                                f"{r['best']:.6e}", f"{r['worst']:.6e}",
                                f"{r['mean']:.6e}", f"{r['std']:.6e}",
                                f"{r['median']:.6e}",
                                f"{r['exploration_ratio']:.4f}",
                                f"{r['exploitation_ratio']:.4f}",
                                f"{r['cf_success_rate']:.4f}",
                                f"{r['avg_regret']:.6e}"])
    print(f"  Saved -> {fp}")


def save_scalability_csv(all_scalability, funcs, dims):
    os.makedirs(TABLES_DIR, exist_ok=True)
    fp = os.path.join(TABLES_DIR, "scalability_analysis.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Function", "Dim", "Best", "Mean", "Std", "Median", "Worst"])
        for fn in funcs:
            for D in dims:
                key = f"{fn}_D{D}"
                if key not in all_scalability:
                    continue
                r = all_scalability[key]
                w.writerow([fn, D,
                            f"{r['best']:.6e}", f"{r['mean']:.6e}",
                            f"{r['std']:.6e}", f"{r['median']:.6e}",
                            f"{r['worst']:.6e}"])
    print(f"  Saved -> {fp}")


def save_runtime_csv(runtime_data):
    os.makedirs(TABLES_DIR, exist_ok=True)
    fp = os.path.join(TABLES_DIR, "runtime_analysis.csv")
    with open(fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Algorithm", "Function", "Dim",
                     "Avg_Runtime_sec", "Avg_Time_Per_Iter_ms",
                     "Iterations", "Num_Runs"])
        for entry in runtime_data:
            w.writerow([entry["algorithm"], entry["function"], entry["dim"],
                        f"{entry['avg_runtime']:.4f}",
                        f"{entry['avg_time_per_iter']:.4f}",
                        entry["iterations"], entry["num_runs"]])
    print(f"  Saved -> {fp}")


# ──────────────────────────────────────────────────────────────
# Section 7: Microgrid experiment
# ──────────────────────────────────────────────────────────────

def run_section_7(quick):
    print("\n" + "=" * 60)
    print("  SECTION 7: Smart Microgrid Experiment")
    print("=" * 60)
    from ncro.microgrid_experiment import run_experiment as run_microgrid
    run_microgrid(quick_mode=quick)


# ──────────────────────────────────────────────────────────────
# Section 8: Ablation study
# ──────────────────────────────────────────────────────────────

def run_section_8(num_runs, max_iter, pop_size, dims):
    print("\n" + "=" * 60)
    print("  SECTION 8: Ablation Study")
    print("=" * 60)
    all_ablation = {}
    ablation_names = list(ABLATION_VARIANTS.keys())
    runtime_data = []
    for D in dims:
        for fn in BENCHMARK_FUNCTIONS_LIST:
            print(f"\n  {fn} D={D}")
            for var_name, var_cfg in ABLATION_VARIANTS.items():
                t0 = time.perf_counter()
                res = run_experiment(
                    function_name=fn, dimension=D,
                    population_size=pop_size, max_iter=max_iter,
                    num_runs=num_runs, algorithm_name=var_name,
                    optimizer_class=var_cfg["class"],
                )
                elapsed = time.perf_counter() - t0
                key = f"{fn}_D{D}"
                if key not in all_ablation:
                    all_ablation[key] = {}
                all_ablation[key][var_name] = res
                runtime_data.append({
                    "algorithm": var_name, "function": fn,
                    "dim": D,
                    "avg_runtime": elapsed / num_runs,
                    "avg_time_per_iter":
                        (elapsed / num_runs / max_iter) * 1000,
                    "iterations": max_iter, "num_runs": num_runs,
                })
            if D == 30 and fn in KEY_SCALABILITY_FUNCS:
                plot_ablation(all_ablation[f"{fn}_D{D}"],
                              fn, ablation_names)
    save_ablation_csv(all_ablation, BENCHMARK_FUNCTIONS_LIST,
                      dims, ablation_names)
    save_runtime_csv(runtime_data)
    # Section 8.6: controlled-variant experiment
    print("\n  Running controlled-variant experiment...")
    from ncro.variant_experiment import main as variant_main
    variant_main()



# ──────────────────────────────────────────────────────────────
# Section 9: Scalability & runtime
# ──────────────────────────────────────────────────────────────

def run_section_9(num_runs, max_iter, pop_size, dims):
    print("\n" + "=" * 60)
    print("  SECTION 9: Scalability & Runtime")
    print("=" * 60)
    all_scalability = {}
    for fn in KEY_SCALABILITY_FUNCS:
        for D in dims:
            print(f"  NCRO scalability: {fn} D={D}")
            res = run_experiment(
                function_name=fn, dimension=D,
                population_size=pop_size, max_iter=max_iter,
                num_runs=num_runs, algorithm_name="NCRO",
            )
            all_scalability[f"{fn}_D{D}"] = res
    save_scalability_csv(all_scalability, KEY_SCALABILITY_FUNCS, dims)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="NCRO complete experiment suite")
    parser.add_argument("--quick", action="store_true",
                        help="Fast validation (3 runs, D=30 only)")
    args = parser.parse_args()

    if args.quick:
        NUM_RUNS = 3
        MAX_ITER = 500
        POP_SIZE = 30
        DIMS = [30]
        print("QUICK MODE: 3 runs, D=30 only")
    else:
        NUM_RUNS = 30
        MAX_ITER = 1000
        POP_SIZE = 30
        DIMS = [10, 30, 50, 100]

    run_section_7(quick=args.quick)
    run_section_8(NUM_RUNS, MAX_ITER, POP_SIZE, DIMS)
    run_section_9(NUM_RUNS, MAX_ITER, POP_SIZE, DIMS)

    print("\n" + "=" * 60)
    print("  ALL SECTIONS COMPLETE")
    print("  Results saved to results/ folder")
    print("=" * 60)

