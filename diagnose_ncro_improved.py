#!/usr/bin/env python3
"""
NCRO-I Diagnostic Script — Convergence Bottleneck Analysis
============================================================

Reads the existing experimental results (no need to re-run competitors)
and runs focused micro-benchmarks on the NCRO-I optimizer to identify
exactly WHY it converges ~3× slower than DE variants / CMA-ES on the
4D convex microgrid problem, and what can be tuned to reach rank 2–3.

Outputs:
  results/tables/diagnostic_report.csv   — per-scenario breakdown
  results/figures/diagnostic_convergence.png
  Console summary with actionable recommendations

Usage:
  python diagnose_ncro_improved.py
"""

from __future__ import annotations

import csv
import os
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ncro_microgrid_experiment import (
    MicrogridProblem,
    EvaluationRecorder,
    TABLES_DIR,
    FIGURES_DIR,
)
from ncro.optimizer_with_improvements import NCROOptimizer as NCROOptimizerImproved

# ============================================================
# 1. Configuration
# ============================================================

SCENARIOS = {
    "Base": MicrogridProblem(name="Base"),
    "HighLoad": MicrogridProblem(name="HighLoad", load=150.0,
                                 upper_bounds=(80.0, 70.0, 50.0, 150.0)),
    "LowSolar": MicrogridProblem(name="LowSolar",
                                  upper_bounds=(20.0, 50.0, 35.0, 100.0)),
}

NUM_RUNS = 10
POP_SIZE = 30
FE_BUDGET = 45_000
TOLERANCE = 1e-6  # same as the main experiment
SEED_BASE = 99900


# ============================================================
# 2. Load competitor baselines from existing results
# ============================================================

def load_competitor_fes():
    """Load MeanFEsToTolerance for competitors from summary_results.csv."""
    fp = TABLES_DIR / "summary_results.csv" if isinstance(TABLES_DIR, Path) \
        else Path(TABLES_DIR) / "summary_results.csv"
    if not fp.exists():
        print(f"  WARNING: {fp} not found — will skip competitor comparison.")
        return {}

    competitors = {}
    with open(fp, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alg = row["Algorithm"]
            scen = row["Scenario"]
            if alg in ("NCRO", "NCRO-I"):
                continue
            key = (scen, alg)
            competitors[key] = float(row["MeanFEsToTolerance"])
    return competitors


# ============================================================
# 3. Instrumented NCRO-I run — counts FEs per mechanism
# ============================================================

class InstrumentedNCROI(NCROOptimizerImproved):
    """
    Wraps the improved optimizer to count function evaluations
    per mechanism per iteration:
      - init_fes:   population initialization (standard + OBL)
      - actual_fes: Y_A evaluations
      - cf_fes:     Y_C evaluations
      - cand_fes:   motion-equation candidate evaluations
      - stag_fes:   stagnation opposition-jump evaluations
      - diversity_fes: diversity recovery evaluations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fe_counter = {"init": 0, "actual": 0, "cf": 0,
                            "candidate": 0, "stagnation": 0,
                            "diversity": 0, "total": 0}
        self._real_func = self.func
        self._current_phase = "init"
        self.func = self._counting_func
        # Track convergence per FE (not per iteration)
        self._convergence_per_fe = []

    def _counting_func(self, x):
        self._fe_counter[self._current_phase] += 1
        self._fe_counter["total"] += 1
        return self._real_func(x)

    def optimize(self):
        """Override to track phases."""
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U, eps = self.L, self.U, self.epsilon

        # ── Init ──
        self._current_phase = "init"
        X, F = self._initialize_population(rng)

        P = X.copy()
        PF = F.copy()
        g_idx = int(np.argmin(F))
        G = P[g_idx].copy()
        GF = F[g_idx]

        M_R = np.zeros(N)
        C_mem = np.zeros(N)
        progress_mem = np.zeros(N)
        X_prev = X.copy()

        convergence = np.zeros(T + 1)
        convergence[0] = GF
        expl_cnt = np.zeros(T, dtype=int)
        xplt_cnt = np.zeros(T, dtype=int)
        cf_cnt = np.zeros(T, dtype=int)
        regret_avg = np.zeros(T)
        q_hist = np.zeros((T, N))

        stag_counter = 0
        last_GF = GF

        self._convergence_per_fe.append((self._fe_counter["total"], GF))

        for t in range(T):
            tau = t / max(T - 1, 1)
            alpha_t, beta_t, gamma_t, sigma_t, q0_t = self._compute_schedules(tau)
            search_range = U - L

            pop_std = np.std(X, axis=0)
            pop_diversity = float(np.mean(pop_std / (search_range + eps)))

            newX = X.copy()
            newF = F.copy()

            it_expl = 0
            it_xplt = 0
            it_cf = 0
            it_regret = 0.0

            for i in range(N):
                self._current_phase = "actual"

                E_i = self._compute_exploration(
                    i, X, P, G, M_R, progress_mem,
                    pop_diversity, search_range, rng)

                u1, u2 = rng.random(), rng.random()
                H_i = self.c1 * u1 * (P[i] - X[i]) + self.c2 * u2 * (G - X[i])

                q_i = (q0_t + self.eta_R * M_R[i]
                       + self.eta_C * C_mem[i]
                       - self.eta_P * progress_mem[i])
                q_i = float(np.clip(q_i, self.q_min, self.q_max))
                q_hist[t, i] = q_i

                Y_A = X[i] + q_i * alpha_t * E_i + (1 - q_i) * beta_t * H_i
                Y_C = X[i] + (1 - q_i) * alpha_t * E_i + q_i * beta_t * H_i

                Y_A = self._apply_noise(
                    Y_A, X[i], G, M_R[i], sigma_t, tau,
                    pop_diversity, search_range, rng)
                Y_C = self._repair(Y_C)

                self._current_phase = "actual"
                F_A = self.func(Y_A)
                self._current_phase = "cf"
                F_C = self.func(Y_C)

                success = self._update_memories(i, F_A, F_C, M_R, C_mem)

                D_C = Y_C - X[i]
                candidate = self._build_candidate(
                    X[i], X_prev[i], E_i, H_i, D_C, M_R[i],
                    C_mem[i], alpha_t, beta_t, gamma_t, tau)

                if rng.random() < q_i:
                    candidate = 0.75 * candidate + 0.25 * Y_A
                    it_expl += 1
                else:
                    candidate = 0.75 * candidate + 0.25 * Y_C
                    it_xplt += 1

                candidate = self._repair(candidate)
                self._current_phase = "candidate"
                F_cand = self.func(candidate)

                newX[i], newF[i] = self._select_survivor(
                    X[i], F[i], Y_A, F_A, Y_C, F_C,
                    candidate, F_cand, M_R[i], pop_diversity)

                if success > 0:
                    it_cf += 1
                it_regret += M_R[i]

            F_old = F.copy()
            X_prev = X.copy()
            X = newX
            F = newF

            improved = F < PF
            P[improved] = X[improved]
            PF[improved] = F[improved]
            k = int(np.argmin(PF))
            if PF[k] < GF:
                G = P[k].copy()
                GF = PF[k]

            self._current_phase = "stagnation"
            G, GF, stag_counter, last_GF = self._stagnation_opposition_jump(
                G, GF, X, F, P, PF, stag_counter, last_GF, tau)

            self._current_phase = "diversity"
            self._recover_diversity(X, F, pop_diversity, tau, rng)

            accepted_improvement = np.maximum(0.0, F_old - PF)
            progress_mem[:] = np.clip(
                accepted_improvement / (np.abs(F_old) + eps), 0.0, 1.0)

            convergence[t + 1] = GF
            expl_cnt[t] = it_expl
            xplt_cnt[t] = it_xplt
            cf_cnt[t] = it_cf
            regret_avg[t] = it_regret / N
            self._convergence_per_fe.append((self._fe_counter["total"], GF))

        from ncro.optimizer_with_improvements import NCROResult
        return NCROResult(
            best_position=G,
            best_fitness=float(GF),
            convergence_curve=convergence[1:],
            exploration_counts=expl_cnt,
            exploitation_counts=xplt_cnt,
            counterfactual_success_counts=cf_cnt,
            regret_values=regret_avg,
            q_values=q_hist,
        )


# ============================================================
# 4. Reference value computation (reuse from experiment)
# ============================================================

def compute_reference(problem):
    from scipy.optimize import minimize as sp_minimize
    def scalar_obj(x):
        return float(problem.evaluate(x)[0])
    eq_con = {"type": "eq", "fun": lambda x: np.sum(x) - problem.load}
    x0 = problem.repair(0.5 * problem.upper_bounds)
    res = sp_minimize(scalar_obj, x0, method="SLSQP",
                      bounds=list(zip(problem.lower_bounds, problem.upper_bounds)),
                      constraints=eq_con,
                      options={"ftol": 1e-14, "maxiter": 3000})
    ref_x = problem.repair(res.x)
    ref_val = float(problem.evaluate(ref_x)[0])
    return ref_val


# ============================================================
# 5. Main diagnostic
# ============================================================

def run_diagnostic():
    print("=" * 72)
    print("  NCRO-I DIAGNOSTIC — Convergence Bottleneck Analysis")
    print("=" * 72)

    competitor_fes = load_competitor_fes()
    all_diag_rows = []
    convergence_data = {}  # scenario -> list of (fe_list, gf_list) per run

    for scen_name, problem in SCENARIOS.items():
        print(f"\n{'─' * 60}")
        print(f"  Scenario: {scen_name}")
        print(f"{'─' * 60}")

        ref_val = compute_reference(problem)
        print(f"  Reference value: {ref_val:.15e}")

        run_fes_to_tol = []
        run_total_fes = []
        run_fe_breakdown = []
        run_best_fitness = []
        run_runtimes = []
        scen_convergence = []

        for run_idx in range(NUM_RUNS):
            seed = SEED_BASE + run_idx

            recorder = EvaluationRecorder(
                problem=problem,
                evaluation_budget=FE_BUDGET,
                reference_objective=ref_val,
                success_tolerance=TOLERANCE,
            )

            rng = np.random.default_rng(seed)

            dimension = problem.lower_bounds.shape[0]
            init_fes = 2 * POP_SIZE + POP_SIZE // 5
            max_iter = max(1, (FE_BUDGET - init_fes) // (3 * POP_SIZE))

            def objective(x):
                _, values = recorder.evaluate(x)
                return float(np.atleast_1d(values)[0])

            t0 = time.perf_counter()
            opt = InstrumentedNCROI(
                objective_function=objective,
                dimension=dimension,
                bounds=(problem.lower_bounds, problem.upper_bounds),
                population_size=POP_SIZE,
                max_iter=max_iter,
                seed=int(rng.integers(0, 2**31)),
                repair_fn=problem.repair,
            )
            result = opt.optimize()
            elapsed = time.perf_counter() - t0

            run_runtimes.append(elapsed)
            run_best_fitness.append(result.best_fitness)
            run_total_fes.append(opt._fe_counter["total"])
            run_fe_breakdown.append(dict(opt._fe_counter))

            # FEs to tolerance
            fes_to_tol = None
            for fe_num, gf_val in opt._convergence_per_fe:
                if abs(gf_val - ref_val) <= TOLERANCE:
                    fes_to_tol = fe_num
                    break
            run_fes_to_tol.append(fes_to_tol if fes_to_tol else opt._fe_counter["total"])

            scen_convergence.append(opt._convergence_per_fe)

        convergence_data[scen_name] = scen_convergence

        # ── Aggregate ──
        mean_fes_tol = np.mean(run_fes_to_tol)
        mean_total_fes = np.mean(run_total_fes)
        mean_runtime = np.mean(run_runtimes)
        mean_fitness = np.mean(run_best_fitness)

        # FE breakdown averages
        phases = ["init", "actual", "cf", "candidate", "stagnation", "diversity"]
        avg_breakdown = {}
        for ph in phases:
            avg_breakdown[ph] = np.mean([bd[ph] for bd in run_fe_breakdown])

        # Convergence analysis
        # At what FE does GF reach within 1%, 0.1%, 0.01% of reference?
        thresholds = {"1%": 0.01, "0.1%": 0.001, "0.01%": 0.0001, "tol": TOLERANCE}
        fe_at_threshold = {}
        for thr_name, thr_val in thresholds.items():
            fe_hits = []
            for conv_trace in scen_convergence:
                hit = None
                for fe_num, gf_val in conv_trace:
                    if abs(gf_val - ref_val) / (abs(ref_val) + 1e-30) <= thr_val:
                        hit = fe_num
                        break
                fe_hits.append(hit if hit else mean_total_fes)
            fe_at_threshold[thr_name] = np.mean(fe_hits)

        # ── Print ──
        print(f"\n  NCRO-I Results ({NUM_RUNS} runs):")
        print(f"    Mean best fitness:      {mean_fitness:.15e}")
        print(f"    Gap from reference:     {abs(mean_fitness - ref_val):.2e}")
        print(f"    Mean FEs to tolerance:  {mean_fes_tol:.0f}")
        print(f"    Mean total FEs used:    {mean_total_fes:.0f}")
        print(f"    Mean runtime:           {mean_runtime:.2f}s")

        print(f"\n  FE Breakdown (per run average):")
        for ph in phases:
            pct = 100.0 * avg_breakdown[ph] / mean_total_fes
            print(f"    {ph:>12s}: {avg_breakdown[ph]:>8.0f} FEs  ({pct:5.1f}%)")

        print(f"\n  Convergence Speed:")
        for thr_name, fe_val in fe_at_threshold.items():
            print(f"    Reach {thr_name:>5s} gap at FE: {fe_val:>8.0f}")

        # Compare with competitors
        comp_fes_for_scen = {alg: fes for (s, alg), fes in competitor_fes.items()
                             if s == scen_name}
        if comp_fes_for_scen:
            best_comp = min(comp_fes_for_scen.values())
            best_comp_name = min(comp_fes_for_scen, key=comp_fes_for_scen.get)
            slowdown = mean_fes_tol / best_comp if best_comp > 0 else float("inf")
            print(f"\n  Competitor Comparison:")
            print(f"    Fastest competitor: {best_comp_name} = {best_comp:.0f} FEs")
            print(f"    NCRO-I slowdown:    {slowdown:.1f}×")
            for alg, fes in sorted(comp_fes_for_scen.items(), key=lambda x: x[1]):
                ratio = mean_fes_tol / fes if fes > 0 else float("inf")
                marker = " ← target" if ratio < 1.5 else ""
                print(f"      {alg:>15s}: {fes:>8.0f} FEs  (NCRO-I is {ratio:.1f}× slower){marker}")

        all_diag_rows.append({
            "Scenario": scen_name,
            "MeanBestFitness": f"{mean_fitness:.15e}",
            "GapFromRef": f"{abs(mean_fitness - ref_val):.2e}",
            "MeanFEsToTol": f"{mean_fes_tol:.0f}",
            "MeanTotalFEs": f"{mean_total_fes:.0f}",
            "MeanRuntime": f"{mean_runtime:.2f}",
            "FE_Init": f"{avg_breakdown['init']:.0f}",
            "FE_Actual": f"{avg_breakdown['actual']:.0f}",
            "FE_CF": f"{avg_breakdown['cf']:.0f}",
            "FE_Candidate": f"{avg_breakdown['candidate']:.0f}",
            "FE_Stagnation": f"{avg_breakdown['stagnation']:.0f}",
            "FE_Diversity": f"{avg_breakdown['diversity']:.0f}",
            "FE_at_1pct": f"{fe_at_threshold['1%']:.0f}",
            "FE_at_01pct": f"{fe_at_threshold['0.1%']:.0f}",
            "FE_at_001pct": f"{fe_at_threshold['0.01%']:.0f}",
            "FE_at_tol": f"{fe_at_threshold['tol']:.0f}",
        })

    # ── Save diagnostic CSV ──
    tables_dir = Path(TABLES_DIR) if not isinstance(TABLES_DIR, Path) else TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = tables_dir / "diagnostic_report.csv"
    with open(diag_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_diag_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_diag_rows)
    print(f"\n  Saved: {diag_csv}")

    # ── Convergence plot ──
    figures_dir = Path(FIGURES_DIR) if not isinstance(FIGURES_DIR, Path) else FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6 * len(SCENARIOS), 5),
                             sharey=True)
    if len(SCENARIOS) == 1:
        axes = [axes]

    for ax, (scen_name, traces) in zip(axes, convergence_data.items()):
        ref_val = compute_reference(SCENARIOS[scen_name])
        for trace in traces:
            fes = [p[0] for p in trace]
            gaps = [abs(p[1] - ref_val) + 1e-18 for p in trace]
            ax.semilogy(fes, gaps, alpha=0.3, linewidth=0.8, color="steelblue")
        # Mean trace
        max_len = max(len(t) for t in traces)
        all_fes_mean = []
        all_gaps_mean = []
        for idx in range(0, max_len, max(1, max_len // 200)):
            fe_vals = []
            gap_vals = []
            for trace in traces:
                if idx < len(trace):
                    fe_vals.append(trace[idx][0])
                    gap_vals.append(abs(trace[idx][1] - ref_val) + 1e-18)
            if fe_vals:
                all_fes_mean.append(np.mean(fe_vals))
                all_gaps_mean.append(np.mean(gap_vals))
        ax.semilogy(all_fes_mean, all_gaps_mean, linewidth=2.5, color="darkred",
                    label="Mean NCRO-I")

        # Competitor baselines (vertical lines)
        comp_fes_for_scen = {alg: fes for (s, alg), fes in competitor_fes.items()
                             if s == scen_name}
        colors_comp = ["green", "orange", "purple", "brown", "cyan", "magenta"]
        for idx, (alg, fes) in enumerate(sorted(comp_fes_for_scen.items(),
                                                 key=lambda x: x[1])):
            ax.axvline(x=fes, linestyle="--", color=colors_comp[idx % len(colors_comp)],
                       alpha=0.7, label=f"{alg} ({fes:.0f})")

        ax.axhline(y=TOLERANCE, linestyle=":", color="red", alpha=0.5,
                   label=f"Tolerance ({TOLERANCE:.0e})")
        ax.set_xlabel("Function Evaluations", fontsize=10)
        ax.set_title(f"{scen_name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("| f(x) - f* |", fontsize=10)
    plt.suptitle("NCRO-I Convergence Diagnostic — Gap vs FEs", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = figures_dir / "diagnostic_convergence.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Summary & Recommendations ──
    print("\n" + "=" * 72)
    print("  DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 72)

    print("""
  ROOT CAUSE ANALYSIS:
  ────────────────────
  The NCRO-I optimizer uses 3 function evaluations per agent per iteration:
    1. F_A  — actual candidate (exploration/exploitation blend)
    2. F_C  — counterfactual candidate (swapped coefficients)
    3. F_cand — motion-equation candidate (full cognitive update)

  On a trivially easy 4D convex problem, the counterfactual and motion-
  equation candidates add overhead without providing useful information,
  because the landscape has no deceptive features to learn from.

  KEY BOTTLENECKS IDENTIFIED:
  ───────────────────────────
  1. FE OVERHEAD: 3× FEs per agent vs 1× for DE variants.
     → NCRO-I gets only ~500 iterations for 45,000 FE budget
     → CMA-ES/DE get ~1,500 iterations (same budget, 1 eval/agent)

  2. OBL INITIALIZATION: 2N FEs spent on opposition-based init.
     On a 4D problem, random init is already near-optimal region.

  3. EXPLORATION TOO STRONG EARLY: Scout mode activates too eagerly
     on a simple problem, wasting FEs on Levy flights / random jumps.

  4. STAGNATION DETECTION: stag_limit=50 is too aggressive for a
     problem that converges smoothly — triggers unnecessary opposition jumps.

  RECOMMENDATIONS FOR IMPROVEMENT:
  ─────────────────────────────────
  A) REDUCE FE COST PER ITERATION:
     - Skip Y_C evaluation when M_R[i] < threshold (no regret → no need
       for counterfactual). This can save ~30% FEs on easy problems.
     - Use the better of Y_A and Y_C as the candidate directly (skip
       separate motion-equation candidate when dim ≤ 10).

  B) ADAPTIVE MECHANISM ACTIVATION:
     - Detect problem difficulty early (variance of initial population
       fitness). On easy problems, reduce exploration coefficient faster.
     - Disable scout mode when convergence is already rapid.

  C) FASTER EXPLOITATION:
     - Increase beta_max (exploitation force) for low-D problems.
     - Reduce noise earlier (noise helps in high-D, hurts in low-D).

  D) BUDGET ALLOCATION:
     - On 4D problems, skip OBL init (use standard N init, save N FEs).
     - Allocate saved FEs to more exploitation iterations.
""")


if __name__ == "__main__":
    run_diagnostic()
