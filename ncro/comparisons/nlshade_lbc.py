"""
NL-SHADE-LBC — NL-SHADE with Linear parameter adaptation Bias Change.

Winner of CEC 2022 single-objective bound-constrained optimization
competition. Remains a reference benchmark algorithm through 2025+.

Key innovations over L-SHADE:
  - Linear bias change in CR memory initialization
  - Improved archive management
  - Generation-dependent parameter adaptation bias

References:
  - Stanovov, V. & Akhmedova, S. (2022). "NL-SHADE-LBC algorithm with linear
    parameter adaptation bias change for CEC 2022 Numerical Optimization."
    IEEE CEC 2022 Competition.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class NLSHADELBCOptimizer:
    """
    NL-SHADE-LBC: CEC 2022 winner.

    Extends L-SHADE with:
      - Non-linear population size reduction
      - Linear bias change in parameter memory
      - Improved archive utilization
    """

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_iter: int = 1000,
        H: int = 5,
        N_min: int = 4,
        seed: int | None = None,
        **kwargs,
    ):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N_init = population_size
        self.T = max_iter
        self.H = H
        self.N_min = N_min
        self.seed = seed
        self.population_size = population_size
        self.max_iter = max_iter

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        D, T = self.D, self.T
        L, U = self.L, self.U
        H = self.H
        N = self.N_init

        X = rng.uniform(L, U, size=(N, D))
        fitness = np.array([self.func(X[i]) for i in range(N)])

        archive = []
        archive_max = int(2.1 * N)  # NL-SHADE-LBC uses larger archive

        # Historical memory with linear bias change
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.9)  # Start with higher CR (bias change)
        k = 0

        g_idx = int(np.argmin(fitness))
        g = X[g_idx].copy()
        g_fit = fitness[g_idx]

        convergence = np.zeros(T)

        for t in range(T):
            tau = (t + 1) / T
            S_F = []
            S_CR = []
            delta_f = []

            # Linear bias change: CR bias decreases over time
            cr_bias = 0.9 * (1.0 - tau) + 0.1 * tau  # From 0.9 to 0.1

            for i in range(N):
                r_idx = rng.integers(H)

                # Generate F_i from Cauchy
                F_i = -1.0
                attempts = 0
                while F_i <= 0 and attempts < 100:
                    F_i = M_F[r_idx] + 0.1 * rng.standard_cauchy()
                    attempts += 1
                if F_i <= 0:
                    F_i = 0.01
                F_i = min(F_i, 1.0)

                # Generate CR_i with linear bias
                if M_CR[r_idx] < 0:
                    CR_i = 0.0
                else:
                    # Apply linear bias change
                    cr_center = M_CR[r_idx] * cr_bias + 0.5 * (1 - cr_bias)
                    CR_i = np.clip(rng.normal(cr_center, 0.1), 0.0, 1.0)

                # DE/current-to-pbest/1 with dynamic p
                p_ratio = max(2 / N, 0.25 * (1 - tau) + 0.05 * tau)
                p = max(2, int(p_ratio * N))
                sorted_idx = np.argsort(fitness[:N])
                pbest_idx = sorted_idx[rng.integers(p)]

                candidates = np.delete(np.arange(N), i)
                r1 = rng.choice(candidates)

                n_archive = len(archive)
                combined_size = N + n_archive
                combined_cands = [c for c in range(combined_size) if c != i and c != r1]
                if len(combined_cands) == 0:
                    combined_cands = [c for c in range(combined_size) if c != i]
                r2_idx = rng.choice(combined_cands)

                if r2_idx < N:
                    x_r2 = X[r2_idx]
                else:
                    x_r2 = archive[r2_idx - N]

                mutant = X[i] + F_i * (X[pbest_idx] - X[i]) + F_i * (X[r1] - x_r2)

                # Boundary repair: midpoint strategy
                for j in range(D):
                    if mutant[j] < L:
                        mutant[j] = (L + X[i][j]) / 2.0
                    elif mutant[j] > U:
                        mutant[j] = (U + X[i][j]) / 2.0

                # Binomial crossover
                trial = X[i].copy()
                j_rand = rng.integers(D)
                for j in range(D):
                    if rng.random() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                f_trial = self.func(trial)
                if f_trial <= fitness[i]:
                    if f_trial < fitness[i]:
                        archive.append(X[i].copy())
                        S_F.append(F_i)
                        S_CR.append(CR_i)
                        delta_f.append(abs(fitness[i] - f_trial))
                    X[i] = trial
                    fitness[i] = f_trial

            # Trim archive
            while len(archive) > archive_max:
                archive.pop(rng.integers(len(archive)))

            # Update memory with weighted Lehmer mean
            if len(S_F) > 0:
                delta_f_arr = np.array(delta_f)
                weights = delta_f_arr / (np.sum(delta_f_arr) + 1e-30)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)

                M_F[k] = float(np.sum(weights * S_F_arr ** 2) / (np.sum(weights * S_F_arr) + 1e-30))

                if np.max(S_CR_arr) == 0:
                    M_CR[k] = -1.0
                else:
                    M_CR[k] = float(np.sum(weights * S_CR_arr ** 2) / (np.sum(weights * S_CR_arr) + 1e-30))

                k = (k + 1) % H

            g_idx_t = int(np.argmin(fitness[:N]))
            if fitness[g_idx_t] < g_fit:
                g = X[g_idx_t].copy()
                g_fit = fitness[g_idx_t]

            convergence[t] = g_fit

            # Non-linear population size reduction (quadratic)
            ratio = ((t + 1) / T) ** 2  # Non-linear (quadratic)
            N_new = max(self.N_min, round(self.N_init - (self.N_init - self.N_min) * ratio))

            if N_new < N:
                sorted_idx = np.argsort(fitness[:N])
                keep = sorted_idx[:N_new]
                X = X[keep]
                fitness = fitness[keep]
                N = N_new
                archive_max = int(2.1 * N)
                while len(archive) > archive_max:
                    archive.pop(rng.integers(len(archive)))

        return OptResult(best_position=g, best_fitness=g_fit, convergence_curve=convergence)
