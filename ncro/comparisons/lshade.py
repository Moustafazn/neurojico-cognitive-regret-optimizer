"""
L-SHADE — SHADE with Linear Population Size Reduction.

Extends SHADE with linear population size reduction over generations.
This is a CEC competition-winning algorithm and represents state-of-the-art
adaptive differential evolution as of 2026.

References:
  - Tanabe, R. & Fukunaga, A.S. (2014). "Improving the Search Performance of
    SHADE Using Linear Population Size Reduction." IEEE CEC, pp. 1658-1665.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class LSHADEOptimizer:
    """
    L-SHADE: SHADE with Linear Population Size Reduction.

    State-of-the-art adaptive DE. Won CEC 2014 and variants continue
    to dominate CEC competitions through 2024+.
    """

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_iter: int = 1000,
        H: int = 5,          # Historical memory size
        N_min: int = 4,       # Minimum population size
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
        # For interface compatibility
        self.population_size = population_size
        self.max_iter = max_iter

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        D, T = self.D, self.T
        L, U = self.L, self.U
        H = self.H
        N = self.N_init

        # Initialize population
        X = rng.uniform(L, U, size=(N, D))
        fitness = np.array([self.func(X[i]) for i in range(N)])

        # External archive
        archive = []
        archive_max = N

        # Historical memory
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0

        g_idx = int(np.argmin(fitness))
        g = X[g_idx].copy()
        g_fit = fitness[g_idx]

        convergence = np.zeros(T)

        for t in range(T):
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # Generate F_i
                r_idx = rng.integers(H)
                F_i = -1.0
                attempts = 0
                while F_i <= 0 and attempts < 100:
                    F_i = M_F[r_idx] + 0.1 * rng.standard_cauchy()
                    attempts += 1
                if F_i <= 0:
                    F_i = 0.01
                F_i = min(F_i, 1.0)

                # Generate CR_i
                if M_CR[r_idx] < 0:
                    CR_i = 0.0
                else:
                    CR_i = np.clip(rng.normal(M_CR[r_idx], 0.1), 0.0, 1.0)

                # DE/current-to-pbest/1 mutation
                p = max(2, int(max(0.05, 0.2 - 0.15 * t / T) * N))
                sorted_idx = np.argsort(fitness[:N])
                pbest_idx = sorted_idx[rng.integers(p)]

                candidates = np.delete(np.arange(N), i)
                r1 = rng.choice(candidates)

                # r2 from population + archive
                n_archive = len(archive)
                combined_size = N + n_archive
                combined_candidates = [c for c in range(combined_size) if c != i and c != r1]
                if len(combined_candidates) == 0:
                    combined_candidates = [c for c in range(combined_size) if c != i]
                r2_idx = rng.choice(combined_candidates)

                if r2_idx < N:
                    x_r2 = X[r2_idx]
                else:
                    x_r2 = archive[r2_idx - N]

                mutant = X[i] + F_i * (X[pbest_idx] - X[i]) + F_i * (X[r1] - x_r2)
                mutant = np.clip(mutant, L, U)

                # Binomial crossover
                trial = X[i].copy()
                j_rand = rng.integers(D)
                for j in range(D):
                    if rng.random() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                # Selection
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

            # Update historical memory
            if len(S_F) > 0:
                delta_f_arr = np.array(delta_f)
                weights = delta_f_arr / (np.sum(delta_f_arr) + 1e-30)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)

                # Weighted Lehmer mean for F
                M_F[k] = float(np.sum(weights * S_F_arr ** 2) / (np.sum(weights * S_F_arr) + 1e-30))

                # Weighted mean for CR (set to -1 if all CR=0)
                if np.max(S_CR_arr) == 0:
                    M_CR[k] = -1.0
                else:
                    M_CR[k] = float(np.sum(weights * S_CR_arr))

                k = (k + 1) % H

            # Update global best
            g_idx_t = int(np.argmin(fitness[:N]))
            if fitness[g_idx_t] < g_fit:
                g = X[g_idx_t].copy()
                g_fit = fitness[g_idx_t]

            convergence[t] = g_fit

            # Linear Population Size Reduction
            N_new = max(
                self.N_min,
                round(self.N_init + (self.N_min - self.N_init) * (t + 1) / T),
            )

            if N_new < N:
                # Keep the best N_new individuals
                sorted_idx = np.argsort(fitness[:N])
                keep = sorted_idx[:N_new]
                X = X[keep]
                fitness = fitness[keep]
                N = N_new
                archive_max = N

                # Trim archive
                while len(archive) > archive_max:
                    archive.pop(rng.integers(len(archive)))

        return OptResult(
            best_position=g,
            best_fitness=g_fit,
            convergence_curve=convergence,
        )
