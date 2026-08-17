"""
SHADE — Success-History based Adaptive Differential Evolution.

Adapts F and CR using a historical memory of successful parameter values.
References:
  - Tanabe, R. & Fukunaga, A.S. (2013). "Success-History Based Parameter
    Adaptation for Differential Evolution." IEEE CEC, pp. 71-78.
"""

import numpy as np
from typing import Callable
from .base import OptResult


class SHADEOptimizer:
    """Success-History based Adaptive Differential Evolution (SHADE)."""

    def __init__(
        self,
        objective_function: Callable,
        dimension: int,
        bounds: tuple,
        population_size: int = 30,
        max_iter: int = 1000,
        H: int = 5,  # Historical memory size
        seed: int | None = None,
        **kwargs,
    ):
        self.func = objective_function
        self.D = dimension
        self.L, self.U = bounds
        self.N = population_size
        self.T = max_iter
        self.H = H
        self.seed = seed

    def optimize(self) -> OptResult:
        rng = np.random.default_rng(self.seed)
        N, D, T = self.N, self.D, self.T
        L, U = self.L, self.U
        H = self.H

        # Initialize population
        X = rng.uniform(L, U, size=(N, D))
        fitness = np.array([self.func(X[i]) for i in range(N)])

        # External archive
        archive = []
        archive_max = N

        # Historical memory for F and CR
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0  # Memory index

        g_idx = int(np.argmin(fitness))
        g = X[g_idx].copy()
        g_fit = fitness[g_idx]

        convergence = np.zeros(T)

        for t in range(T):
            S_F = []    # Successful F values
            S_CR = []   # Successful CR values
            delta_f = [] # Fitness improvements

            for i in range(N):
                # Generate F_i from Cauchy distribution
                r_idx = rng.integers(H)
                F_i = -1.0
                attempts = 0
                while F_i <= 0 and attempts < 100:
                    F_i = M_F[r_idx] + 0.1 * rng.standard_cauchy()
                    attempts += 1
                if F_i <= 0:
                    F_i = 0.01
                F_i = min(F_i, 1.0)

                # Generate CR_i from normal distribution
                CR_i = np.clip(rng.normal(M_CR[r_idx], 0.1), 0.0, 1.0)

                # Mutation: DE/current-to-pbest/1
                # Select p-best (top p% of population)
                p = max(2, int(0.1 * N))
                sorted_idx = np.argsort(fitness)
                pbest_idx = sorted_idx[rng.integers(p)]

                # Select r1 != i
                candidates = np.delete(np.arange(N), i)
                r1 = rng.choice(candidates)

                # Select r2 from population + archive, r2 != i, r2 != r1
                combined = list(range(N)) + list(range(N, N + len(archive)))
                combined = [c for c in combined if c != i and c != r1]
                r2_idx = rng.choice(combined)

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

            # Update memory
            if len(S_F) > 0:
                delta_f = np.array(delta_f)
                weights = delta_f / (np.sum(delta_f) + 1e-30)

                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)

                # Weighted Lehmer mean for F
                M_F[k] = float(np.sum(weights * S_F_arr ** 2) / (np.sum(weights * S_F_arr) + 1e-30))

                # Weighted arithmetic mean for CR
                M_CR[k] = float(np.sum(weights * S_CR_arr))

                k = (k + 1) % H

            g_idx = int(np.argmin(fitness))
            if fitness[g_idx] < g_fit:
                g = X[g_idx].copy()
                g_fit = fitness[g_idx]

            convergence[t] = g_fit

        return OptResult(
            best_position=g,
            best_fitness=g_fit,
            convergence_curve=convergence,
        )
